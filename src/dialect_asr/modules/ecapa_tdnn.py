"""ECAPA-TDNN dialect identification (DID) branch for encoder hidden states.

Reference: Desplanques et al., "ECAPA-TDNN: Emphasized Channel Attention,
Propagation and Aggregation in TDNN Based Speaker Verification" (2020). The
same backbone (SE-Res2Blocks + multilayer feature aggregation + attentive
statistics pooling) is reused here as a dialect classifier instead of a
speaker classifier.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SqueezeExcitation(nn.Module):
    """Rescale channels of a ``[B, C, T]`` map by a per-utterance gate."""

    def __init__(self, channels: int, bottleneck_channels: int = 128) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels phải > 0")
        if bottleneck_channels <= 0:
            raise ValueError("bottleneck_channels phải > 0")

        self.excitation = nn.Sequential(
            nn.Linear(channels, bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor, frame_mask: Tensor | None = None) -> Tensor:
        """Gate ``x`` with statistics pooled over valid frames.

        Args:
            x: Feature map with shape ``[B, C, T_frame]``.
            frame_mask: Optional valid-frame mask ``[B, T_frame]``.
        """
        if frame_mask is None:
            pooled = x.mean(dim=2)  # [B, C, T_frame] -> [B, C].
        else:
            # [B, T_frame] -> [B, 1, T_frame] for broadcasting over C.
            mask = frame_mask.unsqueeze(1).to(dtype=x.dtype)
            # [B, C, T_frame] * [B, 1, T_frame] -> sum over T_frame -> [B, C].
            summed = (x * mask).sum(dim=2)
            valid_counts = mask.sum(dim=2).clamp_min(1.0)  # [B, 1].
            pooled = summed / valid_counts  # [B, C].

        gate = self.excitation(pooled).unsqueeze(2)  # [B, C] -> [B, C, 1].
        return x * gate  # [B, C, T_frame] * [B, C, 1] -> [B, C, T_frame].


class Res2NetLayer(nn.Module):
    """Multi-scale dilated convolution (Res2Net) used inside a SE-Res2Block."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        scale: int = 8,
    ) -> None:
        super().__init__()
        if channels % scale != 0:
            raise ValueError("channels phải chia hết cho scale")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size phải là số lẻ để giữ nguyên chiều T")

        self.scale = scale
        group_channels = channels // scale
        padding = dilation * (kernel_size - 1) // 2
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    group_channels,
                    group_channels,
                    kernel_size,
                    dilation=dilation,
                    padding=padding,
                )
                for _ in range(scale - 1)
            ]
        )
        self.bns = nn.ModuleList([nn.BatchNorm1d(group_channels) for _ in range(scale - 1)])

    def forward(self, x: Tensor) -> Tensor:
        """Apply hierarchical residual convolutions across channel groups."""
        # [B, C, T_frame] -> `scale` groups, each [B, C/scale, T_frame].
        groups = torch.chunk(x, self.scale, dim=1)

        outputs = [groups[0]]  # First group passes through unchanged.
        previous = None
        for index, (conv, bn) in enumerate(zip(self.convs, self.bns), start=1):
            group = groups[index]
            current = group if previous is None else group + previous
            current = F.relu(bn(conv(current)))  # [B, C/scale, T_frame].
            outputs.append(current)
            previous = current

        # scale * [B, C/scale, T_frame] -> [B, C, T_frame].
        return torch.cat(outputs, dim=1)


class SERes2Block(nn.Module):
    """Pointwise conv -> Res2Net -> pointwise conv -> SE gate -> residual."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        res2net_scale: int = 8,
        se_bottleneck_channels: int = 128,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.res2net = Res2NetLayer(channels, kernel_size, dilation, res2net_scale)
        self.bn2 = nn.BatchNorm1d(channels)
        self.conv3 = nn.Conv1d(channels, channels, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(channels)
        self.se = SqueezeExcitation(channels, se_bottleneck_channels)

    def forward(self, x: Tensor, frame_mask: Tensor | None = None) -> Tensor:
        """Return the SE-Res2Block output with the same shape as ``x``."""
        residual = x  # [B, C, T_frame].
        out = F.relu(self.bn1(self.conv1(x)))  # [B, C, T_frame].
        out = F.relu(self.bn2(self.res2net(out)))  # [B, C, T_frame].
        out = self.bn3(self.conv3(out))  # [B, C, T_frame].
        out = self.se(out, frame_mask)  # [B, C, T_frame].
        return out + residual  # [B, C, T_frame].


class AttentiveStatisticsPooling(nn.Module):
    """Attention-weighted mean/std pooling with global context (ECAPA-TDNN)."""

    def __init__(self, channels: int, attention_channels: int = 128) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels * 3, attention_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(attention_channels),
            nn.Tanh(),
            nn.Conv1d(attention_channels, channels, kernel_size=1),
        )

    def forward(self, x: Tensor, frame_mask: Tensor | None = None) -> Tensor:
        """Pool ``[B, C, T_frame]`` into ``[B, 2*C]`` (weighted mean ++ std)."""
        batch_size, _, num_frames = x.shape
        if frame_mask is None:
            mask = x.new_ones(batch_size, 1, num_frames)
        else:
            mask = frame_mask.unsqueeze(1).to(dtype=x.dtype)  # [B, 1, T_frame].
        valid_counts = mask.sum(dim=2, keepdim=True).clamp_min(1.0)  # [B, 1, 1].

        # Unweighted statistics broadcast as global context for the attention.
        mean = (x * mask).sum(dim=2, keepdim=True) / valid_counts  # [B, C, 1].
        variance = ((x - mean).pow(2) * mask).sum(dim=2, keepdim=True) / valid_counts
        std = variance.clamp_min(1e-8).sqrt()  # [B, C, 1].

        global_context = torch.cat(
            [x, mean.expand(-1, -1, num_frames), std.expand(-1, -1, num_frames)],
            dim=1,
        )  # [B, 3*C, T_frame].
        scores = self.attention(global_context)  # [B, C, T_frame].
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=2)  # [B, C, T_frame].

        weighted_mean = (x * weights).sum(dim=2)  # [B, C].
        weighted_variance = (x.pow(2) * weights).sum(dim=2) - weighted_mean.pow(2)
        weighted_std = weighted_variance.clamp_min(1e-8).sqrt()  # [B, C].

        return torch.cat([weighted_mean, weighted_std], dim=1)  # [B, 2*C].


class ECAPA_TDNN_DID(nn.Module):
    """ECAPA-TDNN dialect classifier over PhoWhisper encoder hidden states.

    Architecture: TDNN stem -> 3x SE-Res2Block (increasing dilation) ->
    multilayer feature aggregation -> attentive statistics pooling ->
    embedding -> linear region classifier. Mirrors the speaker-verification
    ECAPA-TDNN but predicts a dialect region instead of a speaker identity.
    """

    def __init__(
        self,
        hidden_size: int,
        num_regions: int = 3,
        channels: int = 512,
        embedding_size: int = 192,
        kernel_sizes: tuple[int, int, int, int] = (5, 3, 3, 3),
        dilations: tuple[int, int, int, int] = (1, 2, 3, 4),
        res2net_scale: int = 8,
        se_bottleneck_channels: int = 128,
        attention_channels: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size phải > 0")
        if num_regions <= 1:
            raise ValueError("num_regions phải > 1")
        if channels <= 0:
            raise ValueError("channels phải > 0")
        if embedding_size <= 0:
            raise ValueError("embedding_size phải > 0")
        if len(kernel_sizes) != 4 or len(dilations) != 4:
            raise ValueError("kernel_sizes và dilations phải có đúng 4 phần tử")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout phải nằm trong khoảng [0, 1)")

        self.hidden_size = hidden_size
        self.num_regions = num_regions

        stem_kernel, *block_kernels = kernel_sizes
        stem_dilation, *block_dilations = dilations
        self.stem = nn.Sequential(
            nn.Conv1d(
                hidden_size,
                channels,
                stem_kernel,
                dilation=stem_dilation,
                padding=stem_dilation * (stem_kernel - 1) // 2,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(channels),
        )
        self.blocks = nn.ModuleList(
            [
                SERes2Block(
                    channels,
                    kernel_size,
                    dilation,
                    res2net_scale,
                    se_bottleneck_channels,
                )
                for kernel_size, dilation in zip(block_kernels, block_dilations)
            ]
        )
        # Concatenated outputs of the 3 SE-Res2Blocks -> back down to `channels`.
        self.feature_aggregation = nn.Conv1d(channels * len(self.blocks), channels, kernel_size=1)
        self.pooling = AttentiveStatisticsPooling(channels, attention_channels)
        self.pooling_norm = nn.BatchNorm1d(channels * 2)
        self.embedding = nn.Linear(channels * 2, embedding_size)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(embedding_size),
            nn.Dropout(dropout),
            nn.Linear(embedding_size, num_regions),
        )

    def forward(
        self,
        hidden_states: Tensor,
        feature_attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(region_logits [B, R], embedding [B, embedding_size])``.

        Args:
            hidden_states: Encoder features with shape ``[B, T_frame, H]``.
            feature_attention_mask: Optional valid-frame mask ``[B, T_frame]``.
        """
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states phải có shape [B, T_frame, H]")
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                "Hidden size không khớp: "
                f"cần {self.hidden_size}, nhận {hidden_states.shape[-1]}"
            )
        if feature_attention_mask is not None:
            expected_mask_shape = hidden_states.shape[:2]
            if feature_attention_mask.shape != expected_mask_shape:
                raise ValueError(
                    "feature_attention_mask phải có shape [B, T_frame]; "
                    f"cần {tuple(expected_mask_shape)}, "
                    f"nhận {tuple(feature_attention_mask.shape)}"
                )

        # [B, T_frame, H] -> [B, H, T_frame] for Conv1d (channels-first).
        x = hidden_states.transpose(1, 2)
        x = self.stem(x)  # [B, H, T_frame] -> [B, channels, T_frame].

        # Every SE-Res2Block preserves T_frame (same padding), so the frame
        # mask keeps aligning across the stack and through pooling.
        block_outputs = []
        for block in self.blocks:
            x = block(x, feature_attention_mask)  # [B, channels, T_frame].
            block_outputs.append(x)

        # 3 * [B, channels, T_frame] -> [B, 3*channels, T_frame].
        aggregated = torch.cat(block_outputs, dim=1)
        aggregated = F.relu(self.feature_aggregation(aggregated))  # [B, channels, T_frame].

        pooled = self.pooling(aggregated, feature_attention_mask)  # [B, 2*channels].
        pooled = self.pooling_norm(pooled)  # [B, 2*channels].
        embedding = self.embedding(pooled)  # [B, embedding_size].
        logits = self.classifier(embedding)  # [B, num_regions].

        return logits, embedding
