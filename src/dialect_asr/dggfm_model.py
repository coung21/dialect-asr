"""Wav2Vec2 CTC with a mid-encoder dialect branch and DG-GFM layers."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Any, Self

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.integrations.fsdp import is_fsdp_managed_module
from transformers.masking_utils import create_bidirectional_mask
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutput
from transformers.models.wav2vec2.configuration_wav2vec2 import Wav2Vec2Config

from .base_model import AbstractWav2Vec2CTC
from .modules import DGGFM, DialectBranch, SoftDialectEmbedding


@dataclass
class DGGFMCTCOutput(CausalLMOutput):
    """CTC output augmented with dialect and modulation diagnostics."""

    ctc_loss: Tensor | None = None
    dialect_loss: Tensor | None = None
    dialect_logits: Tensor | None = None
    dialect_pooled: Tensor | None = None
    dialect_embedding: Tensor | None = None
    dialect_posterior: Tensor | None = None
    gate_statistics: dict[int, dict[str, Tensor]] | None = None


class DGGFMWav2Vec2CTC(AbstractWav2Vec2CTC):
    """Branch after block 6 and apply DG-GFM after blocks 6, 8, 10 and 12."""

    DEFAULT_BRANCH_BLOCK = 6
    DEFAULT_FUSION_BLOCKS = (6, 8, 10, 12)

    def __init__(self, config: Any) -> None:
        super().__init__(config)

        self.branch_block = int(
            getattr(config, "dggfm_branch_block", self.DEFAULT_BRANCH_BLOCK)
        )
        self.fusion_blocks = tuple(
            int(block)
            for block in getattr(
                config,
                "dggfm_fusion_blocks",
                self.DEFAULT_FUSION_BLOCKS,
            )
        )
        self.num_regions = int(getattr(config, "num_regions", 3))
        self.dialect_bottleneck_size = int(
            getattr(config, "dialect_bottleneck_size", 256)
        )
        self.dialect_dim = int(getattr(config, "dialect_dim", 64))
        self.gate_hidden_dim = int(getattr(config, "dggfm_gate_hidden_dim", 256))
        self.dialect_temperature = float(
            getattr(config, "dialect_temperature", 1.0)
        )
        self.dialect_loss_weight = float(
            getattr(config, "dialect_loss_weight", 0.1)
        )
        self.dialect_dropout = float(getattr(config, "dialect_dropout", 0.1))

        self._validate_architecture_config(config)
        self._save_architecture_config(config)
        ignored_at_inference = set(
            getattr(config, "keys_to_ignore_at_inference", []) or []
        )
        ignored_at_inference.update(
            {
                "ctc_loss",
                "dialect_loss",
                "dialect_logits",
                "dialect_pooled",
                "dialect_embedding",
                "dialect_posterior",
                "gate_statistics",
            }
        )
        config.keys_to_ignore_at_inference = sorted(ignored_at_inference)

        self.dialect_branch = DialectBranch(
            hidden_size=config.hidden_size,
            num_regions=self.num_regions,
            bottleneck_size=self.dialect_bottleneck_size,
            dropout=self.dialect_dropout,
        )
        self.soft_dialect_embedding = SoftDialectEmbedding(
            num_regions=self.num_regions,
            embedding_dim=self.dialect_dim,
            temperature=self.dialect_temperature,
        )
        self.dggfm_layers = nn.ModuleDict(
            {
                str(block): DGGFM(
                    hidden_size=config.hidden_size,
                    dialect_dim=self.dialect_dim,
                    gate_hidden_dim=self.gate_hidden_dim,
                )
                for block in self.fusion_blocks
            }
        )

    @classmethod
    def architecture_name(cls) -> str:
        """Return the Hydra/registry identifier for this architecture."""
        return "dggfm"

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike[str] | None,
        *model_args: Any,
        config: Wav2Vec2Config | str | PathLike[str] | None = None,
        cache_dir: str | PathLike[str] | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        **kwargs: Any,
    ) -> Self:
        """Load baseline or DGGFM weights without losing identity initialization."""
        if isinstance(config, Wav2Vec2Config):
            loaded_config = config
        else:
            config_source = config if config is not None else pretrained_model_name_or_path
            loaded_config = Wav2Vec2Config.from_pretrained(
                config_source,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
            )
        source_is_dggfm = getattr(loaded_config, "architecture", None) == cls.architecture_name()

        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=loaded_config,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            **kwargs,
        )
        if not source_is_dggfm:
            # Transformers initializes checkpoint-missing tensors after __init__;
            # restore gamma/beta zero and final gate bias -2 for baseline loading.
            for dggfm_layer in model.dggfm_layers.values():
                dggfm_layer.reset_parameters()
        return model

    def _validate_architecture_config(self, config: Any) -> None:
        if bool(getattr(config, "do_stable_layer_norm", False)):
            raise ValueError("DGGFM hiện chỉ hỗ trợ Wav2Vec2 do_stable_layer_norm=false")
        if bool(getattr(config, "add_adapter", False)):
            raise ValueError("DGGFM hiện chưa hỗ trợ Wav2Vec2 adapter layers")
        if not 1 <= self.branch_block <= config.num_hidden_layers:
            raise ValueError(
                "dggfm_branch_block phải nằm trong số Transformer blocks; "
                f"nhận {self.branch_block}/{config.num_hidden_layers}"
            )
        if not self.fusion_blocks:
            raise ValueError("dggfm_fusion_blocks không được rỗng")
        if len(set(self.fusion_blocks)) != len(self.fusion_blocks):
            raise ValueError("dggfm_fusion_blocks không được chứa block trùng nhau")
        invalid_blocks = [
            block
            for block in self.fusion_blocks
            if block < self.branch_block or block > config.num_hidden_layers
        ]
        if invalid_blocks:
            raise ValueError(
                "Mỗi fusion block phải từ branch block đến block cuối; "
                f"không hợp lệ: {invalid_blocks}"
            )
        if self.dialect_loss_weight < 0.0:
            raise ValueError("dialect_loss_weight phải >= 0")

    def _save_architecture_config(self, config: Any) -> None:
        """Persist custom settings so save_pretrained/from_pretrained round-trips."""
        config.architecture = self.architecture_name()
        config.dggfm_branch_block = self.branch_block
        config.dggfm_fusion_blocks = list(self.fusion_blocks)
        config.num_regions = self.num_regions
        config.dialect_bottleneck_size = self.dialect_bottleneck_size
        config.dialect_dim = self.dialect_dim
        config.dggfm_gate_hidden_dim = self.gate_hidden_dim
        config.dialect_temperature = self.dialect_temperature
        config.dialect_loss_weight = self.dialect_loss_weight
        config.dialect_dropout = self.dialect_dropout

    def _forward_dggfm_encoder(
        self,
        hidden_states: Tensor,
        feature_attention_mask: Tensor | None,
        *,
        output_attentions: bool,
        output_hidden_states: bool,
    ) -> tuple[
        BaseModelOutput,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        dict[int, dict[str, Tensor]],
    ]:
        """Run the Transformer stack and inject dialect fusion at selected blocks."""
        encoder = self.wav2vec2.encoder
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if feature_attention_mask is not None:
            expanded_mask = feature_attention_mask.unsqueeze(-1).repeat(
                1,
                1,
                hidden_states.shape[2],
            )
            # [B, T_frame] -> [B, T_frame, 1] -> [B, T_frame, H].
            hidden_states = hidden_states.masked_fill(~expanded_mask, 0.0)
            # Masking preserves hidden state shape [B, T_frame, H].

        encoder_attention_mask = create_bidirectional_mask(
            config=encoder.config,
            inputs_embeds=hidden_states,
            attention_mask=feature_attention_mask,
        )
        # [B, T_frame] -> attention mask broadcastable across attention heads.
        position_embeddings = encoder.pos_conv_embed(hidden_states)
        # [B, T_frame, H] -> positional embeddings [B, T_frame, H].
        hidden_states = hidden_states + position_embeddings.to(hidden_states.device)
        # [B, T_frame, H] + [B, T_frame, H] -> [B, T_frame, H].
        hidden_states = encoder.layer_norm(hidden_states)
        # LayerNorm preserves shape [B, T_frame, H].
        hidden_states = encoder.dropout(hidden_states)
        # Dropout preserves shape [B, T_frame, H].

        synced_gpus = is_deepspeed_zero3_enabled() or is_fsdp_managed_module(encoder)
        dialect_logits: Tensor | None = None
        dialect_pooled: Tensor | None = None
        dialect_embedding: Tensor | None = None
        dialect_posterior: Tensor | None = None
        gate_statistics: dict[int, dict[str, Tensor]] = {}

        for layer_index, layer in enumerate(encoder.layers):
            block_number = layer_index + 1
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            dropout_probability = torch.rand([])
            skip_layer = self.training and dropout_probability < encoder.config.layerdrop
            layer_outputs: tuple[Any, ...] = (None, None)
            if not skip_layer or synced_gpus:
                layer_outputs = layer(
                    hidden_states,  # [B, T_frame, H].
                    attention_mask=encoder_attention_mask,
                    output_attentions=output_attentions,
                )
                hidden_states = layer_outputs[0]
                # One Transformer block preserves shape [B, T_frame, H].

            if block_number == self.branch_block:
                dialect_logits, dialect_pooled = self.dialect_branch(
                    hidden_states,  # [B, T_frame, H].
                    feature_attention_mask,  # [B, T_frame] or None.
                )
                # [B, T_frame, H] -> logits [B, R], pooled [B, H].
                dialect_embedding, dialect_posterior = self.soft_dialect_embedding(
                    dialect_logits  # [B, R].
                )
                # [B, R] -> embedding [B, E], posterior [B, R].

            if block_number in self.fusion_blocks:
                if dialect_embedding is None:
                    raise RuntimeError("DGGFM fusion chạy trước DialectBranch")
                hidden_states, block_statistics = self.dggfm_layers[str(block_number)](
                    hidden_states,  # [B, T_frame, H].
                    dialect_embedding,  # [B, E].
                    feature_attention_mask,  # [B, T_frame] or None.
                )
                # DGGFM residual modulation preserves shape [B, T_frame, H].
                gate_statistics[block_number] = block_statistics

            if output_attentions:
                attention = layer_outputs[1] if not skip_layer else None
                all_self_attentions = all_self_attentions + (attention,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if any(
            value is None
            for value in (
                dialect_logits,
                dialect_pooled,
                dialect_embedding,
                dialect_posterior,
            )
        ):
            raise RuntimeError("DialectBranch không được thực thi trong encoder")

        encoder_output = BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
        return (
            encoder_output,
            dialect_logits,
            dialect_pooled,
            dialect_embedding,
            dialect_posterior,
            gate_statistics,
        )

    def forward(
        self,
        input_values: Tensor,
        attention_mask: Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        labels: Tensor | None = None,
        region_labels: Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[Any, ...] | DGGFMCTCOutput:
        """Compute joint CTC and optional region-classification losses."""
        del kwargs
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if labels is not None and labels.max() >= self.config.vocab_size:
            raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

        extract_features = self.wav2vec2.feature_extractor(input_values)
        # [B, T_audio] -> convolution features [B, C, T_frame].
        extract_features = extract_features.transpose(1, 2)
        # [B, C, T_frame] -> [B, T_frame, C].

        feature_attention_mask = None
        if attention_mask is not None:
            feature_attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1],
                attention_mask,
                add_adapter=False,
            )
            # [B, T_audio] -> downsampled mask [B, T_frame].

        hidden_states, _ = self.wav2vec2.feature_projection(extract_features)
        # [B, T_frame, C] -> projected states [B, T_frame, H].
        hidden_states = self.wav2vec2._mask_hidden_states(
            hidden_states,
            attention_mask=feature_attention_mask,
        )
        # SpecAugment preserves shape [B, T_frame, H].

        (
            encoder_output,
            dialect_logits,
            dialect_pooled,
            dialect_embedding,
            dialect_posterior,
            gate_statistics,
        ) = self._forward_dggfm_encoder(
            hidden_states,  # [B, T_frame, H].
            feature_attention_mask,  # [B, T_frame] or None.
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = self.dropout(encoder_output.last_hidden_state)
        # [B, T_frame, H] -> dropout output [B, T_frame, H].
        logits = self.lm_head(hidden_states)
        # [B, T_frame, H] -> CTC logits [B, T_frame, V].

        ctc_loss = self._compute_ctc_loss(
            logits,  # [B, T_frame, V].
            input_values,  # [B, T_audio].
            attention_mask,  # [B, T_audio] or None.
            labels,  # [B, T_text] or None.
        )
        dialect_loss = None
        if region_labels is not None:
            if region_labels.ndim != 1 or region_labels.shape[0] != logits.shape[0]:
                raise ValueError("region_labels phải có shape [B]")
            dialect_loss = F.cross_entropy(
                dialect_logits,  # [B, R].
                region_labels,  # [B].
            )  # [B, R] and [B] -> scalar loss [].

        loss = ctc_loss
        if dialect_loss is not None:
            weighted_dialect_loss = self.dialect_loss_weight * dialect_loss
            # Scalar [] multiplied by scalar weight -> scalar [].
            loss = weighted_dialect_loss if loss is None else loss + weighted_dialect_loss
            # Optional CTC scalar [] + dialect scalar [] -> total scalar [].

        if not return_dict:
            output = (
                logits,
                encoder_output.hidden_states,
                encoder_output.attentions,
                dialect_logits,
                dialect_posterior,
            )
            return ((loss,) + output) if loss is not None else output

        return DGGFMCTCOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_output.hidden_states,
            attentions=encoder_output.attentions,
            ctc_loss=ctc_loss,
            dialect_loss=dialect_loss,
            dialect_logits=dialect_logits,
            dialect_pooled=dialect_pooled,
            dialect_embedding=dialect_embedding,
            dialect_posterior=dialect_posterior,
            gate_statistics=gate_statistics,
        )

    def _compute_ctc_loss(
        self,
        logits: Tensor,
        input_values: Tensor,
        attention_mask: Tensor | None,
        labels: Tensor | None,
    ) -> Tensor | None:
        if labels is None:
            return None

        audio_mask = (
            attention_mask
            if attention_mask is not None
            else torch.ones_like(input_values, dtype=torch.long)
        )
        # [B, T_audio] -> valid audio lengths [B].
        input_lengths = self._get_feat_extract_output_lengths(
            audio_mask.sum(-1)
        ).to(torch.long)
        labels_mask = labels >= 0  # [B, T_text] -> boolean mask [B, T_text].
        target_lengths = labels_mask.sum(-1)  # [B, T_text] -> [B].
        flattened_targets = labels.masked_select(labels_mask)
        # Valid entries from [B, T_text] -> flattened targets [sum(target_lengths)].
        log_probs = F.log_softmax(
            logits,
            dim=-1,
            dtype=torch.float32,
        ).transpose(0, 1)
        # [B, T_frame, V] -> log softmax [B, T_frame, V] -> [T_frame, B, V].

        with torch.backends.cudnn.flags(enabled=False):
            return F.ctc_loss(
                log_probs,  # [T_frame, B, V].
                flattened_targets,  # [sum(target_lengths)].
                input_lengths,  # [B].
                target_lengths,  # [B].
                blank=self.config.pad_token_id,
                reduction=self.config.ctc_loss_reduction,
                zero_infinity=self.config.ctc_zero_infinity,
            )  # All inputs -> scalar CTC loss [].
