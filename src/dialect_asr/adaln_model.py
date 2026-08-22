"""PhoWhisper encoder conditioned on a frozen DID embedding via AdaLN.

Every encoder layer's two ``LayerNorm``s are replaced by :class:`AdaLN`
(:mod:`dialect_asr.modules.adaln`), modulated by a conditioning vector derived
from :class:`ECAPA_TDNN_DID`'s embedding of the same input log-mel features.
Design choices (fixed for this architecture):

- The DID branch is **frozen** — it is only ever used as a feature extractor,
  loaded from a checkpoint produced by ``scripts/train_did.py``.
- **All** encoder layers are conditioned (not just the last few).
- The conditioning vector comes from the **real DID embedding of the input
  audio** (not a region-label lookup), so the model reacts to the actual
  dialect signal in each utterance rather than a coarse 3-way label.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import torch
from torch import Tensor, nn
from transformers import WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperEncoder, WhisperEncoderLayer

from .base_model import AbstractPhoWhisperASR, DEFAULT_PRETRAINED_MODEL
from .modules import AdaLN, DialectConditioner, ECAPA_TDNN_DID
from .reproducibility import DEFAULT_SEED, seed_everything


WANDB_ARTIFACT_PREFIX = "wandb-artifact:"


def resolve_did_checkpoint_path(path_or_artifact: str | Path) -> Path:
    """Resolve a local path or a ``wandb-artifact:<entity>/<project>/<name>:<version>``
    reference (as logged by ``scripts/train_did.py``) to a local checkpoint file.
    """
    text = str(path_or_artifact)
    if not text.startswith(WANDB_ARTIFACT_PREFIX):
        return Path(text)

    import wandb

    artifact_reference = text[len(WANDB_ARTIFACT_PREFIX) :]
    download_dir = Path(wandb.Api().artifact(artifact_reference).download())
    checkpoints = sorted(download_dir.glob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(
            f"W&B artifact {artifact_reference!r} không chứa file .pt nào"
        )
    if len(checkpoints) > 1:
        raise ValueError(
            f"W&B artifact {artifact_reference!r} chứa nhiều file .pt: {checkpoints}; "
            "hãy trỏ trực tiếp tới file cụ thể thay vì cả artifact"
        )
    return checkpoints[0]


class AdaLNWhisperEncoderLayer(WhisperEncoderLayer):
    """``WhisperEncoderLayer`` whose two norms take a ``cond`` argument.

    Never constructed directly — an existing ``WhisperEncoderLayer`` instance
    has its norms swapped for :class:`AdaLN` and its ``__class__`` reassigned
    to this subclass in :func:`inject_adaln`, so its ``self_attn``/``fc1``/
    ``fc2`` weights (pretrained or randomly initialized by the base class)
    are left untouched.
    """

    def forward(  # type: ignore[override]
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None,
        cond: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states, cond)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states, cond)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        return hidden_states


class AdaLNWhisperEncoder(WhisperEncoder):
    """``WhisperEncoder`` that threads a ``cond`` vector to every layer.

    Never constructed directly — see :func:`inject_adaln`.
    """

    def forward(  # type: ignore[override]
        self,
        input_features: Tensor,
        attention_mask: Tensor | None = None,
        cond: Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        from transformers.modeling_outputs import BaseModelOutput

        if cond is None:
            raise ValueError("AdaLNWhisperEncoder yêu cầu cond [B, cond_dim], nhận None")

        expected_seq_length = self.config.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]
        if input_features.shape[-1] != expected_seq_length:
            raise ValueError(
                "Whisper expects the mel input features to be of length "
                f"{expected_seq_length}, but found {input_features.shape[-1]}."
            )

        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        all_positions = torch.arange(self.embed_positions.num_embeddings, device=inputs_embeds.device)

        hidden_states = inputs_embeds + self.embed_positions(all_positions)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        for encoder_layer in self.layers:
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    to_drop = True

            if not to_drop:
                hidden_states = encoder_layer(hidden_states, None, cond, **kwargs)

        hidden_states = self.layer_norm(hidden_states)

        return BaseModelOutput(last_hidden_state=hidden_states)


def inject_adaln(encoder: WhisperEncoder, cond_dim: int) -> None:
    """Swap every layer's LayerNorms for :class:`AdaLN`, in place.

    Must run *after* pretrained weights are loaded into ``encoder`` — the
    replacement norms keep the original ``weight``/``bias`` tensors (copied,
    not reinitialized) and only add a zero-initialized conditioning head, so
    running this on an already-loaded checkpoint does not change its outputs
    until the conditioning head is trained away from zero.
    """
    d_model = encoder.config.d_model
    for layer in encoder.layers:
        for norm_name in ("self_attn_layer_norm", "final_layer_norm"):
            old_norm = getattr(layer, norm_name)
            new_norm = AdaLN(d_model, cond_dim)
            with torch.no_grad():
                new_norm.weight.copy_(old_norm.weight)
                new_norm.bias.copy_(old_norm.bias)
            setattr(layer, norm_name, new_norm)
        layer.__class__ = AdaLNWhisperEncoderLayer
    encoder.__class__ = AdaLNWhisperEncoder


class PhoWhisperAdaLNASR(AbstractPhoWhisperASR):
    """PhoWhisper whose encoder is conditioned on a frozen DID embedding.

    Shape contract (in addition to the one inherited from
    ``AbstractPhoWhisperASR``):

    - ``did_model`` consumes the same ``input_features``
      ``[B, num_mel_bins, T_frame]`` as the encoder and returns an embedding
      ``[B, did_embedding_size]``.
    - ``conditioner`` projects that to ``cond`` ``[B, d_model]``, broadcast to
      every encoder layer's AdaLN.
    """

    @classmethod
    def architecture_name(cls) -> str:
        return "adaln"

    def __init__(
        self,
        config: WhisperConfig,
        did_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config)
        inject_adaln(self.model.encoder, cond_dim=config.d_model)

        did_kwargs = dict(did_kwargs or {})
        did_kwargs.setdefault("hidden_size", config.num_mel_bins)
        self.did_model = ECAPA_TDNN_DID(**did_kwargs)
        self.did_model.requires_grad_(False)

        self.conditioner = DialectConditioner(self.did_model.embedding.out_features, config.d_model)

    def freeze_did(self) -> None:
        """Freeze the DID branch; it is only ever used as a feature extractor."""
        self.did_model.requires_grad_(False)
        self.did_model.eval()

    def train(self, mode: bool = True) -> Self:
        # Keep the frozen DID branch in eval() (stable BatchNorm running
        # stats) regardless of the rest of the model's train/eval mode.
        super().train(mode)
        self.did_model.eval()
        return self

    def _did_condition(self, input_features: Tensor) -> Tensor:
        # [B, num_mel_bins, T_frame] -> [B, T_frame, num_mel_bins] to match
        # ECAPA_TDNN_DID's [B, T_frame, H] contract.
        hidden_states = input_features.transpose(1, 2)
        with torch.no_grad():
            _, embedding = self.did_model(hidden_states)  # [B, did_embedding_size].
        return self.conditioner(embedding)  # [B, d_model].

    def forward(self, input_features: Tensor | None = None, **kwargs: Any) -> Any:
        cond = kwargs.pop("cond", None)
        if cond is None and input_features is not None:
            cond = self._did_condition(input_features)
        return super().forward(input_features=input_features, cond=cond, **kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        input_features = kwargs.get("input_features")
        if input_features is None and args:
            input_features = args[0]
        if input_features is not None and "cond" not in kwargs:
            kwargs["cond"] = self._did_condition(input_features)
        return super().generate(*args, **kwargs)

    @classmethod
    def from_vietnamese_pretrained(
        cls,
        pretrained_model_name: str = DEFAULT_PRETRAINED_MODEL,
        *,
        did_checkpoint_path: str | Path,
        did_kwargs: dict[str, Any] | None = None,
        freeze_encoder: bool = True,
        gradient_checkpointing: bool = False,
        seed: int = DEFAULT_SEED,
        full_determinism: bool = False,
        **from_pretrained_kwargs: Any,
    ) -> Self:
        """Load a PhoWhisper checkpoint, inject AdaLN, and attach a frozen DID.

        ``did_checkpoint_path`` must point at a ``final_model.pt`` produced by
        ``scripts/train_did.py`` with a matching ``did_kwargs`` configuration
        (``channels``, ``embedding_size``, ...), either as a local path or as
        ``"wandb-artifact:<entity>/<project>/<name>:<version>"``.
        """
        seed_everything(seed, deterministic=full_determinism)
        model = cls.from_pretrained(
            pretrained_model_name,
            did_kwargs=did_kwargs,
            **from_pretrained_kwargs,
        )

        resolved_did_checkpoint_path = resolve_did_checkpoint_path(did_checkpoint_path)
        did_state_dict = torch.load(resolved_did_checkpoint_path, map_location="cpu")
        model.did_model.load_state_dict(did_state_dict)
        model.freeze_did()

        if freeze_encoder:
            # Freeze the pretrained encoder weights (self_attn, fc1/fc2, and
            # each AdaLN's base weight/bias) but keep every AdaLN's
            # `to_gamma_beta` head and `conditioner` trainable — only the
            # dialect-conditioning path adapts, not the base acoustic model.
            for name, parameter in model.model.encoder.named_parameters():
                parameter.requires_grad_("to_gamma_beta" in name)
            model.conditioner.requires_grad_(True)

        if gradient_checkpointing:
            model.gradient_checkpointing_enable()

        return model
