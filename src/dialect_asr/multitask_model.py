"""Multi-task Wav2Vec2 model for joint ASR and dialect classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor
from torch.nn import functional as F
from transformers.modeling_outputs import CausalLMOutput

from .base_model import AbstractWav2Vec2CTC
from .modules import DialectBranch


@dataclass
class MultitaskCTCOutput(CausalLMOutput):
    """CTC output augmented with dialect-classification results."""

    ctc_loss: Tensor | None = None
    dialect_loss: Tensor | None = None
    dialect_logits: Tensor | None = None
    dialect_pooled: Tensor | None = None


class MultitaskWav2Vec2CTC(AbstractWav2Vec2CTC):
    """Baseline CTC model with an auxiliary dialect branch after block 6."""

    DEFAULT_BRANCH_BLOCK = 6

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.branch_block = int(
            getattr(config, "multitask_branch_block", self.DEFAULT_BRANCH_BLOCK)
        )
        self.num_regions = int(getattr(config, "num_regions", 3))
        self.dialect_bottleneck_size = int(
            getattr(config, "dialect_bottleneck_size", 256)
        )
        self.dialect_dropout = float(getattr(config, "dialect_dropout", 0.1))
        self.dialect_loss_weight = float(
            getattr(config, "dialect_loss_weight", 1.0)
        )
        self._validate_architecture_config(config)
        self._save_architecture_config(config)

        self.dialect_branch = DialectBranch(
            hidden_size=config.hidden_size,
            num_regions=self.num_regions,
            bottleneck_size=self.dialect_bottleneck_size,
            dropout=self.dialect_dropout,
        )

        ignored_at_inference = set(
            getattr(config, "keys_to_ignore_at_inference", []) or []
        )
        ignored_at_inference.update(
            {"ctc_loss", "dialect_loss", "dialect_logits", "dialect_pooled"}
        )
        config.keys_to_ignore_at_inference = sorted(ignored_at_inference)

    @classmethod
    def architecture_name(cls) -> str:
        """Return the Hydra/registry identifier for this architecture."""
        return "multitask"

    def _validate_architecture_config(self, config: Any) -> None:
        if not 1 <= self.branch_block <= config.num_hidden_layers:
            raise ValueError(
                "multitask_branch_block phải nằm trong số Transformer blocks; "
                f"nhận {self.branch_block}/{config.num_hidden_layers}"
            )
        if self.dialect_loss_weight < 0.0:
            raise ValueError("dialect_loss_weight phải >= 0")

    def _save_architecture_config(self, config: Any) -> None:
        """Persist custom settings for save_pretrained/from_pretrained."""
        config.architecture = self.architecture_name()
        config.multitask_branch_block = self.branch_block
        config.num_regions = self.num_regions
        config.dialect_bottleneck_size = self.dialect_bottleneck_size
        config.dialect_dropout = self.dialect_dropout
        config.dialect_loss_weight = self.dialect_loss_weight

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
    ) -> tuple[Any, ...] | MultitaskCTCOutput:
        """Compute CTC loss plus optional auxiliary dialect loss."""
        requested_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        requested_return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )

        ctc_output = super().forward(
            input_values=input_values,  # [B, T_audio].
            attention_mask=attention_mask,  # [B, T_audio] or None.
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
            labels=labels,  # [B, T_text] or None.
            **kwargs,
        )
        if ctc_output.hidden_states is None:
            raise RuntimeError("Wav2Vec2 không trả về intermediate hidden states")

        branch_hidden_states = ctc_output.hidden_states[self.branch_block]
        # hidden_states[6] is block-6 output: [B, T_frame, H].
        feature_attention_mask = None
        if attention_mask is not None:
            feature_attention_mask = self._get_feature_vector_attention_mask(
                branch_hidden_states.shape[1],
                attention_mask,
                add_adapter=False,
            )
            # [B, T_audio] -> feature mask [B, T_frame].

        dialect_logits, dialect_pooled = self.dialect_branch(
            branch_hidden_states,  # [B, T_frame, H].
            feature_attention_mask,  # [B, T_frame] or None.
        )
        # [B, T_frame, H] -> logits [B, R], pooled [B, H].

        dialect_loss = None
        if region_labels is not None:
            if region_labels.ndim != 1 or region_labels.shape[0] != dialect_logits.shape[0]:
                raise ValueError("region_labels phải có shape [B]")
            dialect_loss = F.cross_entropy(
                dialect_logits,  # [B, R].
                region_labels,  # [B].
            )  # [B, R] and [B] -> scalar dialect loss [].

        loss = ctc_output.loss
        if dialect_loss is not None:
            weighted_dialect_loss = self.dialect_loss_weight * dialect_loss
            # Scalar dialect loss [] multiplied by scalar weight -> scalar [].
            loss = weighted_dialect_loss if loss is None else loss + weighted_dialect_loss
            # Optional CTC scalar [] + weighted dialect scalar [] -> total loss [].

        returned_hidden_states = (
            ctc_output.hidden_states if requested_hidden_states else None
        )
        if not requested_return_dict:
            output = (
                ctc_output.logits,  # [B, T_frame, V].
                returned_hidden_states,
                ctc_output.attentions,
                dialect_logits,  # [B, R].
            )
            return ((loss,) + output) if loss is not None else output

        return MultitaskCTCOutput(
            loss=loss,
            logits=ctc_output.logits,  # [B, T_frame, V].
            hidden_states=returned_hidden_states,
            attentions=ctc_output.attentions,
            ctc_loss=ctc_output.loss,
            dialect_loss=dialect_loss,
            dialect_logits=dialect_logits,
            dialect_pooled=dialect_pooled,
        )
