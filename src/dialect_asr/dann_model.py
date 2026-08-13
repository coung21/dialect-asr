"""Domain-adversarial Wav2Vec2 CTC model with a final-block dialect head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor
from torch.nn import functional as F
from transformers.modeling_outputs import CausalLMOutput

from .model import BaselineWav2Vec2CTC
from .modules import DANNDialectHead


@dataclass
class DANNCTCOutput(CausalLMOutput):
    """CTC output augmented with adversarial dialect outputs and losses."""

    ctc_loss: Tensor | None = None
    dialect_loss: Tensor | None = None
    dialect_logits: Tensor | None = None
    dialect_pooled: Tensor | None = None


class DANNWav2Vec2CTC(BaselineWav2Vec2CTC):
    """Use GRL to remove dialect information from the final encoder block."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.num_regions = int(getattr(config, "num_regions", 3))
        self.dialect_bottleneck_size = int(
            getattr(config, "dialect_bottleneck_size", 256)
        )
        self.dialect_dropout = float(getattr(config, "dialect_dropout", 0.1))
        self.dialect_loss_weight = float(getattr(config, "dialect_loss_weight", 1.0))
        self.grl_scale = float(getattr(config, "grl_scale", 1.0))
        if self.dialect_loss_weight < 0.0:
            raise ValueError("dialect_loss_weight phải >= 0")
        if self.grl_scale < 0.0:
            raise ValueError("grl_scale phải >= 0")

        self.dialect_head = DANNDialectHead(
            hidden_size=config.hidden_size,
            num_regions=self.num_regions,
            bottleneck_size=self.dialect_bottleneck_size,
            dropout=self.dialect_dropout,
            grl_scale=self.grl_scale,
        )
        self.dialect_head.apply(self._init_weights)

        config.architecture = "dann"
        config.dann_branch_block = int(config.num_hidden_layers)
        config.num_regions = self.num_regions
        config.dialect_bottleneck_size = self.dialect_bottleneck_size
        config.dialect_dropout = self.dialect_dropout
        config.dialect_loss_weight = self.dialect_loss_weight
        config.grl_scale = self.grl_scale
        ignored_keys = set(getattr(config, "keys_to_ignore_at_inference", []) or [])
        ignored_keys.update(
            {"ctc_loss", "dialect_loss", "dialect_logits", "dialect_pooled"}
        )
        config.keys_to_ignore_at_inference = sorted(ignored_keys)

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
    ) -> tuple[Any, ...] | DANNCTCOutput:
        """Compute CTC and adversarial dialect-classification objectives."""
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
            raise RuntimeError("Wav2Vec2 không trả về hidden states")

        final_hidden_states = ctc_output.hidden_states[-1]
        # Last Transformer block output: [B, T_frame, H].
        feature_attention_mask = None
        if attention_mask is not None:
            feature_attention_mask = self._get_feature_vector_attention_mask(
                final_hidden_states.shape[1],
                attention_mask,
                add_adapter=False,
            )
            # Audio mask [B, T_audio] -> encoder-frame mask [B, T_frame].

        dialect_logits, dialect_pooled = self.dialect_head(
            final_hidden_states,  # [B, T_frame, H].
            feature_attention_mask,  # [B, T_frame] or None.
        )
        # Final states [B, T_frame, H] -> logits [B, R], pooled [B, H].

        dialect_loss = None
        if region_labels is not None:
            if region_labels.ndim != 1 or region_labels.shape[0] != dialect_logits.shape[0]:
                raise ValueError("region_labels phải có shape [B]")
            dialect_loss = F.cross_entropy(
                dialect_logits,  # [B, R].
                region_labels,  # [B].
            )  # [B, R] and [B] -> scalar CE loss [].

        total_loss = ctc_output.loss
        if dialect_loss is not None:
            weighted_dialect_loss = self.dialect_loss_weight * dialect_loss
            # Scalar weight [] * dialect loss [] -> scalar [].
            total_loss = (
                weighted_dialect_loss
                if total_loss is None
                else total_loss + weighted_dialect_loss
            )
            # Optional CTC loss [] + weighted dialect loss [] -> total loss [].

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
            return ((total_loss,) + output) if total_loss is not None else output

        return DANNCTCOutput(
            loss=total_loss,
            logits=ctc_output.logits,  # [B, T_frame, V].
            hidden_states=returned_hidden_states,
            attentions=ctc_output.attentions,
            ctc_loss=ctc_output.loss,
            dialect_loss=dialect_loss,
            dialect_logits=dialect_logits,
            dialect_pooled=dialect_pooled,
        )
