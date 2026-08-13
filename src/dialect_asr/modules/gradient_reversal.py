"""Gradient-reversal layer used by domain-adversarial training."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class _GradientReversalFunction(torch.autograd.Function):
    """Identity in forward, sign-reversed and scaled gradient in backward."""

    @staticmethod
    def forward(ctx: object, inputs: Tensor, scale: float) -> Tensor:
        ctx.scale = scale
        return inputs.view_as(inputs)  # [*] -> [*], values are unchanged.

    @staticmethod
    def backward(ctx: object, grad_output: Tensor) -> tuple[Tensor, None]:
        # Incoming gradient [*] -> reversed encoder gradient [*].
        return -ctx.scale * grad_output, None


def reverse_gradient(inputs: Tensor, scale: float = 1.0) -> Tensor:
    """Return an identity view whose backward gradient is ``-scale * grad``."""
    if scale < 0.0:
        raise ValueError("GRL scale phải >= 0")
    return _GradientReversalFunction.apply(inputs, float(scale))
    # Tensor [*] and scalar scale [] -> tensor [*].


class GradientReversal(nn.Module):
    """Module wrapper around :func:`reverse_gradient`."""

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        if scale < 0.0:
            raise ValueError("GRL scale phải >= 0")
        self.scale = float(scale)

    def forward(self, inputs: Tensor) -> Tensor:
        return reverse_gradient(inputs, self.scale)  # [*] -> [*].
