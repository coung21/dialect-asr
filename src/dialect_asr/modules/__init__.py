"""Reusable neural-network modules for dialect-aware ASR models."""

from .ecapa_tdnn import (
    AttentiveStatisticsPooling,
    ECAPA_TDNN_DID,
    Res2NetLayer,
    SERes2Block,
    SqueezeExcitation,
)

__all__ = [
    "AttentiveStatisticsPooling",
    "ECAPA_TDNN_DID",
    "Res2NetLayer",
    "SERes2Block",
    "SqueezeExcitation",
]
