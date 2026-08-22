"""Reusable neural-network modules for dialect-aware ASR models."""

from .adaln import AdaLN, DialectConditioner
from .ecapa_tdnn import (
    AttentiveStatisticsPooling,
    ECAPA_TDNN_DID,
    Res2NetLayer,
    SERes2Block,
    SqueezeExcitation,
)

__all__ = [
    "AdaLN",
    "AttentiveStatisticsPooling",
    "DialectConditioner",
    "ECAPA_TDNN_DID",
    "Res2NetLayer",
    "SERes2Block",
    "SqueezeExcitation",
]
