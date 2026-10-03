"""SeqForge · 序贯贝叶斯推断与状态空间估计实验台。

作者：晨星(CJX0712)
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "晨星"
__license__ = "MIT"

from core.config import SeqForgeConfig, load_config
from core.errors import (
    BackendUnavailableError,
    ConfigError,
    ConvergenceError,
    DGPError,
    NumericalError,
    SeqForgeError,
    ShapeError,
)
from core.seed import set_all
from core.types import FilterEstimate, FilterReport, SequentialDataset

__all__ = [
    "BackendUnavailableError",
    "ConfigError",
    "ConvergenceError",
    "DGPError",
    "FilterEstimate",
    "FilterReport",
    "NumericalError",
    "SeqForgeConfig",
    "SeqForgeError",
    "SequentialDataset",
    "ShapeError",
    "__author__",
    "__version__",
    "load_config",
    "set_all",
]
