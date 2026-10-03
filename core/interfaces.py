"""SeqForge 跨模块接口协议（Protocol）。

设计原则（架构组要求）：
**所有过滤器共享一套签名**，靠 capability 标记暴露能力差异，
而不是"每类方法一套签名"。这样 pipeline/eval 可以对任意方法
统一计时、统一取指标，跨方法比较才公平。

统一签名
--------
``filter(y)``
    在线滤波，返回 :class:`~core.types.FilterEstimate`。
    粒子法在每步内部重采样；解析法无重采样。

``smooth(y)``
    平滑。**不可用时抛NotImplementedError**，由
    :func:`supports_smoothing` 事前判定；调用方据此标 ``skipped``，
    不伪造数字。

``log_likelihood(y)``
    对数边缘似然（越大越好）。这是本域的主指标。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from core.types import FilterEstimate, SmoothingResult

__all__ = [
    "LinearGaussianModel",
    "NonlinearModel",
    "SequentialEstimator",
    "supports_ess",
    "supports_smoothing",
]


@runtime_checkable
class SequentialEstimator(Protocol):
    """序贯推断器的统一协议。"""

    name: str
    family: str  # "kalman" | "particle" | "baseline"

    def filter(self, y: np.ndarray) -> FilterEstimate:
        """在线滤波：输出后验均值/协方差与增量对数似然。

        Parameters
        ----------
        y
            (T, n_obs) 观测序列。

        Returns
        -------
        FilterEstimate
            滤波估计。``ess`` 为 ``None`` 表示该方法无粒子概念。
        """
        ...

    def smooth(self, y: np.ndarray) -> SmoothingResult:
        """离线平滑（RTS / FFBS）。"""
        ...

    def log_likelihood(self, y: np.ndarray) -> float:
        """对数边缘似然（越大越好）。"""
        ...


@runtime_checkable
class LinearGaussianModel(Protocol):
    """线性高斯转移/观测模型契约。"""

    A: np.ndarray  # (n, n)
    H: np.ndarray  # (m, n)
    Q: np.ndarray  # (n, n)
    R: np.ndarray  # (m, m)

    def predict(self, x: np.ndarray, u: np.ndarray | None = None) -> np.ndarray:
        """一步先验转移。"""
        ...

    def observe(self, x: np.ndarray) -> np.ndarray:
        """无噪声观测映射。"""
        ...


@runtime_checkable
class NonlinearModel(Protocol):
    """非线性状态空间模型契约（观测可为非线性）。"""

    n_state: int
    n_obs: int

    def transition(self, x: np.ndarray) -> np.ndarray:
        """确定性状态转移 f(x)。"""
        ...

    def emission(self, x: np.ndarray) -> np.ndarray:
        """确定性观测映射 h(x)。"""
        ...


def supports_smoothing(estimator: object) -> bool:
    """判断估计器是否支持 ``smooth``（避免 try/except 驱动控制流）。"""
    fn = getattr(estimator, "smooth", None)
    return callable(fn) and getattr(getattr(fn, "__func__", fn), "_unsupported", False) is not True


def supports_ess(estimator: object) -> bool:
    """判断估计器是否报告 ESS（解析法为 False）。"""
    return getattr(estimator, "reports_ess", False) is True
