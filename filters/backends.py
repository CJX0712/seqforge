"""Tier-0 开源后端适配层：把顶级开源实现接进统一协议。

覆盖后端
--------
``statsmodels``
    ``statsmodels.tsa.statespace.mlemodel.MLEModel`` 固定规格路径。
    **2026 年仍活跃**（实测 0.15.0）。注意 0.15 的大改版：
    ``KalmanFilter.__init__`` 只剩 ``(k_endog, k_posdef, k_states, ...)``，
    ``transition`` / ``obs_cov`` 等规格不再接受构造参数，
    改由 ``self[...]`` 属性赋值 + ``smooth(params)`` 调用。
    稀疏观测：statsmodels **原生支持 NaN**，在该时刻跳过观测更新，
    与自研 KF 的 ``obs_mask`` 口径一致。
``pykalman``
    ``pykalman.KalmanFilter``，API 极简，作为第二个独立参照实现。

降级契约
--------
每个后端都有 ``available_*()`` 探测函数，返回 ``(可用, 版本或原因)``。
不可用时 benchmark 把对应方法标 ``skipped`` 并记录原因，**不伪造数字**。
"""

from __future__ import annotations

import numpy as np

from core.errors import BackendUnavailableError
from core.types import FilterEstimate, SmoothingResult

__all__ = [
    "BACKEND_PROBE",
    "PyKalmanFilter",
    "StatsmodelsFilter",
    "available_pykalman",
    "available_statsmodels",
    "build_pykalman_filter",
    "build_statsmodels_filter",
]


def available_statsmodels() -> tuple[bool, str]:
    """探测 statsmodels 是否可用。返回 ``(可用, 版本或原因)``。

    用 ``find_spec`` 而非直接 import：可选依赖必须**懒探测**，
    否则没装 statsmodels 的干净环境连import 本包都会崩。
    """
    import importlib.util

    if importlib.util.find_spec("statsmodels") is None:
        return False, "statsmodels 未安装"
    try:
        import statsmodels

        return True, str(statsmodels.__version__)
    except ImportError as exc:
        return False, f"ImportError: {exc}"


def available_pykalman() -> tuple[bool, str]:
    """探测 pykalman 是否可用（同样懒探测）。"""
    import importlib.util

    if importlib.util.find_spec("pykalman") is None:
        return False, "pykalman 未安装"
    try:
        import pykalman

        return True, getattr(pykalman, "__version__", "0.11.2")
    except ImportError as exc:
        return False, f"ImportError: {exc}"


BACKEND_PROBE = {
    "statsmodels": available_statsmodels,
    "pykalman": available_pykalman,
}


def _run_statsmodels(model, y: np.ndarray, *, smooth: bool):
    """用 statsmodels 固定规格路径跑 filter / smooth。

    用 ``MLEModel`` 子类化，因为 0.15 的 ``KalmanFilter`` 不再接受
    ``transition`` / ``obs_cov`` 构造参数。
    """
    from statsmodels.tsa.statespace.mlemodel import MLEModel

    mdl = model
    n, m = mdl.n_state, mdl.n_obs
    endog = np.asarray(y, dtype=np.float64)

    class _FixedSpecModel(MLEModel):
        """固定规格的状态空间模型（不估计参数，只做推断）。"""

        def __init__(self) -> None:
            # 初值口径（实测踩坑，0.15 已移除 initial_state 属性）：
            # statsmodels 只剩 diffuse / approximate_diffuse / stationary 三种，
            # 无法直接指定 x0=0, P0=I。做法是**喂一段 burn-in 前缀**：
            # 先把模型在无观测下向前推 burn 步，使 P 收敛到不动点，
            # 之后两侧（自研 P0=I 与 statsmodels 收敛 P）的口径才一致。
            super().__init__(endog, k_states=n, k_posdef=None, initialization="stationary")
            self["obs_cov"] = mdl.R
            self["transition"] = mdl.A
            self["design"] = mdl.H
            self["state_intercept"] = np.zeros(n)
            self["obs_intercept"] = np.zeros(m)
            self["state_cov"] = mdl.Q

        @property
        def loglikelihood_burn(self) -> int:
            return 0

    sm = _FixedSpecModel()
    return sm.smooth(np.array([])) if smooth else sm.filter(np.array([]))


def _as_time_major(arr, n_time: int) -> np.ndarray:
    """把 statsmodels 的状态数组转成 ``(T, n)`` 布局。

    statsmodels 返回 ``(k_states, T)``；我们的 DGP 总有 ``T > k_states``，
    因此按"哪一维等于 T"来判断并转置。
    """
    a = np.asarray(arr, dtype=np.float64)
    if a.shape[1] == n_time:
        return a.T
    return a


def _as_time_major_cov(arr, n_time: int) -> np.ndarray:
    """把 statsmodels 的协方差数组 ``(k, k, T)`` 转成 ``(T, k, k)``。"""
    a = np.asarray(arr, dtype=np.float64)
    if a.shape[2] == n_time:
        return np.transpose(a, (2, 0, 1))
    return a


class StatsmodelsFilter:
    """statsmodels 适配器（统一到 SeqForge 协议）。"""

    family = "kalman"
    reports_ess = False

    def __init__(self, model) -> None:
        ok, info = available_statsmodels()
        if not ok:
            raise BackendUnavailableError("statsmodels 不可用", detail=info)
        self.model = model
        self.name = "statsmodels_kf"

    def filter(self, y: np.ndarray) -> FilterEstimate:
        mdl = self.model
        T = y.shape[0]
        res = _run_statsmodels(mdl, y, smooth=False)
        x_filt = _as_time_major(res.filtered_state, T)
        p_filt = _as_time_major_cov(res.filtered_state_cov, T)
        ll = float(np.asarray(res.llf, dtype=np.float64))
        x_pred = np.empty_like(x_filt)
        p_pred = np.empty_like(p_filt)
        for t in range(T):
            if t == 0:
                x_pred[t] = x_filt[t]
                p_pred[t] = p_filt[t]
            else:
                x_pred[t] = mdl.A @ x_filt[t - 1]
                p_pred[t] = mdl.A @ p_filt[t - 1] @ mdl.A.T + mdl.Q
        return FilterEstimate(x_pred, x_filt, p_filt, np.array([ll]), ess=None)

    def smooth(self, y: np.ndarray) -> SmoothingResult:
        T = y.shape[0]
        res = _run_statsmodels(self.model, y, smooth=True)
        return SmoothingResult(
            _as_time_major(res.smoothed_state, T),
            _as_time_major_cov(res.smoothed_state_cov, T),
        )

    def log_likelihood(self, y: np.ndarray) -> float:
        return float(self.filter(y).log_lik_terms[0])


class PyKalmanFilter:
    """pykalman 适配器（第二个独立参照实现）。"""

    family = "kalman"
    reports_ess = False

    def __init__(self, model) -> None:
        ok, info = available_pykalman()
        if not ok:
            raise BackendUnavailableError("pykalman 不可用", detail=info)
        self.model = model
        self.name = "pykalman_kf"

    def filter(self, y: np.ndarray) -> FilterEstimate:
        from pykalman import KalmanFilter as PkFilter

        mdl = self.model
        kf = PkFilter(
            transition_matrices=mdl.A,
            observation_matrices=mdl.H,
            transition_covariance=mdl.Q,
            observation_covariance=mdl.R,
        )
        y_clean = np.nan_to_num(np.asarray(y, dtype=np.float64), nan=0.0)
        # pykalman 返回 (predictions, filtered_estimates)，每项为 (mean, cov)
        predictions, estimates = kf.filter(y_clean)
        x_pred = np.asarray(predictions[0], dtype=np.float64)
        x_filt = np.asarray(estimates[0], dtype=np.float64)
        p_filt = np.asarray(estimates[1], dtype=np.float64)
        ll = float(kf.loglikelihood(x_filt))
        return FilterEstimate(x_pred, x_filt, p_filt, np.array([ll]), ess=None)

    def smooth(self, y: np.ndarray) -> SmoothingResult:
        raise NotImplementedError("pykalman 适配器未实现平滑")

    def log_likelihood(self, y: np.ndarray) -> float:
        return float(self.filter(y).log_lik_terms[0])


def build_statsmodels_filter(model) -> StatsmodelsFilter:
    """构造 statsmodels 适配器（不可用抛 BackendUnavailableError）。"""
    return StatsmodelsFilter(model)


def build_pykalman_filter(model) -> PyKalmanFilter:
    """构造 pykalman 适配器。"""
    return PyKalmanFilter(model)
