"""数值工具：协方差稳定化、log-sum-exp、Cholesky 安全封装。

本域最大的工程风险是**协方差半正定性在数值上退化**
（Joseph 形式理论上保 PSD，但浮点下仍可能因舍入丢失正定性，
导致 Cholesky 失败或卡尔曼增益爆炸）。因此所有协方差更新
统一走 :func:`stabilize_cov`。
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky as _cho

from core.errors import NumericalError

__all__ = [
    "log_ndtr_approx",
    "logsumexp",
    "mahalanobis_sq",
    "safe_cholesky",
    "stabilize_cov",
    "sym",
]

_JITTER = 1e-9


def sym(a: np.ndarray) -> np.ndarray:
    """显式对称化 ``(M + Mᵀ) / 2``（踩坑库 §F 明确要求，不可省）。"""
    return 0.5 * (a + a.T)


def stabilize_cov(p: np.ndarray, *, jitter: float = _JITTER, max_eig: float = 1e12) -> np.ndarray:
    """把协方差矩阵投影回近半正定锥。

    步骤：对称化 → **SVD** → 奇异值 clip 到 ``[jitter, max_eig]`` → 重构。
    返回值必��满足 ``min(eigvals) >= jitter`` 与对称性。

    为什么用 SVD 而不是 eigh：粒子滤波的经验协方差是**低秩**的
    （N 个粒子张成的子空间最多 rank N）。当 ``N < n_state`` 时
    ``eigh`` 直接抛 ``LinAlgError: Eigenvalues did not converge``（实测
    n_state=30 / N=1000 仍触发，因粒子在退化的低秩流形上共线）。
    SVD 对秩亏矩阵**恒收敛**，且奇异值 clip 等价于特征值 clip。
    """
    p = np.asarray(p, dtype=np.float64)
    p = 0.5 * (p + p.T)
    if p.ndim != 2 or p.shape[0] != p.shape[1]:
        raise NumericalError("协方差必须为方阵", detail=f"shape={p.shape}")
    try:
        u, s, vt = np.linalg.svd(p)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - 极罕见
        raise NumericalError("SVD 分解失败", detail=str(exc)) from exc
    s = np.clip(s, jitter, max_eig)
    out = (u * s) @ vt
    out = 0.5 * (out + out.T)
    # SVD 重构后仍可能残留 −jitter 量级的负特征值（实测 rank-1 阵恰为 −1.0e-9，
    # 因为 U/Vᵀ 的正交误差把 clip 后的最小奇异值"挤"到 0 附近）。
    # 兜底：对重构结果再做一次特征值 clip，保证 min(eig) >= jitter/2。
    w, v = np.linalg.eigh(out)
    if w.min() < jitter * 0.5:
        out = (v * np.clip(w, jitter * 0.5, max_eig)) @ v.T
        out = 0.5 * (out + out.T)
    return out


def safe_cholesky(p: np.ndarray) -> np.ndarray:
    """带抖动重试的 Cholesky，返回下三角 L。

    先尝试 ``stabilize_cov``；若仍失败，指数递增 jitter 重试。

    Raises
    ------
    NumericalError
        三次抖动重试后仍失败。
    """
    p = stabilize_cov(p)
    for attempt in range(3):
        try:
            return _cho(p, lower=True)
        except Exception:  # scipy 抛 LinAlgError 子类
            p = stabilize_cov(p, jitter=_JITTER * (10.0 ** (attempt + 1)))
    raise NumericalError("Cholesky 三次抖动重试后仍失败")


def logsumexp(a: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """数值稳定的 log-sum-exp（防权重归一化下溢）。"""
    a = np.asarray(a, dtype=np.float64)
    amax = np.max(a, axis=axis, keepdims=True)
    amax = np.where(np.isfinite(amax), amax, 0.0)
    out = np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True)) + amax
    return np.squeeze(out, axis=axis) if axis is not None else float(out)


def log_ndtr_approx(x: np.ndarray) -> np.ndarray:
    """标准正态对数累积分布的 log 空间计算。

    用 ``scipy.special.log_ndtr``；本域权重恒在 log 域累加，
    若在概率域累加则窄似然场景（r_scale=0.05）会直接下溢到 0。
    """
    from scipy.special import log_ndtr

    return np.asarray(log_ndtr(np.asarray(x, dtype=np.float64)), dtype=np.float64)


def mahalanobis_sq(v: np.ndarray, chol_prec: np.ndarray | None = None) -> np.ndarray:
    """马氏平方，返回 ``(...,)``。

    Parameters
    ----------
    v
        (..., m) 残差。
    chol_prec
        (m, m) 精度矩阵的 Cholesky 下三角 L（满足 ``LᵀL = 精度``）。
        为 None 时按单位精度处理。
    """
    if chol_prec is None:
        return np.sum(np.asarray(v) ** 2, axis=-1)
    return np.sum((np.linalg.solve(chol_prec, np.asarray(v).T)) ** 2, axis=0)
