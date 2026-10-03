"""合成序贯数据生成器（4 个 DGP，各针对一类方法的特定缺陷）。

难度甜点原则（踩坑库 §B「benchmark 全满分或全崩，无区分度」）
--------------------------------------------------------------
每个 DGP 都有明确的"暴露目标"，且必须让基线**明显低于天花板**：

=================  ========================================================
DGP                暴露什么
=================  ========================================================
linear_gaussian    KF 精确最优 ⇒ **任何方法都无法超越**，只作交叉验证锚点。
                   这里 KF 拿满分是**诚实的天花板**，不是"难度不足"。
nonlinear_obs      EKF 一阶泰勒线性化失效 vs UKF 二次精度
sparse_obs         高维低秩+ 稀疏观测 ⇒ bootstrap PF 权重退化
degenerate         窄似然（r=0.05）/ 双峰 ⇒ 提议分布与重采样策略定生死
=================  ========================================================

防泄漏硬约束：``x_true`` 只用于**评估**，绝不进入任何方法的推断路径。
"""

from __future__ import annotations

import numpy as np

from core.errors import DGPError
from core.types import (
    DegenerateSpec,
    LinearGaussianSpec,
    NonlinearSpec,
    SequentialDataset,
    SparseObsSpec,
    Spec,
)
from data.models import ProbabilisticSSM, build_linear_gaussian, build_nonlinear

__all__ = ["DEFAULT_SPECS", "build_model", "make_dataset"]


def _simulate(
    model: ProbabilisticSSM,
    *,
    T: int,
    rng: np.random.RandomState,
    x0: np.ndarray,
    obs_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """通用模拟循环，返回 ``(y, x_true, u)``。

    ``obs_mask`` 为 ``(T, n_obs)`` 布尔掩码：False 位置写 ``nan``（缺失观测），
    方法侧必须按缺失处理（粒子法直接跳过该步似然；解析法做预测不做更新）。
    """
    n, m = model.n_state, model.n_obs
    x = np.asarray(x0, dtype=np.float64).copy()
    y = np.empty((T, m), dtype=np.float64)
    x_true = np.empty((T, n), dtype=np.float64)
    u = np.zeros((max(T - 1, 1), 1), dtype=np.float64)

    for t in range(T):
        x_true[t] = x
        obs = model.sample_emission(x, rng)
        if obs_mask is not None:
            obs = np.where(obs_mask[t], obs, np.nan)
        y[t] = obs
        if t < T - 1:
            x = model.sample_transition(x, rng)
    return y, x_true, u


def build_model(name: str, spec: Spec, seed: int) -> ProbabilisticSSM:
    """按 DGP 名构造对应模型（方法侧与 DGP 侧共用同一构造器，避免口径漂移）。"""
    if isinstance(spec, LinearGaussianSpec):
        return build_linear_gaussian(spec, seed)
    if isinstance(spec, NonlinearSpec):
        return build_nonlinear(spec, seed)
    if isinstance(spec, SparseObsSpec):
        return _build_sparse(spec, seed)
    if isinstance(spec, DegenerateSpec):
        return _build_degenerate(spec, seed)
    raise DGPError("未知 spec 类型", detail=type(spec).__name__)


def _build_sparse(spec: SparseObsSpec, seed: int) -> ProbabilisticSSM:
    """高维 + 稀疏观测（**全秩 A**）。

    设计修订（架构组评审意见，已实测验证）：初版用 rank-3 低秩因子升维，
    实际测的是"低秩结构"而非"维度灾难"，且粒子在低秩流形上退化
    （实测 ESS 仅 7~45%，退化来源被污染）。现改为**全秩 A**：
    ESS 崩溃才真正来自观测维度。

    Parameters
    ----------
    spec.mode
        ``"dense_transition"``（默认，全秩 A，纯维度效应）
        ``"lowrank_transition"``（rank-k，单独测低秩结构，两者不混淆）
    """
    rng = np.random.RandomState(seed)
    n, m = spec.n_state, spec.n_obs
    k = spec.latent_lowrank

    if spec.mode == "lowrank_transition":
        A_low = _stable_a(k, spec.spectral_radius, rng)
        P_up = rng.randn(n, k) / np.sqrt(k)
        A = P_up @ A_low @ np.linalg.pinv(P_up)
    else:
        A = _stable_a(n, spec.spectral_radius, rng)

    # 稀疏观测：每个时刻只观测 m 个状态维度（选择矩阵）
    H = np.zeros((m, n))
    for i in range(m):
        H[i, i * n // m] = rng.choice([-1.0, 1.0])

    return ProbabilisticSSM(
        A, H, np.eye(n) * spec.q_scale**2, np.eye(m) * spec.r_scale**2, name=spec.name
    )


def _build_degenerate(spec: DegenerateSpec, seed: int) -> ProbabilisticSSM:
    """粒子退化场景：极窄似然（r=0.05）或近单位模振荡（双峰）。"""
    rng = np.random.RandomState(seed)
    n, m = spec.n_state, spec.n_obs
    A = _stable_a(n, spec.spectral_radius, rng)
    if spec.mode == "bimodal":
        # 让前两维的|特征值| 逼近 1，形成振荡双峰后验
        A[0, 0] = 0.985
        A[1, 1] = -0.985
    H = np.eye(m, n)
    return ProbabilisticSSM(
        A, H, np.eye(n) * spec.q_scale**2, np.eye(m) * spec.r_scale**2, name=spec.name
    )


def _stable_a(n: int, rho: float, rng: np.random.RandomState) -> np.ndarray:
    """谱半径精确为 ``rho`` 的实对称稳定矩阵（与 models 中同名实现一致）。"""
    V = np.linalg.qr(rng.randn(n, n))[0]
    s = rng.uniform(0.35, 1.0, size=n) * rho * rng.choice([-1.0, 1.0], size=n)
    A = (V * s) @ V.T
    return 0.5 * (A + A.T) * (rho / np.max(np.abs(np.linalg.eigvals((V * s) @ V.T))))


# --------------------------------------------------------------------------
# 四个 DGP
# --------------------------------------------------------------------------
def _dgp_linear_gaussian(spec: LinearGaussianSpec, T: int, seed: int) -> SequentialDataset:
    model = build_linear_gaussian(spec, seed)
    rng = np.random.RandomState(seed + 11)
    x0 = rng.randn(spec.n_state) * 1.2
    y, x_true, u = _simulate(model, T=T, rng=rng, x0=x0)
    return SequentialDataset(spec.name, spec, y, x_true, u, seed=seed)


def _dgp_nonlinear(spec: NonlinearSpec, T: int, seed: int) -> SequentialDataset:
    model = build_nonlinear(spec, seed)
    rng = np.random.RandomState(seed + 22)
    x0 = rng.randn(spec.n_state) * 1.0
    y, x_true, u = _simulate(model, T=T, rng=rng, x0=x0)
    return SequentialDataset(spec.name, spec, y, x_true, u, seed=seed)


def _dgp_sparse(spec: SparseObsSpec, T: int, seed: int) -> SequentialDataset:
    model = _build_sparse(spec, seed)
    rng = np.random.RandomState(seed + 33)
    obs_mask = rng.rand(T, spec.n_obs) < spec.obs_rate
    # 每行至少留一个被观测维度，否则该步无任何信息
    empty_rows = ~obs_mask.any(axis=1)
    obs_mask[empty_rows, rng.randint(0, spec.n_obs, size=int(empty_rows.sum()))] = True
    x0 = rng.randn(spec.n_state) * 0.8
    y, x_true, u = _simulate(model, T=T, rng=rng, x0=x0, obs_mask=obs_mask)
    # 记录 H 覆盖的状态维度：稀疏观测下只有这些维度的 RMSE 有意义
    obs_idx = np.flatnonzero(np.abs(model.H).sum(axis=0) > 1e-12)
    ds = SequentialDataset(spec.name, spec, y, x_true, u, seed=seed)
    ds._observed_idx = obs_idx
    return ds


def _dgp_degenerate(spec: DegenerateSpec, T: int, seed: int) -> SequentialDataset:
    model = _build_degenerate(spec, seed)
    rng = np.random.RandomState(seed + 44)
    x0 = rng.randn(spec.n_state)
    y, x_true, u = _simulate(model, T=T, rng=rng, x0=x0)
    return SequentialDataset(spec.name, spec, y, x_true, u, seed=seed)


DEFAULT_SPECS: dict[str, Spec] = {
    "linear_gaussian": LinearGaussianSpec(),
    "nonlinear_obs": NonlinearSpec(),
    "sparse_obs": SparseObsSpec(),
    "degenerate": DegenerateSpec(),
}

_BUILDERS = {
    "LinearGaussianSpec": _dgp_linear_gaussian,
    "NonlinearSpec": _dgp_nonlinear,
    "SparseObsSpec": _dgp_sparse,
    "DegenerateSpec": _dgp_degenerate,
}


def make_dataset(
    name: str, *, n_steps: int = 120, seed: int = 7, spec: Spec | None = None
) -> SequentialDataset:
    """按名字构造数据集；同 seed ⇒ 逐位一致。"""
    chosen = spec if spec is not None else DEFAULT_SPECS[name]
    builder = _BUILDERS.get(type(chosen).__name__)
    if builder is None:
        raise DGPError("未知 DGP 类型", detail=type(chosen).__name__)
    return builder(chosen, n_steps, seed)  # type: ignore[arg-type]
