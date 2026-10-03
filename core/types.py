"""SeqForge 统一数据类型契约。

本模块定义系统内所有跨模块传递的数据结构。字段语义一经发布不再变更，
因为 benchmark 报告、失败案例归因都依赖这些字段的稳定含义。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "DegenerateSpec",
    "FilterEstimate",
    "FilterReport",
    "LinearGaussianSpec",
    "MethodResult",
    "NonlinearSpec",
    "SequentialDataset",
    "SmoothingResult",
    "SparseObsSpec",
    "Spec",
]


@dataclass(frozen=True)
class LinearGaussianSpec:
    """线性高斯状态空间模型规格。

    x_t = A x_{t-1} + B u_t + w_t,  w_t ~ N(0, Q)
    y_t = H x_t + v_t,          v_t ~ N(0, R)

    KF 在这类模型上是**精确贝叶斯最优**，任何自研算法都不可能超越它。
    因此本 DGP 的作用是：① 作为交叉验证真值锚点 ② 作为"诚实接受不劣"的域。
    """

    name: str = "linear_gaussian"
    n_state: int = 4
    n_obs: int = 4
    spectral_radius: float = 0.82
    q_scale: float = 0.20
    r_scale: float = 0.35
    obs_matrix: str = "full"  # full | partial


@dataclass(frozen=True)
class NonlinearSpec:
    """非线性观测 + 线性状态转移（暴露 EKF/UKF 差异）。"""

    name: str = "nonlinear_obs"
    n_state: int = 4
    n_obs: int = 4
    spectral_radius: float = 0.85
    q_scale: float = 0.15
    r_scale: float = 0.30
    nonlinearity: str = "quadratic"  # quadratic | radar | tanh
    obs_matrix: str = "full"


@dataclass(frozen=True)
class SparseObsSpec:
    """高维 + 稀疏观测（暴露 bootstrap PF 的权重退化）。

    高维下 bootstrap PF 的似然集中度随维度指数增长，ESS 迅速崩溃。
    这是 Rao-Blackwellised / 辅助变量提议分布的天然战场。
    """

    name: str = "sparse_obs"
    n_state: int = 30
    n_obs: int = 6
    spectral_radius: float = 0.90
    q_scale: float = 0.25
    r_scale: float = 0.30
    obs_rate: float = 0.30  # 每个时刻实际被观测的维度比例
    latent_lowrank: int = 3  # 仅 mode="lowrank_transition" 时生效
    mode: str = "dense_transition"  # dense_transition | lowrank_transition


@dataclass(frozen=True)
class DegenerateSpec:
    """粒子退化场景：窄似然 / 多峰似然。"""

    name: str = "degenerate"
    n_state: int = 6
    n_obs: int = 3
    spectral_radius: float = 0.88
    q_scale: float = 0.10
    r_scale: float = 0.05  # 极小观测噪声 ⇒ 窄似然
    mode: str = "narrow"  # narrow | bimodal


Spec = LinearGaussianSpec | NonlinearSpec | SparseObsSpec | DegenerateSpec


@dataclass
class SequentialDataset:
    """合成序贯数据集：观测 + 潜在真值轨迹。

    全部数组形状为 ``(T, dim)``；``x_true`` 供状态 RMSE 评估，
    **绝不**参与任何方法的推断过程（防泄漏硬约束）。
    """

    name: str
    spec: Spec
    y: np.ndarray  # (T, n_obs) 观测
    x_true: np.ndarray  # (T, n_state) 潜在真值状态
    u: np.ndarray  # (T-1, n_ctrl) 控制输入（零均值白噪声）
    seed: int
    _observed_idx: np.ndarray | None = None  # H 观测到的状态维度下标（稀疏 DGP）

    def observed_dims(self) -> np.ndarray:
        """返回 H 真正观测到的状态维度索引（稀疏 DGP 用）。

        部分可观测系统里**未观测维度的 RMSE 没有意义**（只能预测到先验均值），
        把它算进全状态 RMSE 会让所有方法都"看起来一样差"。
        因此报告必须区分：``rmse_state``（全状态，含不可观测维）与
        ``rmse_obs``（仅可观测子空间）。
        """
        return self._observed_idx

    def rmse(self, est: np.ndarray) -> dict[str, float]:
        """返回 ``{"rmse_state", "rmse_obs"}`` 两个口径的 RMSE。"""
        err = np.asarray(est, dtype=np.float64) - self.x_true
        out = {"rmse_state": float(np.sqrt(np.mean(err**2)))}
        if self._observed_idx is not None and len(self._observed_idx) > 0:
            out["rmse_obs"] = float(np.sqrt(np.mean(err[:, self._observed_idx] ** 2)))
        else:
            out["rmse_obs"] = out["rmse_state"]
        return out

    @property
    def n_steps(self) -> int:
        return int(self.y.shape[0])

    @property
    def n_obs(self) -> int:
        return int(self.y.shape[1])

    @property
    def n_state(self) -> int:
        return int(self.x_true.shape[1])

    def describe(self) -> dict[str, Any]:
        """可序列化的数据集描述，写入 benchmark.json。"""
        return {
            "name": self.name,
            "spec": type(self.spec).__name__,
            "n_steps": self.n_steps,
            "n_state": self.n_state,
            "n_obs": self.n_obs,
            "seed": self.seed,
        }


@dataclass
class FilterEstimate:
    """单次滤波运行的原始输出（未聚合指标）。

    Attributes
    ----------
    x_pred
        (T, n_state) 先验均值。
    x_filt
        (T, n_state) 后验均值。
    p_filt
        (T, n_state, n_state) 后验协方差。
    log_lik_terms
        (T,) 每步的增量对数似然（已含观测维度常数项）。
    ess
        (T,) 有效样本量，解析法填 ``nan``（KF 族无粒子概念）。
    """

    x_pred: np.ndarray
    x_filt: np.ndarray
    p_filt: np.ndarray
    log_lik_terms: np.ndarray
    ess: np.ndarray | None = None


@dataclass
class SmoothingResult:
    """平滑结果（RTS 或粒子平滑）。"""

    x_smooth: np.ndarray
    p_smooth: np.ndarray


@dataclass
class FilterReport:
    """方法级聚合结果 + 诊断信息。

    `log_lik` / `rmse_state` 语义统一为**越大/越小越好**已在文档锁定：
    - ``log_lik``：越大越好（对数边缘似然）
    - ``rmse_state``：越小越好
    - ``ess_mean``：越大越好（仅粒子法）
    """

    method: str
    family: str  # kalman | particle | baseline
    backend: str  # builtin | filterpy | pykalman ...
    dataset: str
    seed: int
    log_lik: float
    rmse_state: float
    n_steps: int
    runtime_sec: float
    ess_mean: float = float("nan")
    ais_ess_mean: float = float("nan")
    smoother_rmse: float = float("nan")
    skipped: bool = False
    skip_reason: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class MethodResult:
    """单方法跨数据集的跨 seed 聚合（mean ± std）。"""

    method: str
    family: str
    datasets: list[str]
    log_lik_mean: float
    log_lik_std: float
    rmse_mean: float
    rmse_std: float
    per_dataset: dict[str, dict[str, float]] = field(default_factory=dict)
    skipped: bool = False
    skip_reason: str = ""
