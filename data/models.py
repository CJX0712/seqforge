"""状态空间模型容器：统一提供采样 / 发射 / 噪声分解。

设计要点
--------
所有模型都实现同一组方法，粒子滤波与解析滤波共用：

``sample_transition(x, rng)``
    ``x_t ~ p(x_t | x_{t-1})``（含过程噪声）。
``emission(x)``
    观测均值 ``E[y_t | x_t] = h(x_t)``。
``sample_emission(x, rng)``
    ``y_t ~ p(y_t | x_t)``。
``cholesky()`` → ``(L_Q, logdet_Q, L_R, logdet_R)``
    供粒子法在log 域批量算似然而不显式求逆。

线性高斯时``A``/``H`` 可用于 KF 族；非线性时``h`` 被 EKF/UKF 使用。
"""

from __future__ import annotations

import numpy as np

from core.errors import DGPError, ShapeError
from core.types import LinearGaussianSpec, NonlinearSpec

__all__ = ["ProbabilisticSSM", "build_linear_gaussian", "build_nonlinear", "partial_obs_matrix"]


def partial_obs_matrix(n_state: int, n_obs: int, rng: np.random.RandomState) -> np.ndarray:
    """构造部分观测 H：每行恰有一个 ±1，其余为0。"""
    if n_obs > n_state:
        raise ShapeError("部分观测要求 n_obs <= n_state", detail=f"{n_obs} > {n_state}")
    idx = rng.choice(n_state, size=n_obs, replace=False)
    H = np.zeros((n_obs, n_state), dtype=np.float64)
    for r, j in enumerate(idx):
        H[r, j] = rng.choice([-1.0, 1.0])
    return H


def _stable_a(n: int, rho: float, rng: np.random.RandomState) -> np.ndarray:
    """构造谱半径精确为 ``rho`` 的实对称稳定矩阵。

    ``A = V diag(s) Vᵀ``，``s_i = ±u_i·rho``，``u_i ∈ [0.35, 1]``，
    最后整体缩放到 ``max|s_i| = rho``。
    """
    V = np.linalg.qr(rng.randn(n, n))[0]
    s = rng.uniform(0.35, 1.0, size=n) * rho * rng.choice([-1.0, 1.0], size=n)
    A = (V * s) @ V.T
    return 0.5 * (A + A.T) * (rho / np.max(np.abs(np.linalg.eigvals((V * s) @ V.T))))


class ProbabilisticSSM:
    """线性高斯 / 非线性观测状态空间模型。"""

    def __init__(
        self,
        A: np.ndarray,
        H: np.ndarray | None,
        Q: np.ndarray,
        R: np.ndarray,
        *,
        emission_fn=None,
        jacobian_fn=None,
        name: str = "ssm",
    ) -> None:
        self.A = np.asarray(A, dtype=np.float64)
        self.H = None if H is None else np.asarray(H, dtype=np.float64)
        self.Q = np.asarray(Q, dtype=np.float64)
        self.R = np.asarray(R, dtype=np.float64)
        self.name = name
        self._emission_fn = emission_fn
        self._jacobian_fn = jacobian_fn
        self._n_obs = int(self.H.shape[0]) if self.H is not None else None
        self._L_Q = np.linalg.cholesky(self.Q)
        self._L_R = np.linalg.cholesky(self.R)
        self._logdet_Q = 2.0 * float(np.sum(np.log(np.diag(self._L_Q))))
        self._logdet_R = 2.0 * float(np.sum(np.log(np.diag(self._L_R))))
        self._validate()

    def _validate(self) -> None:
        n = self.A.shape[0]
        if self.A.shape != (n, n):
            raise ShapeError("A 非方阵", detail=f"{self.A.shape}")
        if self.Q.shape != (n, n):
            raise ShapeError("Q 形状不符", detail=f"{self.Q.shape} vs {(n, n)}")
        if self.R.shape[0] != self.R.shape[1]:
            raise ShapeError("R 非方阵", detail=f"{self.R.shape}")
        if self._emission_fn is not None and self._n_obs is not None:
            probe = self._emission_fn(np.zeros(n))
            if probe.shape != (self._n_obs,):
                raise ShapeError(
                    "emission 输出维度不符", detail=f"{probe.shape} vs {(self._n_obs,)}"
                )

    @property
    def n_state(self) -> int:
        return int(self.A.shape[0])

    @property
    def n_obs(self) -> int:
        if self._n_obs is None:
            raise DGPError("模型未声明观测维度")
        return self._n_obs

    # --- 线性通道 ---
    def predict(self, x: np.ndarray, u: np.ndarray | None = None) -> np.ndarray:
        return self.A @ np.asarray(x, dtype=np.float64)

    def jacobian_h(self, x: np.ndarray) -> np.ndarray:
        """发射函数雅可比 (m, n)；线性时返回 H。"""
        if self._jacobian_fn is not None:
            return self._jacobian_fn(np.asarray(x, dtype=np.float64))
        if self.H is None:
            raise DGPError("模型既非线性也无 H 矩阵")
        return self.H

    # --- 通用通道 ---
    def emission(self, x: np.ndarray) -> np.ndarray:
        """观测均值 h(x)，支持 (n,) 与 (N, n) 批量。"""
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 1:
            if self._emission_fn is not None:
                return self._emission_fn(arr)
            if self.H is None:
                raise DGPError("模型无观测映射")
            return self.H @ arr
        if self._emission_fn is not None:
            return np.stack([self._emission_fn(row) for row in arr])
        if self.H is None:
            raise DGPError("模型无观测映射")
        return arr @ self.H.T

    def sample_transition(self, x: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """``x ~ p(x_t | x_{t-1})``，支持 (n,) 与 (N, n)。"""
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 1:
            return self.A @ arr + self._L_Q @ rng.randn(self.n_state)
        return arr @ self.A.T + rng.randn(arr.shape[0], self.n_state) @ self._L_Q.T

    def sample_emission(self, x: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 1:
            return self.emission(arr) + self._L_R @ rng.randn(self.n_obs)
        return self.emission(arr) + rng.randn(arr.shape[0], self.n_obs) @ self._L_R.T

    def cholesky(self) -> tuple[np.ndarray, float, np.ndarray, float]:
        """返回 ``(L_Q, logdet_Q, L_R, logdet_R)``。"""
        return self._L_Q, self._logdet_Q, self._L_R, self._logdet_R


def build_linear_gaussian(spec: LinearGaussianSpec, seed: int) -> ProbabilisticSSM:
    """构造线性高斯 SSM（固定 seed ⇒ 逐位可复现）。"""
    rng = np.random.RandomState(seed)
    n, m = spec.n_state, spec.n_obs
    A = _stable_a(n, spec.spectral_radius, rng)
    if spec.obs_matrix == "partial":
        H = partial_obs_matrix(n, m, rng)
    else:
        H = np.eye(m, n) if m <= n else rng.randn(m, n) / np.sqrt(n)
    return ProbabilisticSSM(
        A, H, np.eye(n) * spec.q_scale**2, np.eye(m) * spec.r_scale**2, name=spec.name
    )


def build_nonlinear(spec: NonlinearSpec, seed: int) -> ProbabilisticSSM:
    """构造线性状态转移 + 非线性观测的SSM。"""
    rng = np.random.RandomState(seed)
    n, m = spec.n_state, spec.n_obs
    A = _stable_a(n, spec.spectral_radius, rng)
    H = np.eye(m, n)  # 线性参考（UKF 在线性退化时用它）
    return ProbabilisticSSM(
        A,
        H,
        np.eye(n) * spec.q_scale**2,
        np.eye(m) * spec.r_scale**2,
        emission_fn=_make_emission(spec.nonlinearity),
        jacobian_fn=_make_jacobian(spec.nonlinearity),
        name=spec.name,
    )


def _make_emission(kind: str):
    """非线性发射函数工厂。"""
    table = {
        "quadratic": lambda x: np.array([x[0] ** 2 - x[1] ** 2, 2 * x[0] * x[1], x[2], x[3]]),
        "radar": lambda x: np.array([np.hypot(x[0], x[1]), np.arctan2(x[1], x[0]), x[2], x[3]]),
        "tanh": np.tanh,
    }
    if kind not in table:
        raise DGPError("未知非线性类型", detail=f"{kind!r} 不在 {sorted(table)}")
    fn = table[kind]
    if kind == "radar":
        base = fn
        # r 的下限：1/r 在 r→0 时爆炸会让 EKF/IEKF 的协方差溢出到 inf
        # （实测 x=0 附近即可触发）。取 r_min 使 1/r_min² · Q 与 R 同量级。
        r_min = 0.15

        def guarded(x: np.ndarray) -> np.ndarray:
            out = base(x)
            out[0] = max(out[0], r_min)
            return out

        return guarded
    return fn


def _make_jacobian(kind: str):
    """发射函数雅可比工厂（解析式，避免中心差分噪声）。"""

    def quad_jac(x: np.ndarray) -> np.ndarray:
        J = np.zeros((4, 4))
        J[0, 0], J[0, 1] = 2 * x[0], -2 * x[1]
        J[1, 0], J[1, 1] = 2 * x[1], 2 * x[0]
        J[2, 2] = 1.0
        J[3, 3] = 1.0
        return J

    def radar_jac(x: np.ndarray) -> np.ndarray:
        # r 下限必须与 _make_emission 的 r_min 一致，否则雅可比与实际发射函数脱节
        r = max(float(np.hypot(x[0], x[1])), 0.15)
        J = np.zeros((4, 4))
        J[0, 0], J[0, 1] = x[0] / r, x[1] / r
        J[1, 0], J[1, 1] = -x[1] / (r * r), x[0] / (r * r)
        J[2, 2] = 1.0
        J[3, 3] = 1.0
        return J

    def tanh_jac(x: np.ndarray) -> np.ndarray:
        return np.diag(1.0 - np.tanh(x) ** 2)

    table = {"quadratic": quad_jac, "radar": radar_jac, "tanh": tanh_jac}
    if kind not in table:
        raise DGPError("未知非线性类型", detail=f"{kind!r} 不在 {sorted(table)}")
    return table[kind]
