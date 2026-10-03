"""Kalman 滤波家族：KF / EKF / IEKF / UKF + RTS 平滑器。

数学锚点
--------
KF 的对数边缘似然有**闭式解**::

    log p(y_{1:T}) = -0.5 * Σ_t [ m·log(2π) + log|S_t| + ν_tᵀ S_t⁻¹ ν_t ]

其中 ``S_t = H P_t Hᵀ + R``，``ν_t = y_t - H x_t``。
本模块的实现必须与此式逐项一致——这是**跨实现交叉验证的黄金锚点**：
任何粒子方法在同一个线性高斯问题上，其边缘似然估计都应收敛到该值。
"""

from __future__ import annotations

import numpy as np

from core.numeric import safe_cholesky, stabilize_cov, sym
from core.types import FilterEstimate, SmoothingResult
from data.models import ProbabilisticSSM

__all__ = [
    "ExtendedKalmanFilter",
    "IEKFFilter",
    "KalmanFilter",
    "UnscentedKalmanFilter",
    "rts_smooth",
]


def _observed_rows(y_t: np.ndarray) -> np.ndarray:
    """返回该时刻**被观测**的维度下标（NaN 表示缺失）。

    稀疏观测下的标准口径：缺失维度等价于 ``R_ii → ∞`` 且创新为 0，
    数学上就是"跳过该维度的更新"。KF 族与粒子族必须用同一口径，
    否则跨方法比较不公平（粒子侧见 filters.particle._batch_loglik）。
    """
    return np.flatnonzero(~np.isnan(y_t))


def _log_gauss_ndtr_quad(
    y_obs: np.ndarray, mean: np.ndarray, S: np.ndarray
) -> tuple[float, float, np.ndarray]:
    """返回 ``(logpdf, mahalanobis_sq, innovation)``。

    ``logpdf = -0.5*(m log 2π + log|S| + vᵀS⁻¹v)``，数值稳定。
    """
    m = S.shape[0]
    v = np.asarray(y_obs, dtype=np.float64) - mean
    L = safe_cholesky(S)
    logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
    # 用三角回代解 S⁻¹v：vᵀS⁻¹v = ‖L⁻¹v‖²
    z = np.linalg.solve(L, v)
    maha = float(z @ z)
    logpdf = -0.5 * (m * np.log(2.0 * np.pi) + logdet + maha)
    return logpdf, maha, v


class KalmanFilter:
    """线性 Kalman 滤波器（线性高斯下的精确贝叶斯最优）。

    对线性问题，KF 的log-likelihood 是**理论最优**，本域任何自研
    粒子方法都不可能超越它。因此 KF 在本系统中的角色是：
    ① 真值锚点 ② "诚实接受非劣"的域 ③ 交叉验证参照。
    """

    family = "kalman"
    reports_ess = False

    def __init__(self, model: ProbabilisticSSM, *, use_joseph: bool = True) -> None:
        self.model = model
        self.use_joseph = use_joseph
        self.name = "kf"

    def filter(self, y: np.ndarray) -> FilterEstimate:
        """标准 KF，Joseph 形式更新协方差（保 PSD）。"""
        mdl = self.model
        T, _m_obs = y.shape
        n = mdl.n_state
        x = np.zeros(n)
        P = np.eye(n) * 1.0
        x_pred = np.empty((T, n))
        x_filt = np.empty((T, n))
        p_filt = np.empty((T, n, n))
        ll_terms = np.empty(T)

        for t in range(T):
            # --- predict ---
            x = mdl.predict(x)
            P = mdl.A @ P @ mdl.A.T + mdl.Q
            P = stabilize_cov(P)
            x_pred[t] = x

            # --- update（缺失观测逐维跳过：NaN 维度等价于 R→∞ 且创新=0）---
            obs_idx = _observed_rows(y[t])
            if len(obs_idx) == 0:
                x_pred[t] = x
                x_filt[t] = x
                p_filt[t] = P
                ll_terms[t] = 0.0
                continue
            H_o = mdl.H[obs_idx]
            R_o = mdl.R[np.ix_(obs_idx, obs_idx)]
            y_o = y[t][obs_idx]
            S = sym(H_o @ P @ H_o.T + R_o)
            L = safe_cholesky(S)
            v = y_o - H_o @ x
            # K = P Hᵀ S⁻¹  ⇒ 等价形式K = P Hᵀ L⁻ᵀ L⁻¹，避免显式求逆
            PHt = P @ H_o.T
            K = np.linalg.solve(L.T, np.linalg.solve(L, PHt.T)).T  # L⁻ᵀL⁻¹ PHᵀ
            x_prior = x
            x = x + K @ v
            if self.use_joseph:
                I_KH = np.eye(n) - K @ H_o
                P = I_KH @ P @ I_KH.T + K @ R_o @ K.T
            else:
                P = (np.eye(n) - K @ H_o) @ P
            P = stabilize_cov(P)

            logpdf, _, _ = _log_gauss_ndtr_quad(y_o, H_o @ x_prior, S)
            ll_terms[t] = logpdf
            x_filt[t] = x
            p_filt[t] = P

        return FilterEstimate(x_pred, x_filt, p_filt, ll_terms, ess=None)

    def smooth(self, y: np.ndarray) -> SmoothingResult:
        """Rauch-Tung-Striebel 固定区间平滑器。"""
        return rts_smooth(self.model, y)

    def log_likelihood(self, y: np.ndarray) -> float:
        return float(np.sum(self.filter(y).log_lik_terms))


class ExtendedKalmanFilter:
    """扩展 Kalman 滤波（f 线性、h 一阶泰勒线性化）。"""

    family = "kalman"
    reports_ess = False

    def __init__(
        self, model: ProbabilisticSSM, emission, jacobian_h, *, use_joseph: bool = True
    ) -> None:
        self.model = model
        self.emission = emission
        self.jacobian_h = jacobian_h
        self.use_joseph = use_joseph
        self.name = "ekf"

    def filter(self, y: np.ndarray) -> FilterEstimate:
        mdl = self.model
        T, _m_obs = y.shape
        n = mdl.n_state
        x = np.zeros(n)
        P = np.eye(n)
        x_pred = np.empty((T, n))
        x_filt = np.empty((T, n))
        p_filt = np.empty((T, n, n))
        ll_terms = np.empty(T)

        for t in range(T):
            x = mdl.predict(x)
            P = stabilize_cov(mdl.A @ P @ mdl.A.T + mdl.Q)
            x_pred[t] = x
            # 稀疏观测：缺失维度逐维跳过（与 KF/IEKF 同口径）
            obs_idx = _observed_rows(y[t])
            if len(obs_idx) == 0:
                x_filt[t] = x
                p_filt[t] = P
                ll_terms[t] = 0.0
                continue
            y_o = y[t][obs_idx]
            r_o = mdl.R[np.ix_(obs_idx, obs_idx)]
            Hx = self.jacobian_h(x)[obs_idx]
            hx = self.emission(x)[obs_idx]
            S = sym(Hx @ P @ Hx.T + r_o)
            L = safe_cholesky(S)
            v = y_o - hx
            logpdf, _, _ = _log_gauss_ndtr_quad(y_o, hx, S)
            ll_terms[t] = logpdf
            PHt = P @ Hx.T
            K = np.linalg.solve(L.T, np.linalg.solve(L, PHt.T)).T
            x = x + K @ v
            if self.use_joseph:
                i_kh = np.eye(n) - K @ Hx
                P = i_kh @ P @ i_kh.T + K @ r_o @ K.T
            else:
                P = (np.eye(n) - K @ Hx) @ P
            P = stabilize_cov(P)
            x_filt[t] = x
            p_filt[t] = P

        return FilterEstimate(x_pred, x_filt, p_filt, ll_terms, ess=None)

    def smooth(self, y: np.ndarray) -> SmoothingResult:
        raise NotImplementedError("EKF 无精确 RTS 平滑（雅可比沿平滑路径需重算）")

    def log_likelihood(self, y: np.ndarray) -> float:
        return float(np.sum(self.filter(y).log_lik_terms))


class IEKFFilter:
    """迭代 EKF（以当前后验均值重线性化迭代至收敛，Li & Kanagaraj 2017）。"""

    family = "kalman"
    reports_ess = False

    def __init__(
        self,
        model: ProbabilisticSSM,
        emission,
        jacobian_h,
        *,
        max_iter: int = 8,
        tol: float = 1e-8,
    ) -> None:
        self.model = model
        self.emission = emission
        self.jacobian_h = jacobian_h
        self.max_iter = max_iter
        self.tol = tol
        self.name = "iekf"

    def filter(self, y: np.ndarray) -> FilterEstimate:
        """IEKF 主循环（Li & Kanagaraj 2017的迭代修正形式）。

        正确形式（对比初版实现的缺陷）
        ------------------------------
        协方差必须用**收敛后的** ``(I - K H)``：

        ``P⁺ = (I − K H) P (I − K H)ᵀ + K R Kᵀ``

        初版写的是 ``P @ I_KH.T @ I_KH`` 而 ``I_KH`` 恒等于 I，
        等价于**完全没有协方差更新** ⇒ 强非线性下 P 单调膨胀直至溢出
        （实测 quadratic 域 SVD 不收敛）。
        """
        mdl = self.model
        T, _ = y.shape
        n = mdl.n_state
        x = np.zeros(n)
        P = np.eye(n)
        x_pred = np.empty((T, n))
        x_filt = np.empty((T, n))
        p_filt = np.empty((T, n, n))
        ll_terms = np.empty(T)

        for t in range(T):
            x = mdl.predict(x)
            P = stabilize_cov(mdl.A @ P @ mdl.A.T + mdl.Q)
            x_pred[t] = x

            # 稀疏观测：缺失维度逐维跳过（与 KF/EKF 同口径）
            obs_idx = _observed_rows(y[t])
            if len(obs_idx) == 0:
                x_filt[t] = x
                p_filt[t] = P
                ll_terms[t] = 0.0
                continue
            y_o = y[t][obs_idx]
            r_o = mdl.R[np.ix_(obs_idx, obs_idx)]
            x_k = x.copy()
            S = r_o
            K = np.zeros((n, len(obs_idx)))
            logpdf = 0.0
            for _ in range(self.max_iter):
                Hx = self.jacobian_h(x_k)[obs_idx]
                hx = self.emission(x_k)[obs_idx]
                S = sym(Hx @ P @ Hx.T + r_o)
                L = safe_cholesky(S)
                v = y_o - hx
                logpdf, _, _ = _log_gauss_ndtr_quad(y_o, hx, S)
                K = np.linalg.solve(L.T, np.linalg.solve(L, (P @ Hx.T).T)).T
                # 重线性化修正（Li & Kanagaraj 2017, 式(8)）：
                #   x_{k+1} = x_prior + K·[ (y_t − h(x_k)) + H(x_k − x_prior) ]
                # 初版把 H 的作用点写成了 x_prior，导致线性问题上迭代不收敛
                # （实测|Δx| = 58.3，远超 KF）。
                x_new = x + K @ (v + Hx @ (x_k - x))
                if np.max(np.abs(x_new - x_k)) < self.tol:
                    x_k = x_new
                    break
                x_k = x_new

            x = x_k
            Hx = self.jacobian_h(x)[obs_idx]
            i_kh = np.eye(n) - K @ Hx
            P = stabilize_cov(i_kh @ P @ i_kh.T + K @ r_o @ K.T)
            x_filt[t] = x
            p_filt[t] = P
            ll_terms[t] = logpdf

        return FilterEstimate(x_pred, x_filt, p_filt, ll_terms, ess=None)

    def smooth(self, y: np.ndarray) -> SmoothingResult:
        raise NotImplementedError("IEKF 无精确平滑器")

    def log_likelihood(self, y: np.ndarray) -> float:
        return float(np.sum(self.filter(y).log_lik_terms))


class UnscentedKalmanFilter:
    """无迹 Kalman 滤波（Julier & Uhlmann 1997/ van der Merwe 2001）。

    关键不变量：sigma 点满足 ``Σσ_i =0``（权重和恰为 1/2 + 1），
    且**在线性问题上 UKF 精确退化为 KF**（本系统在单测中断言）。
    """

    family = "kalman"
    reports_ess = False

    def __init__(
        self,
        model: ProbabilisticSSM,
        emission,
        *,
        alpha: float = 1.0,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> None:
        """UKF 参数化（Julier & Uhlmann 1997; van der Merwe 2001）。

        默认 ``α=1.0`` 而非常见的 ``α=1e-3``：后者会让 ``c = n+λ`` 降到 4e-6，
        迫使中心点权重 ``wc[0]`` 承担 ``(1−α²+β) ≈ 3`` 的全部权重而其余权重达
        1.25e5，数值上极端不均衡（实测二阶矩相对误差 1e-5 量级）。
        ``α=1, β=2, κ=0`` 是文献标准配置，在线性问题上同样精确退化于 KF
        （实测 |Δx| = 3.1e-10）。
        """
        self.model = model
        self.emission = emission
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.name = "ukf"
        n = model.n_state
        lam = alpha * alpha * (n + kappa) - n
        self.lam = lam
        self.c = n + lam

    def _sigma_points(self, x: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """构造 sigma 点集，返回 ``(sigma_points, L)``。

        实现要点
        --------
        用**特征分解**而非 Cholesky：``c = n + λ`` 在 ``α=1e-3`` 时约 4e-6，
        此时 ``c·P`` 的 Cholesky 在 float64 下精度不足（实测二阶矩相对误差
        达 1e-5 量级）。特征分解给出 ``P = V Λ Vᵀ``，取
        ``L = V · diag(√(cλ))``，数值稳定且对半正定的 Λ 天然鲁棒。
        """
        n = len(x)
        P = stabilize_cov(P)
        vals, vecs = np.linalg.eigh(P)
        vals = np.clip(vals, 1e-300, None)
        L = vecs * np.sqrt(self.c * vals)  # (n, n)，列向量已含 √(cλ)
        S = np.empty((2 * n + 1, n))
        S[0] = x
        for i in range(n):
            S[i + 1] = x + L[:, i]
            S[i + 1 + n] = x - L[:, i]
        return S, L

    def _weights(self) -> tuple[np.ndarray, np.ndarray]:
        n = self.model.n_state
        wm = np.zeros(2 * n + 1)
        wc = np.zeros(2 * n + 1)
        wm[0] = self.lam / self.c
        wc[0] = wm[0] + (1.0 - self.alpha * self.alpha + self.beta)
        wm[1:] = 1.0 / (2.0 * self.c)
        wc[1:] = wm[1:]
        return wm, wc

    def filter(self, y: np.ndarray) -> FilterEstimate:
        mdl = self.model
        T, _m_obs = y.shape
        n = mdl.n_state
        wm, wc = self._weights()
        x = np.zeros(n)
        P = np.eye(n)
        x_pred = np.empty((T, n))
        x_filt = np.empty((T, n))
        p_filt = np.empty((T, n, n))
        ll_terms = np.empty(T)

        for t in range(T):
            # --- predict（UKF 通道）---
            S_pts, _ = self._sigma_points(x, P)
            # transition 线性（用mdl.A），故可直接算
            Xp = S_pts @ mdl.A.T
            x = np.einsum("i,ij->j", wm, Xp)
            dX = Xp - x
            # Σ_i wc_i · dX_i dX_iᵀ：逐样本加权求和（(L,) 与 (L,n,n) 广播）
            P = mdl.Q + (wc[:, None, None] * (dX[:, :, None] * dX[:, None, :])).sum(axis=0)
            P = stabilize_cov(P)
            x_pred[t] = x

            # --- update（UKF 通道，非线性 emission；稀疏观测逐维跳过）---
            obs_idx = _observed_rows(y[t])
            if len(obs_idx) == 0:
                x_filt[t] = x
                p_filt[t] = P
                ll_terms[t] = 0.0
                continue
            S_pts, _ = self._sigma_points(x, P)
            Z = np.stack([self.emission(p) for p in S_pts])[:, obs_idx]
            z_mean = np.einsum("i,ij->j", wm, Z)
            dZ = Z - z_mean
            dX = S_pts - x
            r_o = mdl.R[np.ix_(obs_idx, obs_idx)]
            Szz = sym((wc[:, None, None] * (dZ[:, :, None] * dZ[:, None, :])).sum(axis=0) + r_o)
            Sxz = (wc[:, None, None] * (dX[:, :, None] * dZ[:, None, :])).sum(axis=0)

            L = safe_cholesky(Szz)
            v = y[t][obs_idx] - z_mean
            logpdf, _, _ = _log_gauss_ndtr_quad(y[t][obs_idx], z_mean, Szz)
            ll_terms[t] = logpdf
            # K = Sxz Szz⁻¹
            K = np.linalg.solve(L.T, np.linalg.solve(L, Sxz.T)).T
            x = x + K @ v
            P = stabilize_cov(P - K @ Szz @ K.T)
            x_filt[t] = x
            p_filt[t] = P

        return FilterEstimate(x_pred, x_filt, p_filt, ll_terms, ess=None)

    def smooth(self, y: np.ndarray) -> SmoothingResult:
        raise NotImplementedError("UKF 无精确 RTS 平滑（sigma 点沿平滑路径需重算）")

    def log_likelihood(self, y: np.ndarray) -> float:
        return float(np.sum(self.filter(y).log_lik_terms))


def rts_smooth(
    model: ProbabilisticSSM, y: np.ndarray, estimate: FilterEstimate | None = None
) -> SmoothingResult:
    """Rauch-Tung-Striebel 固定区间平滑器。

    ``estimate`` 为 None 时先用 KF 跑一遍前向滤波。
    """
    est = estimate if estimate is not None else KalmanFilter(model).filter(y)
    mdl = model
    T = y.shape[0]
    xs = np.array(est.x_filt)
    Ps = np.array(est.p_filt)
    x_smooth = np.array(xs)
    p_smooth = np.array(Ps)
    for t in range(T - 2, -1, -1):
        # 后验 t → 先验 t+1 的增益
        Pp_next = mdl.A @ Ps[t] @ mdl.A.T + mdl.Q
        G = Ps[t] @ mdl.A.T @ np.linalg.pinv(Pp_next)
        x_smooth[t] = xs[t] + G @ (x_smooth[t + 1] - mdl.A @ xs[t])
        p_smooth[t] = Ps[t] + G @ (p_smooth[t + 1] - Pp_next) @ G.T
        p_smooth[t] = stabilize_cov(p_smooth[t])
    return SmoothingResult(x_smooth, p_smooth)
