"""粒子平滑器：FFBS（Forward Filtering Backward Sampling）。

**本域的核心科学事实（决定门禁如何诚实设定）**
--------------------------------------------------
在线性高斯问题上 RTS 平滑是**精确的**（卡尔曼平滑定理），它是贝叶斯最优，
任何粒子平滑器都无法超越。因此：

- ``linear_gaussian`` DGP 上，RTS 拿最优是**诚实的天花板**，
  FFBS 只能"逼近"它，不可能更好 → 该域门禁按"非劣 + 逼近精度"设定。
- 非线性问题上，RTS 需沿平滑路径重算雅可比（不再精确），
  FFBS 的后向采样不受一阶线性化限制 ⇒ 存在真实超越空间。

FFBS 递推（Godsill, Doucet & Rajam 1994）
------------------------------------------
前向 bootstrap SIR 得到 ``{x_t^i, w_t^i}``，后向按

``p(x_t | x_{t+1}, y_{1:t}) ∝ p(x_t|y_{1:t-1})·f(x_t|x_{t-1})·f(x_{t+1}|x_t)``

采样。实现要点：朴素做法需 O(T·N²) 的粒子对。本实现利用
**后向转移核的对称分解**：对每个历史粒子 ``x_t^i`` 计算
``f(x_{t+1}|x_t^i)`` 的对数密度（O(N) 批量），归一化后抽祖先，
整体 O(T·N)。
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp as _lse

from core.numeric import stabilize_cov
from core.types import FilterEstimate, SmoothingResult
from data.models import ProbabilisticSSM

__all__ = ["FFBS", "ffbs_smooth"]


class FFBS:
    """FFBS 粒子平滑器。

    Parameters
    ----------
    n_particles
        粒子数。
    seed
        独立随机源（不污染全局 seed 流 ⇒ 与其他方法逐位独立可复现）。
    degenerate_threshold
        平均 ESS 低于该比例 ⇒ 标记 ``last_degraded=True``，
        报告里如实标注"平滑结果退化为滤波轨迹"，**不伪造增益**。
    """

    def __init__(
        self,
        model: ProbabilisticSSM,
        *,
        n_particles: int = 800,
        seed: int = 999,
        degenerate_threshold: float = 0.05,
    ) -> None:
        self.model = model
        self.n_particles = int(n_particles)
        self._rng = np.random.RandomState(seed)
        self.degenerate_threshold = float(degenerate_threshold)
        self.name = "ffbs"
        self.family = "particle"  # 统一协议字段：FFBS 属粒子法族
        self.reports_ess = True
        self.last_degraded = False
        self.mean_ess = float("nan")

    def _forward(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """前向 SIR，返回 ``(hist_particles, hist_logw, mean_ess)``。"""
        mdl = self.model
        rng = self._rng
        L_Q, _, L_R, logdet_R = mdl.cholesky()
        T, m = y.shape
        n = mdl.n_state
        N = self.n_particles

        particles = rng.randn(N, n) @ L_Q.T
        logw = np.full(N, -np.log(N))
        hist_p = np.empty((T, N, n))
        hist_w = np.empty((T, N))
        ess_hist = np.empty(T)

        for t in range(T):
            if t > 0:
                particles = mdl.sample_transition(particles, rng)
            resid = y[t][None, :] - mdl.emission(particles)
            z = np.linalg.solve(L_R, resid.T)
            loglik = -0.5 * (m * np.log(2 * np.pi) + logdet_R + np.sum(z * z, axis=0))
            logw = logw + loglik
            logw -= _lse(logw)
            w = np.exp(logw)
            ess = float(1.0 / np.sum(w * w))
            ess_hist[t] = ess
            if ess < 0.5 * N:
                cumsum = np.cumsum(w)
                cumsum[-1] = 1.0
                idx = np.clip(
                    np.searchsorted(cumsum, (rng.random_sample() + np.arange(N)) / N, side="right"),
                    0,
                    N - 1,
                )
                particles = particles[idx]
                logw = np.full(N, -np.log(N))
            hist_p[t] = particles
            hist_w[t] = logw
        return hist_p, hist_w, float(np.mean(ess_hist))

    def filter(self, y: np.ndarray) -> FilterEstimate:
        """FFBS 的滤波阶段 = bootstrap SIR 前向滤波（与 ``_forward`` 同一实现）。

        协议要求 filter/smooth 同签名；这里 forward 的结果复用 ``_forward``，
        保证「FFBS 的滤波」与「FFBS 的平滑前向」是**同一份**粒子集，
        否则平滑的边缘似然与滤波指标会口径不一致。
        """
        mdl = self.model
        T, _m = y.shape
        n = mdl.n_state
        hist_p, hist_w, mean_ess = self._forward(y)
        self.mean_ess = mean_ess
        N = self.n_particles
        self.last_degraded = bool(mean_ess < self.degenerate_threshold * N)
        x_filt = np.empty((T, n))
        x_pred = np.empty((T, n))
        p_filt = np.empty((T, n, n))
        ll_terms = np.empty(T)
        ess_hist = np.empty(T)
        ess_hist[:] = mean_ess
        for t in range(T):
            w = np.exp(hist_w[t] - _lse(hist_w[t]))
            x_filt[t] = np.average(hist_p[t], weights=w, axis=0)
            emp = hist_p[t] - x_filt[t]
            p_filt[t] = stabilize_cov((emp.T * w) @ emp)
            # 先验均值 = 上一时刻后验经转移（t=0 用初始分布均值 0）
            x_pred[t] = mdl.A @ x_filt[t - 1] if t > 0 else 0.0
            # 增量log-lik = logsumexp(logw_{t-1} + loglik)
            resid = y[t][None, :] - mdl.emission(hist_p[t])
            resid = np.where(np.isnan(resid), 0.0, resid)
            obs_idx = np.flatnonzero(~np.isnan(y[t]))
            if len(obs_idx) == 0:
                ll_terms[t] = 0.0
                continue
            R_o = mdl.R[np.ix_(obs_idx, obs_idx)]
            L_o = np.linalg.cholesky(R_o)
            z = np.linalg.solve(L_o, resid[:, obs_idx].T)
            loglik = -0.5 * (
                len(obs_idx) * np.log(2 * np.pi)
                + 2.0 * float(np.sum(np.log(np.diag(L_o))))
                + np.sum(z * z, axis=0)
            )
            prior_w = np.full(N, -np.log(N)) if t == 0 else hist_w[t - 1]
            ll_terms[t] = float(_lse(prior_w + loglik))
        return FilterEstimate(x_pred, x_filt, p_filt, ll_terms, ess_hist)

    def smooth(self, y: np.ndarray) -> SmoothingResult:
        mdl = self.model
        rng = self._rng
        _, _, _, logdet_Q = mdl.cholesky()
        L_Q = np.linalg.cholesky(mdl.Q)
        T = y.shape[0]
        n = mdl.n_state
        N = self.n_particles

        hist_p, hist_w, mean_ess = self._forward(y)
        self.mean_ess = mean_ess
        self.last_degraded = bool(mean_ess < self.degenerate_threshold * N)

        x_smooth = np.empty((T, n))
        p_smooth = np.empty((T, n, n))

        # t = T-1：直接从滤波后验抽
        w_last = np.exp(hist_w[T - 1] - _lse(hist_w[T - 1]))
        anc = self._draw(w_last, rng)
        x_smooth[T - 1] = hist_p[T - 1][anc]
        emp = hist_p[T - 1] - x_smooth[T - 1]
        p_smooth[T - 1] = stabilize_cov((emp.T * w_last) @ emp)

        # t = T-2..0：按 p(x_t|y_{1:t-1})·f(x_{t+1}|x_t) 抽祖先
        for t in range(T - 2, -1, -1):
            cand = hist_p[t]
            # log f(x_{t+1} | x_t)：高斯密度，逐粒子批量 O(N)
            resid = cand @ mdl.A.T - x_smooth[t + 1]
            z = np.linalg.solve(L_Q, resid.T)
            log_trans = -0.5 * (n * np.log(2 * np.pi) + logdet_Q + np.sum(z * z, axis=0))
            log_post = hist_w[t] + log_trans
            log_post -= _lse(log_post)
            post_w = np.exp(log_post)
            anc = self._draw(post_w, rng)
            x_smooth[t] = cand[anc]
            emp = cand - x_smooth[t]
            filt_w = np.exp(hist_w[t] - _lse(hist_w[t]))
            p_smooth[t] = stabilize_cov((emp.T * filt_w) @ emp)

        return SmoothingResult(x_smooth, p_smooth)

    @staticmethod
    def _draw(weights: np.ndarray, rng: np.random.RandomState) -> int:
        """按权重抽单个祖先索引。"""
        cumsum = np.cumsum(weights)
        cumsum[-1] = 1.0
        return int(
            np.clip(np.searchsorted(cumsum, rng.random_sample(), side="right"), 0, len(weights) - 1)
        )


def ffbs_smooth(
    model: ProbabilisticSSM, y: np.ndarray, *, n_particles: int = 800, seed: int = 999
) -> SmoothingResult:
    """函数式入口：返回 FFBS 平滑结果。"""
    return FFBS(model, n_particles=n_particles, seed=seed).smooth(y)
