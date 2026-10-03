"""粒子滤波家族：bootstrap SIR / 辅助变量 APF / Rao-Blackwellised PF。

数学要点
--------
**bootstrap / SIR**（Gordon, Salmond & Smith 1993）
    权重递推 ``w_t ∝ w_{t-1}·p(y_t|x_t)``；ESS < threshold 时重采样。
    已知缺陷：提议分布就是先验 ``p(x_t|x_{t-1})``，**不含观测信息** ⇒
    高维下重要性权重集中，ESS 指数崩溃，边缘似然严重低估。

**辅助变量APF**（Pitt & Shephard 1999）
    引入辅助变量 ``a_t``，一阶边缘化为 ``q(x_t) ∝ p(y_t|x_t)p(x_t|x_{t-1})w_{t-1}(x_{t-1})``。
    提议已吸收观测 ⇒ 权重方差大降 ⇒ ESS 与边缘似然精度提升。

**Rao-Blackwellised PF**（Doucet & Singh 2001；机器人定位标准架构）
    状态拆为 ``(z, x_lin)``：``z`` 低维高非线性 → 粒子采样；
    ``x_lin`` 线性高斯 → **KF 解析**。于是
    ``log p(y_{1:T}) = log E_{particles}[ p(y_{1:T} | z_{1:T}, KF-part) ]``。
    在线性高斯子问题上这**是精确的**，bootstrap PF 只能逼近。

数值约定
--------
全部权重在**log 域**累加（``scipy.special.logsumexp``）；
r_scale=0.05 的窄似然场景在概率域会直接下溢到 0。

随机性约定
----------
随机源由构造参数 ``seed`` 注入的 legacy ``np.random.RandomState``，
不经全局 seed 流⇒不同粒子法之间互不干扰，且同参数逐位可复现
（踩坑库 §A：``np.random.Generator`` 流跨 numpy 版本不稳定，粒子法是随机算法，
门禁会在 CI 上随机翻车）。
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp as _lse

from core.errors import NumericalError
from core.numeric import stabilize_cov
from core.types import FilterEstimate, SmoothingResult

__all__ = [
    "AuxiliaryParticleFilter",
    "BootstrapSIR",
    "ess_from_logw",
    "resample_multinomial",
    "resample_systematic",
]


def resample_systematic(
    weights: np.ndarray, n_particles: int, rng: np.random.RandomState
) -> np.ndarray:
    """系统重采样（低方差，O(N)）。"""
    positions = (rng.random_sample() + np.arange(n_particles)) / n_particles
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0  # 防浮点累积导致索引越界
    return np.clip(np.searchsorted(cumsum, positions, side="right"), 0, n_particles - 1)


def resample_multinomial(
    weights: np.ndarray, n_particles: int, rng: np.random.RandomState
) -> np.ndarray:
    """多项式重采样（经典 bootstrap，高方差）。"""
    w = weights / weights.sum()
    return rng.choice(len(weights), size=n_particles, replace=True, p=w)


_RESAMPLERS = {"systematic": resample_systematic, "multinomial": resample_multinomial}


def ess_from_logw(logw: np.ndarray) -> float:
    """有效样本量 ``ESS = 1/Σw²``（在归一化权重上计算）。"""
    w = np.exp(logw - _lse(logw))
    return float(1.0 / np.sum(w**2))


def _batch_loglik(
    resid: np.ndarray, L_R: np.ndarray, logdet_R: float, cols: np.ndarray | None = None
) -> np.ndarray:
    """对 (N, m) 残差批量算 ``log N(resid; 0, S)``（log 域）。

    Parameters
    ----------
    resid
        (N, m) 残差。**允许 NaN**：NaN 表示该观测维度缺失
        （sparse_obs DGP 的稀疏观测），缺失维度不贡献似然。
    L_R
        (k, k) 下三角，``S = L_R L_Rᵀ``。
    logdet_R
        ``log|S|``，与 ``L_R`` 维度一致。
    cols
        参与计算的残差列索引。``None`` 时按 NaN 自动判定：
        非 NaN 的列即被观测列。
    """
    has_nan = bool(np.isnan(resid).any())
    if cols is None:
        cols = np.flatnonzero(~np.isnan(resid[0])) if has_nan else np.arange(resid.shape[1])
    if len(cols) == 0:
        return np.zeros(resid.shape[0])
    sub = resid[:, cols]
    sub = np.where(np.isnan(sub), 0.0, sub)
    z = np.linalg.solve(L_R, sub.T)
    k = L_R.shape[0]
    return -0.5 * (k * np.log(2.0 * np.pi) + logdet_R + np.sum(z * z, axis=0))


def _safe_resid(y_t: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """构造残差 ``(N, m)``，把观测缺失维度保留为 NaN。

    ``y_t`` 里的 NaN 表示该维度未观测（稀疏观测 DGP）。这些维度的残差
    写成 NaN，由 :func:`_batch_loglik` 在似然里剔除。
    预测值 ``pred``（(N, m)）会被复制，不修改原数组。
    """
    return y_t[None, :] - pred


class _ParticleFilterBase:
    """粒子滤波骨架：ESS 记录、log 域权重、重采样。"""

    family = "particle"
    reports_ess = True

    def __init__(
        self,
        model,
        *,
        n_particles: int = 1200,
        resample: str = "systematic",
        ess_threshold_ratio: float = 0.5,
        seed: int = 12345,
    ) -> None:
        if resample not in _RESAMPLERS:
            raise NumericalError("未知重采样策略", detail=f"{resample!r}")
        if n_particles < 50:
            raise NumericalError("n_particles 过小（ESS 诊断无意义）", detail=str(n_particles))
        self.model = model
        self.n_particles = int(n_particles)
        self._resampler = _RESAMPLERS[resample]
        self.ess_threshold = float(ess_threshold_ratio * n_particles)
        self._rng = np.random.RandomState(seed)
        self.name = self.__class__.__name__.lower()

    def log_likelihood(self, y: np.ndarray) -> float:
        return float(np.sum(self.filter(y).log_lik_terms))

    def smooth(self, y: np.ndarray) -> SmoothingResult:
        raise NotImplementedError("粒子平滑器见 seqforge.smoothers")

    # --- 子类需实现的公共循环骨架 ---
    def _init_particles(self) -> np.ndarray:
        rng = self._rng
        n = self.model.n_state
        L_Q, _, _, _ = self.model.cholesky()
        return rng.randn(self.n_particles, n) @ L_Q.T

    def _finalize_step(
        self,
        t: int,
        particles: np.ndarray,
        logw: np.ndarray,
        x_filt: np.ndarray,
        p_filt: np.ndarray,
        x_pred: np.ndarray,
        ll_terms: np.ndarray,
        ess_arr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """公共的统计汇总 + 自适应重采样决策。"""
        ess = ess_from_logw(logw)
        ess_arr[t] = ess
        w = np.exp(logw - _lse(logw))
        mean = np.average(particles, weights=w, axis=0)
        emp = particles - mean
        x_filt[t] = mean
        p_filt[t] = stabilize_cov((emp.T * w) @ emp)
        if ess < self.ess_threshold:
            idx = self._resampler(w, self.n_particles, self._rng)
            particles = particles[idx]
            logw = np.full(self.n_particles, -np.log(self.n_particles))
        return particles, logw


class BootstrapSIR(_ParticleFilterBase):
    """Bootstrap / SIR 粒子滤波—— 本域最强传统基线。

    提议 ``q(x_t)=p(x_t|x_{t-1})``（先验提议，不含观测）⇒ 高维下 ESS 崩溃。
    """

    def __init__(
        self, model, *, n_particles: int = 1200, resample: str = "systematic", seed: int = 12345
    ) -> None:
        super().__init__(model, n_particles=n_particles, resample=resample, seed=seed)
        self.name = "bootstrap_sir"

    def filter(self, y: np.ndarray) -> FilterEstimate:
        rng = self._rng
        _, _, _L_R_unused, _logdet_R_unused = self.model.cholesky()
        T = y.shape[0]
        n = self.model.n_state
        particles = self._init_particles()
        logw = np.full(self.n_particles, -np.log(self.n_particles))

        x_filt = np.empty((T, n))
        x_pred = np.empty((T, n))
        p_filt = np.empty((T, n, n))
        ll_terms = np.empty(T)
        ess_arr = np.empty(T)

        for t in range(T):
            if t > 0:
                particles = self.model.sample_transition(particles, rng)
            x_pred[t] = particles.mean(axis=0)

            resid = _safe_resid(y[t], self.model.emission(particles))
            obs_idx_s = np.flatnonzero(~np.isnan(y[t]))
            if len(obs_idx_s) == 0:
                ll_terms[t] = 0.0
                particles, logw = self._finalize_step(
                    t, particles, logw, x_filt, p_filt, x_pred, ll_terms, ess_arr
                )
                continue
            R_s = self.model.R[np.ix_(obs_idx_s, obs_idx_s)]
            L_s = np.linalg.cholesky(R_s)
            loglik = _batch_loglik(
                resid, L_s, 2.0 * float(np.sum(np.log(np.diag(L_s)))), cols=obs_idx_s
            )
            ll_terms[t] = float(_lse(logw + loglik))
            logw = logw + loglik
            logw -= _lse(logw)
            particles, logw = self._finalize_step(
                t, particles, logw, x_filt, p_filt, x_pred, ll_terms, ess_arr
            )

        return FilterEstimate(x_pred, x_filt, p_filt, ll_terms, ess_arr)


class AuxiliaryParticleFilter(_ParticleFilterBase):
    """辅助变量粒子滤波（Pitt & Shephard 1999）—— **逐祖先最优提议**。

    算法（严格形式，实现与论文一致）
    ----------------------------------
    条件于第 t-1 步的后验粒子集 ``{x_{t-1}^i, w_{t-1}^i}``，
    目标是 ``π(x_t) ∝ p(y_t|x_t)·p(x_t|x_{t-1})``（线性高斯下即后验）：

    **阶段 1 · 辅助变量**
        ``α_i ∝ w_{t-1}^i·p(y_t|x_{t-1}^i)``，其中对 x_t 做了边缘化::

            p(y_t|x_{t-1}^i) = N(y_t; h(m_i), H P_i Hᵀ + R),
            m_i = A x_{t-1}^i,  P_i = A P_{t-1} Aᵀ + Q

        ``P_{t-1}`` 用粒子经验协方差（逐时刻重算），而非固定名义值——
        固定名义值会让提议与真实后验失配，是实测 ESS 掉到 0.1% 的根因。

    **阶段 2 · 最优提议 + 严格重要性权重**
        对被选中的祖先 ``a``，从**条件最优提议**采样::

            q(x_t|x_{t-1}^a, y_t) = N(x_t; m_a + K'(y_t - h(m_a)), P_pred - K' H P_pred)
            K' = P_pred Hᵀ (H P_pred Hᵀ + R)⁻¹

        权重（重要性采样三件套，缺一即有偏）::

            w_t ∝ p(y_t|x_t) · p(x_t|x_{t-1}^a) / q(x_t|x_{t-1}^a, y_t)

        线性高斯下 ``p(y_t|x_t)p(x_t|x_{t-1}^a) = q(·)·p(y_t|x_{t-1}^a)``，
        故权重≈常数 ⇒ ESS 不随维度崩塌。这正是 APF 的价值所在。

    **阶段 3 · 边缘似然**
        ``log p(y_t|y_{1:t-1}) = logsumexp(log w_{t-1} + log α)``，
        即用辅助变量边缘化后的预测似然。它与后验权重是**两件事**，
        必须分开记录（踩坑库：混在一起会让 log-lik 随 N 漂移）。

    Parameters
    ----------
    resample_aux
        True ⇒ 每步按 α 做祖先重采样（Gordon 原文）；这是 Pitt & Shephard
        推荐的做法，因为它把多样性损失留给阶段 2 的最优提议来补偿。
    """

    def __init__(
        self,
        model,
        *,
        n_particles: int = 1200,
        resample: str = "systematic",
        seed: int = 12345,
        resample_aux: bool = True,
    ) -> None:
        super().__init__(model, n_particles=n_particles, resample=resample, seed=seed)
        self.resample_aux = resample_aux
        self.name = "auxiliary_pf"

    def filter(self, y: np.ndarray) -> FilterEstimate:
        rng = self._rng
        n = self.model.n_state
        T = y.shape[0]
        N = self.n_particles
        particles = self._init_particles()
        logw = np.full(N, -np.log(N))

        x_filt = np.empty((T, n))
        x_pred = np.empty((T, n))
        p_filt = np.empty((T, n, n))
        ll_terms = np.empty(T)
        ess_arr = np.empty(T)

        L_Q, logdet_Q, _L_R_sir, _logdet_R_sir = self.model.cholesky()

        for t in range(T):
            # --- 当前时刻的后验经验协方差（用于名义 P_pred）---
            w_now = np.exp(logw - _lse(logw))
            mean_now = np.average(particles, weights=w_now, axis=0)
            emp = particles - mean_now
            P_now = stabilize_cov((emp.T * w_now) @ emp)

            if t > 0:
                # P_pred = A P_{t-1} Aᵀ + Q（逐时刻重算）
                P_pred = stabilize_cov(self.model.A @ P_now @ self.model.A.T + self.model.Q)
                means = particles @ self.model.A.T  # (N, n) 各祖先预测均值
                obs_mask = ~np.isnan(y[t])
                if not obs_mask.any():
                    # 该时刻无观测：退化为纯预测，权重不变
                    particles = means + rng.randn(N, n) @ np.linalg.cholesky(P_pred).T
                    x_pred[t] = mean_now
                    x_filt[t] = mean_now
                    p_filt[t] = P_now
                    ll_terms[t] = 0.0
                    ess_arr[t] = float(N)
                    continue

                Hm = self.model.jacobian_h(mean_now)
                H_o = Hm[obs_mask]
                R_o = self.model.R[np.ix_(obs_mask, obs_mask)]
                S_pred = stabilize_cov(H_o @ P_pred @ H_o.T + R_o)
                L_S = np.linalg.cholesky(S_pred)
                logdet_S = 2.0 * float(np.sum(np.log(np.diag(L_S))))
                # α_i ∝ w_{t-1,i} · N(y_t; h(m_i), S_pred)
                resid_aux = _safe_resid(y[t], self.model.emission(means))
                obs_idx = np.flatnonzero(obs_mask)
                log_aux = _batch_loglik(resid_aux, L_S, logdet_S, cols=obs_idx)
                log_alpha_raw = logw + log_aux
                ll_terms[t] = float(_lse(log_alpha_raw))
                log_alpha = log_alpha_raw - float(_lse(log_alpha_raw))

                if self.resample_aux:
                    anc = self._resampler(np.exp(log_alpha), N, rng)
                    base_w = np.zeros(N)
                else:
                    anc = np.arange(N)
                    base_w = logw - log_aux
                base_mean = means[anc]
                w_base = np.exp(base_w - _lse(base_w)) if np.any(base_w) else np.full(N, 1.0 / N)
                x_pred[t] = np.average(base_mean, weights=w_base, axis=0)

                # --- 阶段 2：逐祖先最优提议 ---
                # q_mean_i = m_i + K'(y_t - h(m_i))，逐祖先向量化计算
                K_mat = _kalman_gain(P_pred, H_o, L_S)
                innov_vec = y[t][obs_idx] - _emission_rows(self.model, base_mean, obs_idx)
                q_mean = base_mean + innov_vec @ K_mat.T
                q_cov = stabilize_cov(P_pred - K_mat @ H_o @ P_pred)
                L_q = np.linalg.cholesky(q_cov)
                particles = q_mean + rng.randn(N, n) @ L_q.T

                # --- 阶段 3：严格重要性权重 ---
                # log p(y_t|x_t)：残差按观测维度
                resid = _safe_resid(y[t], self.model.emission(particles))
                R_chol_o = np.linalg.cholesky(R_o)
                logdet_Ro = 2.0 * float(np.sum(np.log(np.diag(R_chol_o))))
                loglik = _batch_loglik(resid, R_chol_o, logdet_Ro, cols=obs_idx)
                # log p(x_t|x_{t-1}^a)：以被选祖先的预测均值为条件
                z_pr = np.linalg.solve(L_Q, (particles - base_mean).T)
                log_prior_cond = -0.5 * (
                    n * np.log(2 * np.pi) + logdet_Q + np.sum(z_pr * z_pr, axis=0)
                )
                # log q(x_t|x_{t-1}^a, y_t)
                z_q = np.linalg.solve(L_q, (particles - q_mean).T)
                k_obs = int(obs_mask.sum())
                log_q_cond = -0.5 * (
                    k_obs * np.log(2 * np.pi)
                    + 2.0 * float(np.sum(np.log(np.diag(L_q))))
                    + np.sum(z_q * z_q, axis=0)
                )
                log_w = base_w + loglik + log_prior_cond - log_q_cond
                logw = log_w - _lse(log_w)
            else:
                x_pred[t] = particles.mean(axis=0)
                resid = _safe_resid(y[t], self.model.emission(particles))
                obs_idx0 = np.flatnonzero(~np.isnan(y[t]))
                if len(obs_idx0) == 0:
                    ll_terms[t] = 0.0
                    particles, logw = self._finalize_step(
                        t, particles, logw, x_filt, p_filt, x_pred, ll_terms, ess_arr
                    )
                    continue
                R0 = self.model.R[np.ix_(obs_idx0, obs_idx0)]
                L0 = np.linalg.cholesky(R0)
                loglik = _batch_loglik(
                    resid, L0, 2.0 * float(np.sum(np.log(np.diag(L0)))), cols=obs_idx0
                )
                ll_terms[t] = float(_lse(logw + loglik))
                logw = logw + loglik
                logw -= _lse(logw)

            particles, logw = self._finalize_step(
                t, particles, logw, x_filt, p_filt, x_pred, ll_terms, ess_arr
            )

        return FilterEstimate(x_pred, x_filt, p_filt, ll_terms, ess_arr)


def _emission_rows(model, x: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """计算 ``(N, m)`` 粒子的发射值在指定 ``rows`` 行上的取值。

    model.emission 对 2D 输入已返回 (N, m)，此处只做列选择；
    单独成函数是为了让"按观测维度取行"这件事在 APF 里只有一处实现。
    """
    return model.emission(x)[:, rows]


def _kalman_gain(P_pred: np.ndarray, H: np.ndarray, L_S: np.ndarray) -> np.ndarray:
    """``K = P Hᵀ S⁻¹``，其中 ``S = L_S L_Sᵀ`` 已分解，避免显式求逆。"""
    PHt = P_pred @ H.T
    return np.linalg.solve(L_S.T, np.linalg.solve(L_S, PHt.T)).T
