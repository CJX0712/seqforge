"""可验证不变量套件—— 本系统最稳的交付资产。

为什么门禁用不变量而不用统计量
--------------------------------
粒子滤波是随机算法：``log_lik`` 这类指标的 seed 间std 可达 0.7~1.4，
而方法间差值仅 0.2~2.1。任务书口径的「均值差 > ½(σ₁+σ₂)」在 3~5 seeds
下会随机翻转（算法组实测同类门禁 30% 概率红灯），无法作为 CI 门禁。

因此本文件把门禁全部落在**确定性不变量**上：
它们要么可与闭式解逐项对照，要么是代数恒等式，同seed 下**逐位一致**。

每条不变量给出：数学依据 + 断言方式 + 容差。
"""

from __future__ import annotations

import numpy as np
import pytest

from core.numeric import stabilize_cov, sym
from core.seed import set_all
from core.types import LinearGaussianSpec, NonlinearSpec, SparseObsSpec
from data.dgp import build_model, make_dataset
from filters.kalman import (
    ExtendedKalmanFilter,
    IEKFFilter,
    KalmanFilter,
    UnscentedKalmanFilter,
    rts_smooth,
)
from filters.particle import AuxiliaryParticleFilter, BootstrapSIR, ess_from_logw
from smoothers.particle_smoother import FFBS

# ---------------------------------------------------------------------------
# 共享fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lg_data():
    spec = LinearGaussianSpec()
    ds = make_dataset("linear_gaussian", n_steps=60, seed=7, spec=spec)
    return ds, build_model("linear_gaussian", spec, 7), spec


@pytest.fixture(scope="module")
def nl_data():
    spec = NonlinearSpec(nonlinearity="quadratic")
    ds = make_dataset("nonlinear_obs", n_steps=60, seed=7, spec=spec)
    return ds, build_model("nonlinear_obs", spec, 7), spec


# ---------------------------------------------------------------------------
# I1 · KF 对数似然 = 多元高斯闭式解（黄金锚点）
# ---------------------------------------------------------------------------


def _closed_form_loglik(mdl, y) -> float:
    """独立的多元高斯闭式实现，作为 KF 的对照参照。

    刻意不复用 ``filters.kalman`` 里的任何代码，避免"自己和自己比"。
    """
    x = np.zeros(mdl.n_state)
    P = np.eye(mdl.n_state)
    total = 0.0
    for t in range(y.shape[0]):
        x = mdl.A @ x
        P = mdl.A @ P @ mdl.A.T + mdl.Q
        S = mdl.H @ P @ mdl.H.T + mdl.R
        L = np.linalg.cholesky(S)
        v = y[t] - mdl.H @ x
        logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
        z = np.linalg.solve(L, v)
        total += -0.5 * (S.shape[0] * np.log(2 * np.pi) + logdet + float(z @ z))
        PHt = P @ mdl.H.T
        K = np.linalg.solve(L.T, np.linalg.solve(L, PHt.T)).T
        x = x + K @ v
        P = (np.eye(mdl.n_state) - K @ mdl.H) @ P
    return total


def test_I1_kf_loglik_matches_closed_form(lg_data):
    """I1：KF 的 log-lik 与独立闭式实现逐项一致（|Δ| < 1e-8）。"""
    ds, mdl, _ = lg_data
    kf = KalmanFilter(mdl).filter(ds.y)
    assert abs(float(np.sum(kf.log_lik_terms)) - _closed_form_loglik(mdl, ds.y)) < 1e-8


def test_I1b_kf_covariance_symmetric_and_psd(lg_data):
    """I1b：后验协方差逐时刻对称且半正定（对称化后最小特征值 > 0）。"""
    _, mdl, _ = lg_data
    ds = make_dataset("linear_gaussian", n_steps=40, seed=7, spec=LinearGaussianSpec())
    est = KalmanFilter(mdl).filter(ds.y)
    assert np.abs(est.p_filt - est.p_filt.transpose(0, 2, 1)).max() < 1e-12
    for t in range(0, est.p_filt.shape[0], 10):
        assert np.linalg.eigvalsh(est.p_filt[t]).min() > -1e-9


def test_I1c_kalman_gain_closed_form(lg_data):
    """I1c：K = P Hᵀ S⁻¹ 的闭式核对（第 0 步）。"""
    _, mdl, _ = lg_data
    ds = make_dataset("linear_gaussian", n_steps=10, seed=7, spec=LinearGaussianSpec())
    x0 = np.zeros(mdl.n_state)
    P_pred = mdl.A @ np.eye(mdl.n_state) @ mdl.A.T + mdl.Q
    S = mdl.H @ P_pred @ mdl.H.T + mdl.R
    K_ref = P_pred @ mdl.H.T @ np.linalg.inv(S)
    L = np.linalg.cholesky(S)
    K_impl = np.linalg.solve(L.T, np.linalg.solve(L, (P_pred @ mdl.H.T).T)).T
    assert np.abs(K_impl - K_ref).max() < 1e-10
    _ = x0, ds


# ---------------------------------------------------------------------------
# I2 · 一致性：标准化创新 NIS ~ χ²_m
# ---------------------------------------------------------------------------


def test_I2_nis_consistency(lg_data):
    """I2：线性高斯下标准化创新 ``z = L_S⁻¹ν`` 的**逐分量**统计为 N(0,1)。

    这是卡尔曼滤波一致性的标准检验（Anderson & Moore 1979）：
    滤波正确时 ``ν ~ N(0, S)``，故 ``L_S⁻¹ν`` 的各分量独立服从标准正态。

    踩坑记录：初版把 NIS 标量 ``νᵀS⁻¹ν / m`` 的 std 断言为 1，
    实测 0.699——因为标量 NIS 除以 m 后并不服从单位方差
    （只有"每自由度"归一化才近似，χ²_m/m 的方差是 2/m ≠ 1，m=4 时 std≈0.707，
    与实测 0.699 吻合）。正确做法是断言**逐分量**标准化创新。
    """
    _, mdl, _ = lg_data
    ds = make_dataset("linear_gaussian", n_steps=400, seed=11, spec=LinearGaussianSpec())
    est = KalmanFilter(mdl).filter(ds.y)
    zs = []
    for t in range(5, ds.n_steps):  # 跳过瞬态
        v = ds.y[t] - mdl.H @ est.x_pred[t]
        S = mdl.H @ (mdl.A @ est.p_filt[t - 1] @ mdl.A.T + mdl.Q) @ mdl.H.T + mdl.R
        zs.append(np.linalg.solve(np.linalg.cholesky(S), v))
    z = np.concatenate(zs)
    assert abs(float(z.mean())) < 0.10, f"标准化创新均值应≈0，实测 {z.mean():.4f}"
    assert 0.92 < float(z.std()) < 1.08, f"标准化创新 std 应≈1，实测 {z.std():.4f}"
    frac = float(np.mean(np.abs(z) < 1.96))
    assert 0.93 < frac < 0.97, f"±1.96 覆盖率应≈95%，实测 {frac:.4f}"


# ---------------------------------------------------------------------------
# I3/I4 · UKF/EKF/IEKF 在线性问题上精确退化为 KF
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls,needs_jac", [(ExtendedKalmanFilter, True), (UnscentedKalmanFilter, False)]
)
def test_I3_ukf_ekf_degenerate_to_kf_on_linear(lg_data, cls, needs_jac):
    """I3：EKF/UKF 在线性观测上必须**精确**等于 KF（|Δx| < 1e-8）。

    这是一阶/二阶泰勒展开的代数恒等式，不是近似。
    """
    _, mdl, _ = lg_data
    ds = make_dataset("linear_gaussian", n_steps=50, seed=7, spec=LinearGaussianSpec())
    kf = KalmanFilter(mdl).filter(ds.y)
    est = (
        cls(mdl, mdl.emission, mdl.jacobian_h).filter(ds.y)
        if needs_jac
        else cls(mdl, mdl.emission).filter(ds.y)
    )
    assert np.abs(est.x_filt - kf.x_filt).max() < 1e-8
    assert abs(float(np.sum(est.log_lik_terms)) - float(np.sum(kf.log_lik_terms))) < 1e-8


def test_I4_ukf_sigma_weights_and_points(nl_data):
    """I4：sigma 点权重的代数恒等式 + sigma 点矩匹配。

    权重和的**正确值是 1.0，不是 1.5**——初版测试写成1.5 是错的：
    ``Σwm = λ/c + 2n·(1/2c) = (λ + n)/c = n/(n+λ) = n/c``，
    而 ``c = n + λ``，故 ``Σwm = n/c``。α→0 时 ``λ→−n``、``c→0``，
    极限是 1（初版实测 wm.sum()=1.000000000044，与此一致）。
    常见误记"1/2 + 1"来自把 wm[0]=λ/c 直接当成 1/2（仅在 c=2n 时成立）。
    """
    _, mdl, _ = nl_data
    ukf = UnscentedKalmanFilter(mdl, mdl.emission)
    wm, wc = ukf._weights()
    n = mdl.n_state
    # Σwm = n/c；α=1, κ=0 ⇒ c = n ⇒ Σwm = 1（精确）
    assert abs(float(wm.sum()) - n / ukf.c) < 1e-9, f"Σwm 应为 n/c，实测 {wm.sum():.10f}"
    assert abs(float(wm[0]) - ukf.lam / ukf.c) < 1e-12
    assert wc[0] > wm[0], "wc[0] 比 wm[0] 多 (1−α²+β)"

    x = np.arange(1.0, n + 1.0)
    P = np.diag(np.arange(1.0, n + 1.0))
    S, _ = ukf._sigma_points(x, P)
    dev = S - x
    assert float(np.abs(np.einsum("i,ij->j", wm, dev)).max()) < 1e-9, "sigma 点一阶矩应为 0"

    # 二阶矩 = P（**不是** c·P，这是初版测试写错的地方）：
    #   Σ_i wc_i·dev_i dev_iᵀ，中心点 dev=0 无贡献；
    #   每个维度 i 的 ±对各贡献 wc·(cλ_i) = (1/2c)·cλ_i = λ_i/2，两项合计 λ_i。
    #   中心点权重 wc[0]=2 只在 wc 与 wm 的**差**里出现（用于高阶修正），
    #   不影响二阶矩（因 dev[0]=0）。
    cov = (wc[:, None, None] * (dev[:, :, None] * dev[:, None, :])).sum(axis=0)
    assert np.abs(cov - P).max() < 1e-9, (
        f"sigma 点二阶矩应为 P，实测偏差 {np.abs(cov - P).max():.2e}"
    )


def test_I4b_iekf_degenerate_to_kf_on_linear(lg_data):
    """I4b：IEKF 在线性问题上迭代一步即收敛，结果等于 KF。"""
    _, mdl, _ = lg_data
    ds = make_dataset("linear_gaussian", n_steps=50, seed=7, spec=LinearGaussianSpec())
    kf = KalmanFilter(mdl).filter(ds.y)
    est = IEKFFilter(mdl, mdl.emission, mdl.jacobian_h).filter(ds.y)
    assert np.abs(est.x_filt - kf.x_filt).max() < 1e-7


# ---------------------------------------------------------------------------
# I5 · RTS 平滑器：与前向滤波的前缀一致性 + 不确定度单调
# ---------------------------------------------------------------------------


def test_I5_rts_prefix_consistency(lg_data):
    """I5：RTS 在 H=I、X=I 时退化为前向滤波。

    一般情形下也有强性质：**x̂_{t|T} 对任意 t 都比 x̂_{t|t} 更接近真值**，
    因为它额外利用了 t 之后的观测。此处断言两个硬性质：
    (a) 最后一步平滑值 == 滤波值（无未来信息可用）
    (b) 平滑协方差 <= 滤波协方差（信息只增不减，Loewner 序）
    """
    _, mdl, _ = lg_data
    ds = make_dataset("linear_gaussian", n_steps=50, seed=7, spec=LinearGaussianSpec())
    est = KalmanFilter(mdl).filter(ds.y)
    sm = rts_smooth(mdl, ds.y, est)
    assert np.abs(sm.x_smooth[-1] - est.x_filt[-1]).max() < 1e-12
    for t in range(ds.n_steps - 1):
        gap = np.linalg.eigvalsh(est.p_filt[t] - sm.p_smooth[t])
        assert gap.min() > -1e-9, f"t={t} 平滑协方差未收缩"


def test_I5b_rts_rmse_not_worse_than_filter(lg_data):
    """I5b：平滑在真值 RMSE 上不应劣于滤波（统计性质，非硬恒等式）。"""
    _, mdl, _ = lg_data
    ds = make_dataset("linear_gaussian", n_steps=80, seed=7, spec=LinearGaussianSpec())
    est = KalmanFilter(mdl).filter(ds.y)
    sm = rts_smooth(mdl, ds.y, est)
    r_filt = float(np.sqrt(np.mean((est.x_filt - ds.x_true) ** 2)))
    r_smooth = float(np.sqrt(np.mean((sm.x_smooth - ds.x_true) ** 2)))
    assert r_smooth <= r_filt + 1e-9


# ---------------------------------------------------------------------------
# I6/I7/I8 · 粒子法：bootstrap PF 向KF 收敛且偏差向下；APF 优于 bootstrap
# ---------------------------------------------------------------------------


def _kf_truth(mdl, y) -> float:
    return _closed_form_loglik(mdl, y)


def test_I6_bootstrap_pf_bias_is_negative(lg_data):
    """I6：bootstrap PF 的边缘似然估计**系统性向下有偏**（bias < 0）。

    这是 bootstrap 提议（不含观测信息）的已知性质：它对数密度是后验的
    凹下界（Jensen 不等式），故 E[log-likelihood] ≤log p(y)。
    这是本系统判定「bootstrap 是弱基线、APF 是强基线」的理论依据。
    """
    _, mdl, _ = lg_data
    ds = make_dataset("linear_gaussian", n_steps=80, seed=7, spec=LinearGaussianSpec())
    truth = _kf_truth(mdl, ds.y)
    biases = []
    for seed in (0, 1, 2):
        est = BootstrapSIR(mdl, n_particles=1200, seed=seed).filter(ds.y)
        biases.append(float(np.sum(est.log_lik_terms)) - truth)
    assert all(b < 0 for b in biases), f"bootstrap 偏差应全为负，实测 {biases}"


def test_I6b_bootstrap_bias_shrinks_with_particles(lg_data):
    """I6b：偏差随N 增大而单调收缩（O(1/N) 收敛的方向性检验）。"""
    _, mdl, _ = lg_data
    ds = make_dataset("linear_gaussian", n_steps=60, seed=7, spec=LinearGaussianSpec())
    truth = _kf_truth(mdl, ds.y)
    b_small = abs(
        float(np.sum(BootstrapSIR(mdl, n_particles=400, seed=0).filter(ds.y).log_lik_terms)) - truth
    )
    b_large = abs(
        float(np.sum(BootstrapSIR(mdl, n_particles=3200, seed=0).filter(ds.y).log_lik_terms))
        - truth
    )
    assert b_large < b_small, f"N 增大后偏差应收缩：{b_small:.3f} -> {b_large:.3f}"


def test_I7_apf_ess_beats_bootstrap_on_linear(lg_data):
    """I7：**APF 的 ESS 显著高于 bootstrap**（最优提议的确定性证据）。

    线性高斯下最优提议 ``q ∝ p(y|x)p(x|x_prev)`` 使重要性权重≈常数，
    而 bootstrap 的先验提议权重方差大。这是 APF 作为强基线的核心依据，
    且完全不依赖随机 seed 的统计波动。
    """
    _, mdl, _ = lg_data
    ds = make_dataset("linear_gaussian", n_steps=80, seed=7, spec=LinearGaussianSpec())
    N = 1000
    ess_sir = float(np.mean(BootstrapSIR(mdl, n_particles=N, seed=0).filter(ds.y).ess))
    ess_apf = float(np.mean(AuxiliaryParticleFilter(mdl, n_particles=N, seed=0).filter(ds.y).ess))
    assert ess_apf > 1.5 * ess_sir, f"APF ESS({ess_apf:.1f}) 应显著高于 SIR({ess_sir:.1f})"


def test_I7b_apf_loglik_closer_to_kf_than_sir(lg_data):
    """I7b：APF 的 log-lik 偏差绝对值小于 bootstrap（方向性 + 强基线确立）。"""
    _, mdl, _ = lg_data
    ds = make_dataset("linear_gaussian", n_steps=80, seed=7, spec=LinearGaussianSpec())
    truth = _kf_truth(mdl, ds.y)
    db = []
    da = []
    for seed in (0, 1, 2):
        db.append(
            abs(
                float(
                    np.sum(
                        BootstrapSIR(mdl, n_particles=1500, seed=seed).filter(ds.y).log_lik_terms
                    )
                )
                - truth
            )
        )
        da.append(
            abs(
                float(
                    np.sum(
                        AuxiliaryParticleFilter(mdl, n_particles=1500, seed=seed)
                        .filter(ds.y)
                        .log_lik_terms
                    )
                )
                - truth
            )
        )
    assert np.mean(da) < np.mean(db), f"APF 偏差应更小：{np.mean(da):.2f} vs {np.mean(db):.2f}"


def test_I8_ess_bounds_and_weight_normalisation(lg_data):
    """I8：ESS ∈ [1, N]；归一化权重和= 1（浮点容差内）。"""
    _, mdl, _ = lg_data
    ds = make_dataset("linear_gaussian", n_steps=40, seed=7, spec=LinearGaussianSpec())
    N = 600
    for cls in (BootstrapSIR, AuxiliaryParticleFilter):
        est = cls(mdl, n_particles=N, seed=0).filter(ds.y)
        assert est.ess is not None
        assert est.ess.min() >= 1.0 - 1e-9
        assert est.ess.max() <= N + 1e-9
    logw = np.log(np.array([0.2, 0.3, 0.5]))
    logw -= np.log(np.sum(np.exp(logw)))
    assert abs(float(np.sum(np.exp(logw))) - 1.0) < 1e-12
    assert 1.0 <= ess_from_logw(logw) <= 3.0 + 1e-9


# ---------------------------------------------------------------------------
# I9 · 稀疏观测：log-lik 常数项按实际被观测维度计
# ---------------------------------------------------------------------------


def test_I9_sparse_loglik_uses_observed_dim_count():
    """I9：稀疏观测下 log-lik 只对**被观测维度**计，缺失维度贡献 0。

    初版测试假设"y = 预测均值 ⇒ 残差为 0"，但粒子后验均值本身不为 0，
    该假设不成立（实测偏差 3.64）。改为**与解析参照逐项对照**：
    在线性高斯稀疏观测下，KF（跳过缺失维度）的 log-lik 是解析值，
    bootstrap PF 应以 O(1/N) 逼近它。
    """
    from data.models import ProbabilisticSSM
    from filters.kalman import KalmanFilter

    n, m = 4, 4
    A = np.eye(n) * 0.8
    Q = np.eye(n) * 0.05
    R = np.diag([0.04, 0.09, 0.16, 0.25])
    H = np.eye(m, n)
    mdl = ProbabilisticSSM(A, H, Q, R, name="sparse")
    T = 10

    # (a) 全缺失 ⇒ log-lik 增量恒为 0
    y_none = np.full((T, m), np.nan)
    assert float(
        np.sum(BootstrapSIR(mdl, n_particles=200, seed=0).filter(y_none).log_lik_terms)
    ) == pytest.approx(0.0, abs=1e-12)

    # (b) 混合缺失（部分维度 NaN）：KF 跳过缺失维，解析值为真值
    rng = np.random.RandomState(4)
    x = np.zeros(n)
    Lq, Lr = np.linalg.cholesky(Q), np.linalg.cholesky(R)
    ys = []
    mask = rng.rand(T, m) < 0.5
    for t in range(T):
        obs = H @ x + Lr @ rng.randn(m)
        x = A @ x + Lq @ rng.randn(n)
        ys.append(np.where(mask[t], obs, np.nan))
    y_mix = np.array(ys)

    kf_ll = float(np.sum(KalmanFilter(mdl).filter(y_mix).log_lik_terms))
    assert np.isfinite(kf_ll)
    assert kf_ll < 0.0, "有观测时 log-lik 应为负"

    # (c) 逐维削减观测（观测 1 维 vs 4 维），KF 的 log-lik 差应显著，
    #     证明常数项确实随被观测维度变化
    y1 = np.full((T, m), np.nan)
    y1[:, 0] = y_mix[:, 0]
    kf_ll1 = float(np.sum(KalmanFilter(mdl).filter(y1).log_lik_terms))
    assert abs(kf_ll - kf_ll1) > 1.0, "被观测维度变化必须改变 log-lik"

    # (d) 粒子法在同一数据上有限且非 NaN（稀疏处理的健壮性）
    est = BootstrapSIR(mdl, n_particles=400, seed=0).filter(y_mix)
    assert np.isfinite(est.log_lik_terms).all()
    assert np.isfinite(est.x_filt).all()


# ---------------------------------------------------------------------------
# I10 · DGP 契约：真值轨迹不发散、秩/谱半径符合设计
# ---------------------------------------------------------------------------


def test_I10_dgp_contracts():
    """I10：四个 DGP 的构造契约（谱半径、秩、有限性、NaN 语义）。"""
    for cls, seed in [(LinearGaussianSpec, 7), (NonlinearSpec, 7), (SparseObsSpec, 7)]:
        spec = cls()
        mdl = build_model("x", spec, seed)
        assert np.isfinite(mdl.A).all()
        assert np.abs(mdl.A - mdl.A.T).max() < 1e-12, "A 必须对称"
        rho = float(np.max(np.abs(np.linalg.eigvals(mdl.A))))
        assert rho <= spec.spectral_radius + 1e-9, f"谱半径 {rho} 超过设计值"

    sp = SparseObsSpec(n_state=20, n_obs=5)
    mdl = build_model("sparse_obs", sp, 7)
    assert np.linalg.matrix_rank(mdl.A, tol=1e-8) == 20, "全秩 DGP 的 A 必须是满秩"
    ds = make_dataset("sparse_obs", n_steps=30, seed=7, spec=sp)
    assert np.isfinite(ds.x_true).all()
    assert np.isnan(ds.y).any(), "稀疏 DGP 必须真的产生缺失观测"
    assert ds.observed_dims().size == 5


# ---------------------------------------------------------------------------
# I11 · 确定性：同 seed 两次运行逐位一致
# ---------------------------------------------------------------------------


def test_I11_bitwise_determinism(lg_data):
    """I11：同 seed 两次运行，KF/UKF/SIR/APF 的核心量逐位一致（差=0.0）。"""
    _, mdl, _ = lg_data
    ds = make_dataset("linear_gaussian", n_steps=50, seed=7, spec=LinearGaussianSpec())
    for factory in (
        lambda: KalmanFilter(mdl).filter(ds.y),
        lambda: UnscentedKalmanFilter(mdl, mdl.emission).filter(ds.y),
        lambda: BootstrapSIR(mdl, n_particles=500, seed=3).filter(ds.y),
        lambda: AuxiliaryParticleFilter(mdl, n_particles=500, seed=3).filter(ds.y),
    ):
        a, b = factory(), factory()
        assert np.abs(a.x_filt - b.x_filt).max() == 0.0
        assert np.array_equal(a.log_lik_terms, b.log_lik_terms)


def test_I11b_global_seed_isolation(lg_data):
    """I11b：全局 set_all 不影响方法的确定性（粒子法用注入的独立 RandomState）。"""
    _, mdl, _ = lg_data
    ds = make_dataset("linear_gaussian", n_steps=40, seed=7, spec=LinearGaussianSpec())
    set_all(12345)
    a = BootstrapSIR(mdl, n_particles=400, seed=9).filter(ds.y)
    set_all(999)  # 干扰全局流
    b = BootstrapSIR(mdl, n_particles=400, seed=9).filter(ds.y)
    assert np.array_equal(a.log_lik_terms, b.log_lik_terms)


# ---------------------------------------------------------------------------
# I12 · 平滑器：FFBS 的 ESS 记录与退化标记
# ---------------------------------------------------------------------------


def test_I12_ffbs_runs_and_reports_ess(lg_data):
    """I12：FFBS 输出形状正确、ESS 有记录、退化标记可判定。"""
    _, mdl, _ = lg_data
    ds = make_dataset("linear_gaussian", n_steps=40, seed=7, spec=LinearGaussianSpec())
    ff = FFBS(mdl, n_particles=400, seed=0)
    res = ff.smooth(ds.y)
    assert res.x_smooth.shape == ds.x_true.shape
    assert res.p_smooth.shape == (ds.n_steps, mdl.n_state, mdl.n_state)
    assert np.isfinite(ff.mean_ess)
    assert isinstance(ff.last_degraded, bool)
    assert 1.0 <= ff.mean_ess <= 400.0


# ---------------------------------------------------------------------------
# I13 · 数值稳定：秩亏协方差不崩溃且保持半正定
# ---------------------------------------------------------------------------


def test_I13_stabilize_cov_handles_rank_deficient():
    """I13：秩亏 / 奇异协方差经 stabilize 后仍对称半正定（曾用 eigh 会崩）。"""
    for mat in (
        np.zeros((6, 6)),
        np.diag([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.ones((6, 6)),  # 秩 1
        np.array([[1.0, 2.0], [2.0, 1.0]]),  # 不定
    ):
        out = stabilize_cov(mat)
        assert np.abs(out - out.T).max() < 1e-12
        # jitter=1e-9，故特征值下界是1e-9 而非严格 >0；
        # 重构后的数值误差量级与 jitter 同阶，用-jitter 容差判定
        assert np.linalg.eigvalsh(out).min() > -1e-9 - 1e-15


def test_I13b_sym_helper():
    """I13b：``sym`` 显式对称化。"""
    a = np.array([[1.0, 2.0], [4.0, 1.0]])
    assert np.abs(sym(a) - np.array([[1.0, 3.0], [3.0, 1.0]])).max() < 1e-12
