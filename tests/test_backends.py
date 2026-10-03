"""Tier-0 后端与离线降级路径测试。

这些测试的价值在于**验证降级链真的能走通**，而不是"有就测、没有就跳过"：
如果后端缺失时pipeline崩了，说明离线兜底契约是假的。
"""

from __future__ import annotations

import numpy as np
import pytest

from core.config import SeqForgeConfig
from core.errors import BackendUnavailableError
from core.types import FilterReport, LinearGaussianSpec
from data.dgp import build_model, make_dataset
from eval.metrics import aggregate
from filters.backends import (
    BACKEND_PROBE,
    available_pykalman,
    available_statsmodels,
    build_pykalman_filter,
    build_statsmodels_filter,
)
from pipeline.runner import SeqForgePipeline, build_methods


@pytest.fixture(scope="module")
def lg():
    spec = LinearGaussianSpec()
    ds = make_dataset("linear_gaussian", n_steps=60, seed=7, spec=spec)
    return ds, build_model("linear_gaussian", spec, 7), spec


def test_probe_contract(lg):
    """每个 ``available_*`` 必须返回 ``(bool, str)``，不抛异常。"""
    for name, probe in BACKEND_PROBE.items():
        result = probe()
        assert isinstance(result, tuple) and len(result) == 2, f"{name} 探测返回格式错误"
        ok, info = result
        assert isinstance(ok, bool)
        assert isinstance(info, str) and info, f"{name} 未返回原因/版本"


def test_statsmodels_backend_runs(lg):
    """statsmodels 后端能跑通（可用时）。

    **口径说明（实测踩坑）**：statsmodels 0.15 移除了 ``initial_state`` 属性，
    只剩 diffuse/approximate_diffuse/stationary 三种初始化，
    无法直接指定 ``x0=0, P0=I``；实测三种初值下 llf 与自研 KF 都差一个
    初值项（−141~−59）。因此**本测试只断言「跑得通 + 数值有限」**，
    不断言与自研 KF 逐位一致——逐位交叉验证由 ``test_I1``（闭式解）承担。
    这是诚实的设计：外部库的初值约定差异是它的实现细节，不是我们的 bug。
    """
    ds, mdl, _ = lg
    ok, _ = available_statsmodels()
    if not ok:
        pytest.skip("statsmodels 不可用（这是合法的降级路径）")
    est = build_statsmodels_filter(mdl).filter(ds.y)
    assert np.isfinite(est.log_lik_terms).all()
    assert est.x_filt.shape == ds.x_true.shape
    assert est.p_filt.shape == (ds.n_steps, mdl.n_state, mdl.n_state)


def test_pykalman_backend_runs(lg):
    """pykalman 后端能跑通（可用时）。"""
    ds, mdl, _ = lg
    ok, _ = available_pykalman()
    if not ok:
        pytest.skip("pykalman 不可用（这是合法的降级路径）")
    est = build_pykalman_filter(mdl).filter(ds.y)
    assert np.isfinite(est.x_filt).all()


def test_backend_unavailable_raises(lg):
    """后端不可用时构造函数抛 BackendUnavailableError（E400）。"""
    import filters.backends as bmod

    _, mdl, _ = lg

    class _Fake:
        def __init__(self, *a, **k):
            raise BackendUnavailableError("forced", detail="test")

    orig = bmod.StatsmodelsFilter
    try:
        bmod.StatsmodelsFilter = _Fake  # type: ignore[misc]
        with pytest.raises(BackendUnavailableError) as exc:
            bmod.StatsmodelsFilter(mdl)
        assert exc.value.code == "E400"
    finally:
        bmod.StatsmodelsFilter = orig  # type: ignore[misc]


def test_offline_fallback_works_without_backends(monkeypatch):
    """**核心契约**：即使所有 Tier-0 后端都不可用，benchmark 仍能跑完。

    这直接验证「离线兜底」不是口号——把探测函数全部打成 False，
    流水线必须照常产出结果（可用方法变少，但**不崩、不伪造数字**）。
    """
    import filters.backends as bmod

    monkeypatch.setitem(bmod.BACKEND_PROBE, "statsmodels", lambda: (False, "forced off"))
    monkeypatch.setitem(bmod.BACKEND_PROBE, "pykalman", lambda: (False, "forced off"))
    monkeypatch.setattr(bmod, "available_statsmodels", lambda: (False, "forced off"))
    monkeypatch.setattr(bmod, "available_pykalman", lambda: (False, "forced off"))

    rep = SeqForgePipeline(SeqForgeConfig(n_steps=40, n_particles=200)).run(
        datasets=["linear_gaussian"], seeds=(7,)
    )
    assert rep["results"], "降级后仍须产出结果"
    assert any(not r["skipped"] for r in rep["results"]), "至少内置方法可用"
    # 关键：不可用后端**不得**出现在结果里（不能伪造）
    names = {r["method"] for r in rep["results"]}
    assert "statsmodels_kf" not in names
    assert "pykalman_kf" not in names


def test_skipped_method_is_aggregated_as_skipped():
    """skipped 的方法在聚合里保留 skip_reason，不被静默丢弃。"""
    rows = [
        FilterReport(
            method="broken",
            family="kalman",
            backend="x",
            dataset="linear_gaussian",
            seed=7,
            log_lik=float("nan"),
            rmse_state=float("nan"),
            n_steps=10,
            runtime_sec=0.0,
            skipped=True,
            skip_reason="E300 NumericalError: boom",
        ),
        FilterReport(
            method="ok",
            family="kalman",
            backend="y",
            dataset="linear_gaussian",
            seed=7,
            log_lik=-10.0,
            rmse_state=0.5,
            n_steps=10,
            runtime_sec=0.1,
        ),
    ]
    out = {r.method: r for r in aggregate(rows)}
    assert out["broken"].skipped
    assert "E300" in out["broken"].skip_reason
    assert not out["ok"].skipped


def test_linear_domain_excludes_nonlinear_methods(lg):
    """线性域**故意不放** EKF/UKF/FFBS（它们必须精确等于 KF，重复无意义）。"""
    _, mdl, _ = lg
    cfg = SeqForgeConfig()
    methods = build_methods(mdl, "linear_gaussian", cfg)
    assert set(methods) == {"kf", "sir", "apf"}
    assert "ekf" not in methods and "ukf" not in methods


def test_nonlinear_domain_includes_all_methods(lg):
    """非线性域放出全部方法。"""
    ds, _, _ = lg
    from core.types import NonlinearSpec

    spec = NonlinearSpec()
    mdl = build_model("nonlinear_obs", spec, 7)
    methods = build_methods(mdl, "nonlinear_obs", SeqForgeConfig())
    assert {"kf", "sir", "apf", "ekf", "ukf", "iekf", "ffbs"} == set(methods)
    _ = ds
