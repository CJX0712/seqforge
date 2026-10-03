"""CLI 冒烟 + 配置契约 + 确定性端到端测试。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cli import build_parser, main
from core.config import SeqForgeConfig, load_config
from core.errors import ConfigError
from core.seed import set_all, spawn_rng
from data.dgp import DEFAULT_SPECS, make_dataset
from eval.metrics import format_table, mean_std, significance
from pipeline.runner import SeqForgePipeline

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


def test_config_defaults_valid():
    cfg = SeqForgeConfig()
    assert cfg.n_particles >= 50
    assert len(cfg.bench_seeds) >= 3, "统计严谨性要求 ≥3 seeds"


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("ENV_SEQFORGE_N_PARTICLES", "777")
    monkeypatch.setenv("ENV_SEQFORGE_N_STEPS", "64")
    cfg = load_config()
    assert cfg.n_particles == 777
    assert cfg.n_steps == 64


def test_config_rejects_bad_values(monkeypatch):
    monkeypatch.setenv("ENV_SEQFORGE_N_PARTICLES", "10")
    with pytest.raises(ConfigError) as exc:
        load_config()
    assert exc.value.code == "E100"


def test_config_rejects_non_integer(monkeypatch):
    monkeypatch.setenv("ENV_SEQFORGE_N_STEPS", "abc")
    with pytest.raises(ConfigError):
        load_config()


def test_config_unknown_key_raises():
    with pytest.raises(ConfigError):
        SeqForgeConfig().with_overrides(no_such_key=1)


# ---------------------------------------------------------------------------
# seed 确定性
# ---------------------------------------------------------------------------


def test_spawn_rng_is_stable_and_isolated():
    set_all(42)
    a = spawn_rng("dgp.linear_gaussian").random_sample(5)
    b = spawn_rng("dgp.nonlinear_obs").random_sample(5)
    c = spawn_rng("dgp.linear_gaussian").random_sample(5)
    assert np.array_equal(a, c), "同名子流必须可复现"
    assert not np.array_equal(a, b), "不同名子流必须独立"


def test_dataset_determinism():
    spec = DEFAULT_SPECS["linear_gaussian"]
    d1 = make_dataset("linear_gaussian", n_steps=30, seed=7, spec=spec)
    d2 = make_dataset("linear_gaussian", n_steps=30, seed=7, spec=spec)
    assert np.array_equal(d1.y, d2.y)
    assert np.array_equal(d1.x_true, d2.x_true)
    d3 = make_dataset("linear_gaussian", n_steps=30, seed=8, spec=spec)
    assert not np.array_equal(d1.y, d3.y)


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------


def test_mean_std():
    m, s = mean_std([1.0, 2.0, 3.0])
    assert m == pytest.approx(2.0)
    assert s == pytest.approx(1.0)
    assert mean_std([]) != (0.0, 0.0)
    assert mean_std([5.0]) == (5.0, 0.0)


def test_significance_rule():
    # Δ=1.0 > 0.5*(0.2+0.2)=0.2 → 显著
    assert significance(1.0, 0.2, 0.2)
    # Δ=0.1 < 0.2 → 不显著
    assert not significance(0.1, 0.2, 0.2)


def test_format_table_handles_skipped():
    class _R:
        def __init__(self, name, skipped):
            self.method = name
            self.family = "kalman"
            self.skipped = skipped
            self.log_lik_mean = float("nan")
            self.log_lik_std = float("nan")
            self.rmse_mean = float("nan")
            self.per_dataset = {}
            self.skip_reason = ""

    out = format_table([_R("ok", False), _R("bad", True)])
    assert "ok" in out and "bad" in out and "skipped" in out


# ---------------------------------------------------------------------------
# 端到端确定性（DoD 关键项）
# ---------------------------------------------------------------------------


def test_pipeline_bitwise_determinism(tmp_path):
    """同 seed 两次跑，除 elapsed 外所有数值**逐位一致**。

    注意：``NaN != NaN``，直接用 ``==`` 比较 dict 会因ESS 字段的
    ``nan``（解析法无粒子概念）而误报失败。正确做法是逐字段比较，
    对浮点用 ``np.array_equal(..., equal_nan=True)``。
    """
    cfg = SeqForgeConfig(n_steps=40, n_particles=300)
    rep1 = SeqForgePipeline(cfg).run(datasets=["linear_gaussian", "nonlinear_obs"], seeds=(7, 101))
    rep2 = SeqForgePipeline(cfg).run(datasets=["linear_gaussian", "nonlinear_obs"], seeds=(7, 101))

    assert len(rep1["raw"]) == len(rep2["raw"])
    for a, b in zip(rep1["raw"], rep2["raw"], strict=True):
        for key in a:
            if key == "runtime_sec":
                continue
            va, vb = a[key], b[key]
            if isinstance(va, float):
                assert np.array_equal(va, vb, equal_nan=True), f"{key} 不一致：{va} vs {vb}"
            else:
                assert va == vb, f"{key} 不一致：{va} vs {vb}"


def test_benchmark_numbers_are_finite_and_real():
    """benchmark.json 的每个数字必须来自真实运行（无 NaN/inf 混入）。"""
    cfg = SeqForgeConfig(n_steps=40, n_particles=200)
    rep = SeqForgePipeline(cfg).run(datasets=["linear_gaussian"], seeds=(7,))
    for row in rep["raw"]:
        if not row["skipped"]:
            assert np.isfinite(row["log_lik"]), f"{row['method']} log_lik 非有限"
            assert np.isfinite(row["rmse_state"]), f"{row['method']} rmse 非有限"
    assert rep["kf_reference"], "必须提供 KF 精确 log-lik 作为金标准"


def test_failure_cases_present():
    """失败案例必须从实测结果派生，且非空。"""
    cfg = SeqForgeConfig(n_steps=40, n_particles=200)
    rep = SeqForgePipeline(cfg).run(seeds=(7,))
    cases = rep["failure_cases"]
    assert cases, "必须产出失败案例"
    for c in cases:
        assert c["symptom"] and c["root_cause"] and c["evidence"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_parser_subcommands():
    p = build_parser()
    assert p.prog == "seqforge"
    for cmd in ("demo", "list", "selftest", "version"):
        args = p.parse_args([cmd] if cmd != "demo" else ["demo", "--steps", "30"])
        assert args.func is not None


def test_cli_version(capsys):
    assert main(["version"]) == 0
    assert "SeqForge" in capsys.readouterr().out


def test_cli_list_runs():
    assert main(["list"]) == 0


def test_cli_demo_smoke(tmp_path, capsys):
    out = tmp_path / "bench"
    rc = main(
        [
            "demo",
            "--steps",
            "30",
            "--particles",
            "150",
            "--seeds",
            "7",
            "--datasets",
            "linear_gaussian",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    payload = json.loads((out / "benchmark.json").read_text(encoding="utf-8"))
    assert payload["system"] == "SeqForge"
    assert payload["author"] == "晨星"
    assert payload["results"]


def test_cli_selftest():
    """selftest 子命令能跑通（CI 快速门禁的入口）。"""
    assert main(["selftest"]) == 0


def test_module_entrypoint_help():
    """``python cli.py --help`` 正常（CI 冒烟）。"""
    r = subprocess.run(
        [sys.executable, str(ROOT / "cli.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(ROOT),
    )
    assert r.returncode == 0
    assert "seqforge" in r.stdout.lower()
