"""SeqForge 端到端演示。

用法::

    python examples/run_demo.py                # 标准规模
    python examples/run_demo.py --quick        # 快速冒烟
    python examples/run_demo.py --ablation     # 附带消融对照

落盘 ``artifacts/benchmark.json``，并打印对照表。
所有数字来自真实运行，**不手填**。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import SeqForgeConfig
from core.seed import set_all
from data.dgp import DEFAULT_SPECS, build_model, make_dataset
from filters.kalman import KalmanFilter
from filters.particle import AuxiliaryParticleFilter, BootstrapSIR
from pipeline.runner import SeqForgePipeline


def _bar(title: str) -> None:
    print(f"\n{'=' * 74}\n {title}\n{'=' * 74}")


def show_kf_reference(cfg: SeqForgeConfig, seeds: tuple[int, ...]) -> dict[str, float]:
    """打印线性高斯域的 KF 精确 log-lik——粒子法偏差的唯一金标准。"""
    _bar("线性高斯域：KF 精确 log-lik（金标准）")
    spec = DEFAULT_SPECS["linear_gaussian"]
    out: dict[str, float] = {}
    for seed in seeds:
        ds = make_dataset("linear_gaussian", n_steps=cfg.n_steps, seed=seed, spec=spec)
        mdl = build_model("linear_gaussian", spec, seed)
        est = KalmanFilter(mdl).filter(ds.y)
        out[str(seed)] = float(np.sum(est.log_lik_terms))
        rmse = ds.rmse(est.x_filt)
        print(
            f"  seed {seed:>4}: log-lik = {out[str(seed)]:12.6f}   RMSE(obs) = {rmse['rmse_obs']:.4f}"
        )
    print("\n  这条线的意义：KF 在线性高斯下是**精确贝叶斯最优**，")
    print("  任何粒子法都只能以 O(1/N) 逼近它—— 因此它定义了「不可能超越的天花板」。")
    return out


def show_ess_comparison(cfg: SeqForgeConfig) -> None:
    """APF vs bootstrap 的 ESS 对照（本系统最核心的确定性证据）。"""
    _bar("核心对照：最优提议(APF) vs 先验提议(bootstrap) 的 ESS")
    spec = DEFAULT_SPECS["linear_gaussian"]
    ds = make_dataset("linear_gaussian", n_steps=cfg.n_steps, seed=7, spec=spec)
    mdl = build_model("linear_gaussian", spec, 7)
    n = cfg.n_particles
    sir = BootstrapSIR(mdl, n_particles=n, seed=0).filter(ds.y)
    apf = AuxiliaryParticleFilter(mdl, n_particles=n, seed=0).filter(ds.y)
    e_sir = float(np.mean(sir.ess))
    e_apf = float(np.mean(apf.ess))
    truth = float(np.sum(KalmanFilter(mdl).filter(ds.y).log_lik_terms))
    print(f"  粒子数 N = {n}")
    print(f"  bootstrap SIR : mean ESS = {e_sir:8.1f} ({100 * e_sir / n:5.1f}%)")
    print(f"  auxiliary PF  : mean ESS = {e_apf:8.1f} ({100 * e_apf / n:5.1f}%)")
    print(f"  ESS 提升      : {e_apf / max(e_sir, 1e-9):.2f}×")
    print("\n  log-lik 偏差（相对 KF 精确值，越接近 0 越好）：")
    print(f"    bootstrap : {float(np.sum(sir.log_lik_terms)) - truth:+10.4f}")
    print(f"    auxiliary : {float(np.sum(apf.log_lik_terms)) - truth:+10.4f}")


def run_ablation(cfg: SeqForgeConfig) -> list[dict]:
    """消融：关闭组件看增益是否真实存在。

    消融必须**复用pipeline 的同一个函数/判据**，不重写一份，
    否则口径漂移会报出假增益。
    """
    _bar("消融对照（线性高斯域，seed=7）")
    spec = DEFAULT_SPECS["linear_gaussian"]
    ds = make_dataset("linear_gaussian", n_steps=cfg.n_steps, seed=7, spec=spec)
    mdl = build_model("linear_gaussian", spec, 7)
    truth = float(np.sum(KalmanFilter(mdl).filter(ds.y).log_lik_terms))
    n = cfg.n_particles
    rows: list[dict] = []

    def record(name: str, est_obj, note: str) -> None:
        est = est_obj.filter(ds.y)
        ll = float(np.sum(est.log_lik_terms))
        ess = float(np.mean(est.ess)) if est.ess is not None else float("nan")
        rows.append(
            {
                "variant": name,
                "log_lik": ll,
                "bias_vs_kf": ll - truth,
                "mean_ess": ess,
                "ess_ratio": ess / n if np.isfinite(ess) else float("nan"),
                "note": note,
            }
        )
        print(
            f"  {name:<26} ll={ll:10.3f}  bias={ll - truth:+8.3f}  "
            f"ESS={ess:7.1f} ({100 * ess / n if np.isfinite(ess) else 0:5.1f}%)  {note}"
        )

    record("apf (full)", AuxiliaryParticleFilter(mdl, n_particles=n, seed=0), "最优提议 + 辅助变量")
    record(
        "apf no_aux_resample",
        AuxiliaryParticleFilter(mdl, n_particles=n, seed=0, resample_aux=False),
        "关闭辅助重采样",
    )
    record("sir (bootstrap)", BootstrapSIR(mdl, n_particles=n, seed=0), "先验提议（弱基线）")
    record(
        "sir multinomial",
        BootstrapSIR(mdl, n_particles=n, resample="multinomial", seed=0),
        "多项式重采样（高方差）",
    )
    return rows


def main() -> int:
    """演示主入口。"""
    parser = argparse.ArgumentParser(description="SeqForge 端到端演示")
    parser.add_argument("--quick", action="store_true", help="快速模式（少步数少粒子）")
    parser.add_argument("--ablation", action="store_true", help="附带消融对照")
    parser.add_argument("--seeds", type=str, default="7,101,202")
    parser.add_argument("--out", type=str, default="artifacts")
    args = parser.parse_args()

    cfg = SeqForgeConfig(
        n_steps=40 if args.quick else 120,
        n_particles=200 if args.quick else 1000,
    )
    seeds = tuple(int(s) for s in args.seeds.split(","))
    set_all(cfg.master_seed)

    _bar("SeqForge · 序贯贝叶斯推断与状态空间估计")
    print("作者: 晨星 (CJX0712)    版本: 0.1.0    License: MIT")
    print(f"配置: n_steps={cfg.n_steps}  n_particles={cfg.n_particles}  seeds={seeds}")

    t0 = time.perf_counter()
    report = SeqForgePipeline(cfg).run(seeds=seeds)
    elapsed = time.perf_counter() - t0

    _bar(f"基准结果（{elapsed:.1f}s）")
    print(f"{'method':<10}{'family':<10}{'log_lik':>16}{'rmse':>10}{'ess':>10}")
    print("-" * 56)
    for r in report["results"]:
        if r["skipped"]:
            print(f"{r['method']:<10}{'skipped':<10}  {r['skip_reason'][:36]}")
            continue
        e_vals = [
            v.get("ess_mean", float("nan"))
            for v in r["per_dataset"].values()
            if np.isfinite(v.get("ess_mean", float("nan")))
        ]
        ess = f"{np.mean(e_vals):.1f}" if e_vals else "-"
        print(
            f"{r['method']:<10}{r['family']:<10}"
            f"{r['log_lik_mean']:>10.2f}±{r['log_lik_std']:<5.2f}{r['rmse_mean']:>10.4f}{ess:>10}"
        )

    show_kf_reference(cfg, seeds[:2])
    show_ess_comparison(cfg)

    ablation: list[dict] = []
    if args.ablation:
        ablation = run_ablation(cfg)

    cases = report.get("failure_cases") or []
    if cases:
        _bar("失败案例（全部从实测结果派生，无预设结论）")
        for i, c in enumerate(cases, 1):
            print(f"  {i}. [{c['kind']}] {c['symptom']}")
            print(f"     根因: {c['root_cause']}")
            print(f"     证据: {c['evidence']}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report["demo_meta"] = {
        "elapsed_sec": elapsed,
        "ablation": ablation,
        "quick": args.quick,
    }
    path = out_dir / "benchmark.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    _bar(f"报告已写入: {path}")
    print(f"总耗时 {elapsed:.1f}s（预算 60s）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
