"""SeqForge 命令行入口。

子命令
------
``demo``       跑完整基准并落盘 ``benchmark.json``（默认）
``list``       列出全部方法 / 数据集 / 能力
``selftest``   只跑确定性不变量（CI 快速门禁，不跑随机基准）
``version``    打印版本

Windows 注意：所有打印显式 UTF-8，避免 ✅/⚠️ 触发 UnicodeEncodeError
（踩坑库 §D：非 GBK 输出流会崩）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.config import SeqForgeConfig, load_config
from data.dgp import DEFAULT_SPECS
from eval.metrics import format_table
from filters import backends
from pipeline.runner import SeqForgePipeline


def _print_banner(title: str) -> None:
    print(f"\n{'=' * 72}\n {title}\n{'=' * 72}")


def cmd_demo(args: argparse.Namespace) -> int:
    """跑基准并落盘。"""
    cfg = load_config(
        n_steps=args.steps,
        n_particles=args.particles,
        output_dir=args.out,
    )
    seeds = tuple(int(s) for s in args.seeds.split(","))
    datasets = args.datasets.split(",") if args.datasets else None

    _print_banner("SeqForge ·序贯贝叶斯推断与状态空间估计")
    print(f"配置: steps={cfg.n_steps} particles={cfg.n_particles} seeds={seeds}")
    print(f"数据集: {datasets or list(DEFAULT_SPECS)}")
    print("\n后端探测:")
    for name, probe in backends.BACKEND_PROBE.items():
        ok, info = probe()
        mark = "OK " if ok else "SKIP"
        print(f"  [{mark}] {name:<14} {info}")

    t0 = time.perf_counter()
    report = SeqForgePipeline(cfg).run(datasets=datasets, seeds=seeds)
    elapsed = time.perf_counter() - t0

    _print_banner(f"基准结果（{elapsed:.1f}s）")
    rows = report["results"]
    for m in rows:
        if m["skipped"]:
            print(f"  {m['method']:<10} skipped  {m['skip_reason']}")

    class _R:
        def __init__(self, d):
            self.__dict__.update(d)
            self.per_dataset = d["per_dataset"]

    print()
    print(format_table([_R(m) for m in rows if not m["skipped"]]))

    truth = report.get("kf_reference") or {}
    if truth:
        _print_banner("线性高斯域：KF 精确 log-lik（粒子法偏差的唯一金标准）")
        for seed, val in truth.items():
            print(f"  seed {seed:>4}:KF log-lik = {val:.6f}")
        print("\n  粒子法偏差（负值 = bootstrap 的已知向下有偏）：")
        for m in rows:
            if m["family"] != "particle" or m["skipped"]:
                continue
            pd_ = m["per_dataset"].get("linear_gaussian", {})
            if "log_lik_mean" in pd_:
                deltas = [pd_["log_lik_mean"] - truth[s] for s in truth if s in pd_]
                if deltas:
                    print(f"    {m['method']:<8} bias = {sum(deltas) / len(deltas):+8.4f}")

    cases = report.get("failure_cases") or []
    if cases:
        _print_banner("失败案例（全部从实测结果派生）")
        for i, c in enumerate(cases, 1):
            print(f"  {i}. [{c['kind']}] {c['symptom']}")
            print(f"     根因: {c['root_cause']}")
            print(f"     证据: {c['evidence']}")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "benchmark.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n报告已写入: {path}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """列出方法与数据集。"""
    _print_banner("数据集")
    for name, spec in DEFAULT_SPECS.items():
        print(f"  {name:<18} {type(spec).__name__:<20} {spec}")
    _print_banner("后端能力")
    for name, probe in backends.BACKEND_PROBE.items():
        ok, info = probe()
        print(f"  {name:<14} {'可用' if ok else '不可用'}  {info}")
    _print_banner("方法矩阵（按数据集）")
    from pipeline.runner import build_methods

    for ds in DEFAULT_SPECS:
        cfg = SeqForgeConfig()
        mdl = None
        try:
            mdl = __import__("data.dgp", fromlist=["build_model"]).build_model(
                ds, DEFAULT_SPECS[ds], 7
            )
            methods = build_methods(mdl, ds, cfg)
            names = ", ".join(sorted(methods))
        except Exception as exc:
            names = f"<构造失败: {exc}>"
        print(f"  {ds:<18} {names}")
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    """只跑不变量（CI 快速门禁）。"""
    import subprocess

    root = Path(__file__).resolve().parent
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(root / "tests" / "test_invariants.py"),
            "-q",
            "-W",
            "ignore::UserWarning",
            "-p",
            "no:cacheprovider",
        ],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    print(r.stdout[-3000:])
    return r.returncode


def cmd_version(args: argparse.Namespace) -> int:
    """打印版本。"""
    import core

    print(f"SeqForge {core.__version__} by {core.__author__} ({core.__license__})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """构造 argparse 解析器。"""
    p = argparse.ArgumentParser(
        prog="seqforge",
        description="SeqForge · 序贯贝叶斯推断与状态空间估计",
    )
    p.add_argument("--version", action="store_true", help="打印版本并退出")
    sub = p.add_subparsers(dest="cmd")

    d = sub.add_parser("demo", help="跑完整基准并落盘 benchmark.json")
    d.add_argument("--steps", type=int, default=120, help="时间步长（默认 120）")
    d.add_argument("--particles", type=int, default=1000, help="粒子数（默认 1000）")
    d.add_argument("--seeds", type=str, default="7,101,202", help="逗号分隔的 seeds")
    d.add_argument("--datasets", type=str, default="", help="逗号分隔的数据集名（默认全部）")
    d.add_argument("--out", type=str, default="artifacts", help="输出目录")
    d.set_defaults(func=cmd_demo)

    lst = sub.add_parser("list", help="列出方法/数据集/后端能力")
    lst.set_defaults(func=cmd_list)

    st = sub.add_parser("selftest", help="只跑确定性不变量（CI 门禁）")
    st.set_defaults(func=cmd_selftest)

    v = sub.add_parser("version", help="打印版本")
    v.set_defaults(func=cmd_version)
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI 主入口。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "version", False) and not getattr(args, "cmd", None):
        return cmd_version(args)
    if not getattr(args, "cmd", None):
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
