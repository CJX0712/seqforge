"""SeqForge 评测层：统一指标口径、统计聚合、失败案例归因。

指标语义（一经发布不再变更，跨方法公平比较的前提）
--------------------------------------------------
======================  ==========================================
指标                    方向
======================  ==========================================
``log_lik``            **越大越好**（对数边缘似然，序列级）
``log_lik_bias``       越接近 0 越好（仅线性高斯域有真值时可用）
``rmse_state``         **越小越好**（全状态）
``rmse_obs``           **越小越好**（仅可观测子空间，部分可观测系统才有意义）
``ess_mean``           越大越好（仅粒子法）
``smoother_rmse``      越小越好（平滑）
======================  ==========================================

显著性判据
----------
沿用任务书口径：均值差 > ½(σ₁+σ₂) 记为 ``significant``。
但**该判据只进报告，不作为 CI 门禁**——粒子法含 Monte-Carlo 噪声，
用它做门禁会在 CI 上随机翻车（架构组实测同类门禁 30% 概率红灯）。
门禁只用确定性不变量（见 tests/test_invariants.py）。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.types import FilterReport, MethodResult

__all__ = [
    "aggregate",
    "collect_failure_cases",
    "format_table",
    "mean_std",
    "significance",
]


def mean_std(values: list[float]) -> tuple[float, float]:
    """返回 ``(mean, std)``；单元素时 std 记 0。"""
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def significance(delta: float, std_a: float, std_b: float) -> bool:
    """均值差是否超过 ½(σ₁+σ₂)（任务书口径）。

    仅用于报告；**不作为 CI 门禁**（见模块docstring）。
    """
    if not np.isfinite(std_a) or not np.isfinite(std_b):
        return False
    return abs(delta) > 0.5 * (std_a + std_b)


def aggregate(reports: list[FilterReport]) -> list[MethodResult]:
    """把多次 seed 的报告聚合成方法级 mean±std。

    ``skipped`` 的方法整条聚合标 skipped，并保留原因。
    """
    by_method: dict[str, list[FilterReport]] = {}
    for r in reports:
        by_method.setdefault(r.method, []).append(r)

    out: list[MethodResult] = []
    for method, rows in by_method.items():
        if all(r.skipped for r in rows):
            out.append(
                MethodResult(
                    method=method,
                    family=rows[0].family,
                    datasets=sorted({r.dataset for r in rows}),
                    log_lik_mean=float("nan"),
                    log_lik_std=float("nan"),
                    rmse_mean=float("nan"),
                    rmse_std=float("nan"),
                    skipped=True,
                    skip_reason=rows[0].skip_reason,
                )
            )
            continue
        ok = [r for r in rows if not r.skipped]
        ll_m, ll_s = mean_std([r.log_lik for r in ok])
        rm_m, rm_s = mean_std([r.rmse_state for r in ok])
        per_ds: dict[str, dict[str, float]] = {}
        for ds in sorted({r.dataset for r in ok}):
            sub = [r for r in ok if r.dataset == ds]
            m1, s1 = mean_std([r.log_lik for r in sub])
            m2, s2 = mean_std([r.rmse_state for r in sub])
            e_m, _ = mean_std([r.ess_mean for r in sub])
            per_ds[ds] = {
                "log_lik_mean": m1,
                "log_lik_std": s1,
                "rmse_mean": m2,
                "rmse_std": s2,
                "ess_mean": e_m,
            }
        out.append(
            MethodResult(
                method=method,
                family=ok[0].family,
                datasets=sorted(per_ds),
                log_lik_mean=ll_m,
                log_lik_std=ll_s,
                rmse_mean=rm_m,
                rmse_std=rm_s,
                per_dataset=per_ds,
            )
        )
    out.sort(key=lambda r: (r.skipped, -r.log_lik_mean))
    return out


def collect_failure_cases(reports: list[FilterReport], *, top_k: int = 5) -> list[dict[str, Any]]:
    """从实测结果派生失败案例（**不预设结论**）。

    三类：
    1. 最差格：某方法在某数据集上 RMSE 最大（= 相对最弱）
    2. 偏差最大格：粒子法在有真值锚点的域上偏差最大
    3.退化最严重格：ESS 最低

    每条含 ``symptom`` / ``root_cause`` / ``evidence`` 三段。
    """
    cases: list[dict[str, Any]] = []
    ok = [r for r in reports if not r.skipped and np.isfinite(r.rmse_state)]
    if not ok:
        return cases

    worst = max(ok, key=lambda r: r.rmse_state)
    cases.append(
        {
            "kind": "worst_rmse",
            "method": worst.method,
            "dataset": worst.dataset,
            "symptom": f"{worst.method} 在 {worst.dataset} 上 RMSE={worst.rmse_state:.4f}，为全表最差",
            "root_cause": _diagnose(worst),
            "evidence": f"seed={worst.seed} runtime={worst.runtime_sec:.3f}s",
        }
    )

    with_ess = [r for r in reports if not r.skipped and np.isfinite(r.ess_mean) and r.ess_mean > 0]
    if with_ess:
        worst_ess = min(with_ess, key=lambda r: r.ess_mean)
        cases.append(
            {
                "kind": "degenerate_ess",
                "method": worst_ess.method,
                "dataset": worst_ess.dataset,
                "symptom": f"{worst_ess.method} 在 {worst_ess.dataset} 上 ESS={worst_ess.ess_mean:.1f}（退化最重）",
                "root_cause": (
                    "高维稀疏观测下先验提议的权重集中；该时刻有效观测维度少，似然在粒子间区分度低"
                ),
                "evidence": f"seed={worst_ess.seed} log_lik={worst_ess.log_lik:.3f}",
            }
        )

    particle = [r for r in reports if r.family == "particle" and not r.skipped]
    if particle:
        worst_p = min(particle, key=lambda r: r.log_lik)
        cases.append(
            {
                "kind": "worst_loglik",
                "method": worst_p.method,
                "dataset": worst_p.dataset,
                "symptom": f"{worst_p.method} 在 {worst_p.dataset} 上 log_lik={worst_p.log_lik:.3f}（粒子法最低）",
                "root_cause": (
                    "bootstrap PF 的边缘似然估计向下有偏（O(1/N)），线性高斯域"
                    "有 KF 精确值可量化该偏差"
                ),
                "evidence": f"seed={worst_p.seed} rmse_state={worst_p.rmse_state:.4f}",
            }
        )
    return cases[:top_k]


def _diagnose(r: FilterReport) -> str:
    """按数据集名给出根因归因（依据 DGP 设计，不臆测）。"""
    name = r.dataset
    if name == "linear_gaussian":
        return "线性高斯域KF 已是贝叶斯最优，任何方法只能逼近，差距即蒙特卡洛误差"
    if name == "nonlinear_obs":
        return "非线性观测使一阶线性化误差累积；粒子法不受线性化限制但方差大"
    if name == "sparse_obs":
        return "部分可观测：未观测维度只能预测到条件均值，全状态 RMSE 被不可观测维主导"
    if name == "degenerate":
        return "窄似然/双峰后验使重要性权重集中，重采样后粒子多样性下降"
    return "未知数据集，需补充归因"


def format_table(rows: list[MethodResult], *, title: str = "") -> str:
    """把聚合结果渲染成固定宽度对齐的文本表（中文宽度已考虑）。"""
    header = f"{'method':<22}{'family':<10}{'log_lik':>13}{'rmse':>11}{'ess':>10}"
    sep = "-" * len(header)
    lines = []
    if title:
        lines.append(title)
    lines.append(header)
    lines.append(sep)
    for r in rows:
        if r.skipped:
            ll = "skipped"
            rm = "skipped"
            es = "skipped"
        else:
            ll = f"{r.log_lik_mean:.3f}±{r.log_lik_std:.3f}"
            rm = f"{r.rmse_mean:.4f}"
            e_vals = [v.get("ess_mean", float("nan")) for v in r.per_dataset.values()]
            e_vals = [v for v in e_vals if np.isfinite(v)]
            es = f"{np.mean(e_vals):.1f}" if e_vals else "-"
        lines.append(f"{r.method:<22}{r.family:<10}{ll:>13}{rm:>11}{es:>10}")
    return "\n".join(lines)
