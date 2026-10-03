"""SeqForge 主流水线：统一跑「方法 × 数据集 × seed」的基准。

调用方向（单向无环）：``cli → pipeline → {data, filters, smoothers, eval} → core``
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from core.config import SeqForgeConfig
from core.seed import set_all
from core.types import FilterReport
from data.dgp import DEFAULT_SPECS, build_model, make_dataset
from eval.metrics import aggregate, collect_failure_cases
from filters.kalman import (
    ExtendedKalmanFilter,
    IEKFFilter,
    KalmanFilter,
    UnscentedKalmanFilter,
)
from filters.particle import AuxiliaryParticleFilter, BootstrapSIR
from smoothers.particle_smoother import FFBS

__all__ = ["SeqForgePipeline", "build_methods"]

# 非线性观测才需要 EKF/IEKF/UKF；线性域它们必须精确退化为 KF（单测断言）
_NONLINEAR_ONLY = {"ekf", "iekf", "ukf", "ffbs"}


def build_methods(model, dataset_name: str, cfg: SeqForgeConfig) -> dict[str, Any]:
    """构造该数据集上适用��方法集合。

    线性域**故意不放** EKF/UKF：它们在线性问题上必须精确等于 KF，
    放进来只会产出重复数字，掩盖真实差异。非线性域才启用。
    """
    is_linear = dataset_name == "linear_gaussian"
    methods: dict[str, Any] = {
        "kf": KalmanFilter(model),
        "sir": BootstrapSIR(model, n_particles=cfg.n_particles, seed=0),
        "apf": AuxiliaryParticleFilter(model, n_particles=cfg.n_particles, seed=0),
    }
    if not is_linear:
        methods["ekf"] = ExtendedKalmanFilter(model, model.emission, model.jacobian_h)
        methods["ukf"] = UnscentedKalmanFilter(model, model.emission)
        methods["iekf"] = IEKFFilter(model, model.emission, model.jacobian_h)
        methods["ffbs"] = FFBS(model, n_particles=max(400, cfg.n_particles // 2), seed=0)
    return methods


class SeqForgePipeline:
    """端到端基准流水线。

    确定性契约：同一 ``master_seed`` 下两次运行，除``elapsed_sec`` 外
    所有数值**逐位一致**。所有随机性都来自：
    - DGP 构造的显式 ``seed``
    - 各方法构造时注入的独立 ``RandomState(seed)``

    不依赖全局 ``np.random`` 流，因此不同方法的粒子数/重采样策略
    变化不会互相污染（踩坑库§G「随机源被其他模块消耗」）。
    """

    def __init__(self, cfg: SeqForgeConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else SeqForgeConfig()
        set_all(self.cfg.master_seed)

    def run(
        self,
        datasets: list[str] | None = None,
        seeds: tuple[int, ...] | None = None,
        *,
        n_particles: int | None = None,
    ) -> dict[str, Any]:
        """跑完整基准，返回可JSON 序列化的报告字典。"""
        cfg = self.cfg if n_particles is None else self.cfg.with_overrides(n_particles=n_particles)
        names = datasets if datasets is not None else list(DEFAULT_SPECS)
        use_seeds = seeds if seeds is not None else cfg.bench_seeds
        reports: list[FilterReport] = []

        for name in names:
            spec = DEFAULT_SPECS[name]
            for seed in use_seeds:
                set_all(cfg.master_seed)
                ds = make_dataset(name, n_steps=cfg.n_steps, seed=seed, spec=spec)
                model = build_model(name, spec, seed)
                methods = build_methods(model, name, cfg)
                for method_name, est in methods.items():
                    reports.append(self._run_one(est, ds, method_name, name, seed))
        return self._assemble(reports, names, list(use_seeds), cfg)

    def _run_one(self, est, ds, method_name: str, dataset: str, seed: int) -> FilterReport:
        """跑单方法单数据集单 seed，失败时如实标 skipped。"""
        family = getattr(est, "family", "kalman")
        backend = getattr(est, "name", method_name)
        t0 = time.perf_counter()
        try:
            result = est.filter(ds.y)
            ll = float(np.sum(result.log_lik_terms))
            rmse = ds.rmse(result.x_filt)
            ess = float(np.nanmean(result.ess)) if result.ess is not None else float("nan")
            extra = {"rmse_obs": rmse["rmse_obs"]}
            # 平滑单独评估（失败只记录，不影响滤波主指标）
            try:
                sm = est.smooth(ds.y)
                extra["smoother_rmse"] = ds.rmse(sm.x_smooth)["rmse_obs"]
            except (NotImplementedError, Exception) as exc:
                extra["smoother_rmse"] = float("nan")
                extra["smoother_skip"] = f"{type(exc).__name__}: {exc}"[:120]
            return FilterReport(
                method=method_name,
                family=family,
                backend=backend,
                dataset=dataset,
                seed=seed,
                log_lik=ll,
                rmse_state=rmse["rmse_state"],
                n_steps=ds.n_steps,
                runtime_sec=time.perf_counter() - t0,
                ess_mean=ess,
                smoother_rmse=extra.get("smoother_rmse", float("nan")),
                extra=extra,
            )
        except Exception as exc:
            return FilterReport(
                method=method_name,
                family=family,
                backend=backend,
                dataset=dataset,
                seed=seed,
                log_lik=float("nan"),
                rmse_state=float("nan"),
                n_steps=ds.n_steps,
                runtime_sec=time.perf_counter() - t0,
                skipped=True,
                skip_reason=f"{type(exc).__name__}: {exc}"[:160],
            )

    def _assemble(
        self,
        reports: list[FilterReport],
        datasets: list[str],
        seeds: list[int],
        cfg: SeqForgeConfig,
    ) -> dict[str, Any]:
        """组装最终报告：聚合 + 失败案例 + 线性域真值对照。"""
        results = aggregate(reports)
        truth = self._kf_reference(datasets, seeds)
        return {
            "system": "SeqForge",
            "author": "晨星",
            "config": cfg.to_dict(),
            "datasets": datasets,
            "seeds": seeds,
            "results": [
                {
                    "method": r.method,
                    "family": r.family,
                    "log_lik_mean": r.log_lik_mean,
                    "log_lik_std": r.log_lik_std,
                    "rmse_mean": r.rmse_mean,
                    "rmse_std": r.rmse_std,
                    "per_dataset": r.per_dataset,
                    "skipped": r.skipped,
                    "skip_reason": r.skip_reason,
                }
                for r in results
            ],
            "kf_reference": truth,
            "failure_cases": collect_failure_cases(reports),
            "raw": [
                {
                    "method": r.method,
                    "dataset": r.dataset,
                    "seed": r.seed,
                    "log_lik": r.log_lik,
                    "rmse_state": r.rmse_state,
                    "rmse_obs": r.extra.get("rmse_obs"),
                    "ess_mean": r.ess_mean,
                    "smoother_rmse": r.smoother_rmse,
                    "runtime_sec": r.runtime_sec,
                    "skipped": r.skipped,
                    "skip_reason": r.skip_reason,
                }
                for r in reports
            ],
        }

    def _kf_reference(self, datasets: list[str], seeds: list[int]) -> dict[str, Any]:
        """线性高斯域的KF 精确 log-lik（粒子法偏差的唯一金标准）。

        这是本系统最重要的交叉验证锚点：KF 在线性高斯下是**精确**贝叶斯，
        任何粒子法只能以 O(1/N) 逼近它。
        """
        if "linear_gaussian" not in datasets:
            return {}
        out: dict[str, Any] = {}
        spec = DEFAULT_SPECS["linear_gaussian"]
        for seed in seeds:
            ds = make_dataset("linear_gaussian", n_steps=self.cfg.n_steps, seed=seed, spec=spec)
            model = build_model("linear_gaussian", spec, seed)
            est = KalmanFilter(model).filter(ds.y)
            out[str(seed)] = float(np.sum(est.log_lik_terms))
        return out
