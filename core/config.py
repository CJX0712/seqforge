"""SeqForge 配置：默认值 + ``ENV_SEQFORGE_*`` 覆盖 + schema 校验。

环境变量命名规范：``ENV_SEQFORGE_<大写下划线字段名>``。
例如 ``ENV_SEQFORGE_N_PARTICLES=2000`` 覆盖 ``n_particles``。
所有字段都是 int/float/bool/str，解析失败或越界一律抛 ConfigError，
**绝不静默fallback**（静默 fallback 是踩坑库反复出现的根因）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Any

from core.errors import ConfigError

__all__ = ["CONFIG_FIELDS", "SeqForgeConfig", "load_config"]

_PREFIX = "ENV_SEQFORGE_"


@dataclass(frozen=True)
class SeqForgeConfig:
    """系统级配置（不可变；改配置请重新构造）。"""

    # --- 实验规模 ---
    n_steps: int = 120
    master_seed: int = 20260904
    bench_seeds: tuple[int, ...] = (7, 101, 202)

    # --- 粒子族规模 ---
    n_particles: int = 1200
    pf_resample: str = "systematic"  # systematic | multinomial | residual

    # --- 数值护栏 ---
    cov_jitter: float = 1e-9
    max_cov_eig: float = 1e12

    # --- 输出 ---
    output_dir: str = "artifacts"
    verbose: bool = True

    def with_overrides(self, **kwargs: Any) -> SeqForgeConfig:
        """返回覆盖若干字段的新配置（不可变风格）。"""
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        for key, value in kwargs.items():
            if key not in data:
                raise ConfigError("未知配置项", detail=f"{key!r} 不在配置字段中: {sorted(data)}")
            data[key] = value
        return SeqForgeConfig(**data)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, Any]:
        """可序列化视图（写入 benchmark.json）。"""
        return {f.name: getattr(self, f.name) for f in fields(self)}


CONFIG_FIELDS = tuple(f.name for f in fields(SeqForgeConfig))

_TYPES: dict[str, tuple[type, ...]] = {
    "n_steps": (int,),
    "master_seed": (int,),
    "n_particles": (int,),
    "pf_resample": (str,),
    "cov_jitter": (float, int),
    "max_cov_eig": (float, int),
    "output_dir": (str,),
    "verbose": (bool,),
}


def _coerce(name: str, raw: str, declared: type | None) -> Any:
    """把环境变量字符串强制转为声明类型；失败抛 ConfigError。"""
    if declared is bool:
        low = raw.strip().lower()
        if low in {"1", "true", "yes", "on"}:
            return True
        if low in {"0", "false", "no", "off"}:
            return False
        raise ConfigError("布尔环境变量解析失败", detail=f"{name}={raw!r}")
    if declared is int:
        try:
            return int(raw)
        except ValueError as exc:
            raise ConfigError("整型环境变量解析失败", detail=f"{name}={raw!r}") from exc
    if declared is float:
        try:
            return float(raw)
        except ValueError as exc:
            raise ConfigError("浮点环境变量解析失败", detail=f"{name}={raw!r}") from exc
    return raw


def load_config(**kwargs: Any) -> SeqForgeConfig:
    """构造配置：默认值 → 环境变量覆盖 → 显式 kwargs 覆盖 → schema 校验。

    优先级：显式 kwargs >环境变量 > 默认值。
    """
    values: dict[str, Any] = {}

    for name in CONFIG_FIELDS:
        env_key = _PREFIX + name.upper()
        if env_key in os.environ:
            values[name] = _coerce(env_key, os.environ[env_key], _TYPES[name].__getitem__(0))
    values.update(kwargs)

    cfg = SeqForgeConfig(**values)
    _validate(cfg)
    return cfg


def _validate(cfg: SeqForgeConfig) -> None:
    """schema 校验：范围 + 枚举 + 跨字段约束。"""
    if cfg.n_steps < 10:
        raise ConfigError("n_steps 过小", detail=f"n_steps={cfg.n_steps} < 10")
    if cfg.n_particles < 50:
        raise ConfigError(
            "n_particles 过小（ESS 诊断无意义）", detail=f"n_particles={cfg.n_particles} < 50"
        )
    if cfg.pf_resample not in {"systematic", "multinomial", "residual"}:
        raise ConfigError("pf_resample 非法", detail=f"{cfg.pf_resample!r} 不在枚举内")
    if not 0.0 <= cfg.cov_jitter < 1e-3:
        raise ConfigError("cov_jitter 越界", detail=f"{cfg.cov_jitter}")
    if cfg.max_cov_eig <= 1.0:
        raise ConfigError("max_cov_eig 越界", detail=f"{cfg.max_cov_eig}")
    if len(cfg.bench_seeds) < 3:
        raise ConfigError("统计严谨性要求 ≥3 seeds", detail=f"bench_seeds={cfg.bench_seeds}")
