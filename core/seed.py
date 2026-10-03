"""SeqForge 全局确定性入口。

唯一 seed 入口：`set_all(seed)`。

设计要点（踩坑库 §A「门禁在 CI 上随机翻车」的根因）：
粒子滤波是随机算法，若用 `np.random.Generator`（PCG64），
随机流不保证跨 numpy 版本 / BLAS 稳定（NEP 19 只保证 legacy
`RandomState`）。因此本项目**统一使用 legacy `np.random.RandomState`**，
其 MT19937 流与 numpy 版本解耦，保证：
  - 同 seed 两次运行 benchmark.json 逐位一致
  - CI 矩阵3.12/3.13 上门禁不会随机翻转

所有模块必须通过本模块取随机源，禁止在业务代码里直接
`np.random.seed` / `np.random.rand`。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

__all__ = ["SeedState", "current_seed", "get_rng", "seed_state", "set_all", "spawn_rng"]

_DEFAULT_SEED = 20260904


@dataclass
class SeedState:
    """全局 seed 状态。"""

    seed: int = _DEFAULT_SEED
    seeds: dict[str, int] = field(default_factory=dict)


_STATE = SeedState()


def set_all(seed: int | None = None) -> int:
    """设置全局 seed：stdlib random、numpy legacy 全局状态、模块内命名 RNG。

    Parameters
    ----------
    seed
        主 seed。``None`` 表示沿用当前主 seed（便于子流程复用）。

    Returns
    -------
    int
        实际生效的主 seed。
    """
    if seed is not None:
        _STATE.seed = int(seed)
    s = _STATE.seed
    # legacy RandomState：MT19937，跨 numpy 版本流稳定
    np.random.seed(s % (2**32 - 1))
    random.seed(s)
    return s


def current_seed() -> int:
    """返回当前主 seed。"""
    return _STATE.seed


def spawn_rng(name: str) -> np.random.RandomState:
    """按名字派生一个确定性子 RNG。

    同一主 seed + 同一 name ⇒ 同一子流；不同 name 互不干扰，
    避免"某个模块多抽了一次随机数导致全局流错位"这类难查 bug。

    Parameters
    ----------
    name
        子流名称，例如 ``"dgp.linear_gaussian"``。

    Returns
    -------
    numpy.random.RandomState
        legacy RandomState 实例。
    """
    if name not in _STATE.seeds:
        # 用主 seed 与名字的稳定哈希派生，绝不使用 Python 内置 hash（随机化）
        digest = 0
        for ch in name.encode("utf-8"):
            digest = (digest * 131 + ch) % (2**31 - 1)
        _STATE.seeds[name] = int((_STATE.seed + digest * 7919) % (2**31 - 1))
    return np.random.RandomState(_STATE.seeds[name])


def get_rng(name: str) -> np.random.RandomState:
    """`spawn_rng` 的语义别名，语义更贴近"取一个随机源"。"""
    return spawn_rng(name)


def seed_state() -> dict[str, object]:
    """导出可序列化的 seed 状态快照（写入 benchmark.json 用）。"""
    return {"master_seed": _STATE.seed, "derived_streams": dict(sorted(_STATE.seeds.items()))}
