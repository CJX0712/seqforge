# SeqForge 实现架构

**作者：晨星 (CJX0712)** · 本文档描述**已实现**的架构；
选型论证见 [`selections.md`](selections.md)。

## 1. 设计出发点：本域有一条硬约束

序贯贝叶斯推断与状态空间估计与其他 ML 域有一个根本差别：

> **在线性高斯问题上，Kalman 滤波是精确贝叶斯最优，任何算法都不可能超越它。**

这不是工程能力问题，是数学事实（卡尔曼滤波 = 线性高斯下的条件高斯
后验闭式解）。它决定了本库的设计：

| 目标 | 设计对策 |
|------|---------|
| "在主指标上打败 KF" | ❌ 数学上不可达 ⇒ 任何此类门禁都是自欺 |
| "把不可能超越变成可验证的天花板" | ✅ KF 精确 log-lik 作为偏差金标准 |
| "在有真实超越空间的地方竞争" | ✅ 提议分布设计、高维退化鲁棒性 |
| "门禁不依赖随机波动" | ✅ 确定性不变量，而非统计量 |

## 2. 调用拓扑（单向无环）

```
                cli.py  ·  examples/run_demo.py
                          │
                          ▼
                 pipeline/runner.py            SeqForgePipeline.run()
                          │
     ┌────────────┬───────┴────────┬──────────────┐
     ▼            ▼                ▼              ▼
  data/       filters/        smoothers/       eval/
  models      kalman          particle_        metrics
  dgp         particle        smoother
              backends
     └────────────┴────────────────┴──────────────┘
                          │
                          ▼
                        core/
        types · errors · config · interfaces · seed · numeric
```

**无环保证**：`core` 不 import 任何上层；`data` 只依赖 `core`；
`filters`/`smoothers` 依赖 `data` 与 `core`；`eval` 只依赖 `core.types`；
`pipeline` 是唯一编排层；`cli` 只调用 `pipeline`。

## 3. 统一协议

所有方法共享三个签名，能力差异用**标记**而非签名表达：

```python
class SequentialEstimator(Protocol):
    def filter(self, y) -> FilterEstimate: ...
    def smooth(self, y) -> SmoothingResult: ...  # 不可用抛 NotImplementedError
    def log_likelihood(self, y) -> float: ...
```

| 标记 | 含义 |
|------|------|
| `family = "kalman" \| "particle"` | 归族，决定能力矩阵 |
| `reports_ess` | 是否报告 ESS（解析法为 False） |
| `smooth()` 抛异常 | 无平滑器（EKF/UKF：雅可比沿平滑路径需重算） |

**收益**：pipeline 对任意方法统一计时、统一取指标，跨方法比较才公平。
平滑不可用时如实标 `skipped`，**不伪造数字**。

## 4. 稀疏观测的统一口径（跨族一致性）

`sparse_obs` DGP 用 `NaN` 表示缺失观测。数学上，缺失维度等价于
``R_ii → ∞`` 且创新项 = 0，也就是**跳过该维度的更新**。

关键纪律：**所有方法走同一个判据** `filters.kalman._observed_rows(y_t)`：

```python
obs_idx = np.flatnonzero(~np.isnan(y_t))
if len(obs_idx) == 0:   # 该时刻无观测 ⇒ 纯预测，log-lik 增量 = 0
H_o = mdl.H[obs_idx];  R_o = mdl.R[np.ix_(obs_idx, obs_idx)]
```

若某个方法用 `np.nan_to_num` 后照常更新，它会把缺失当零观测，
**系统性拉低 log-lik** —— 跨方法比较里这是致命的。
（实测：EKF/IEKF/UKF 最初都踩了这个坑，log-lik 直接变 NaN。）

## 5. 指标语义与双 RMSE 口径

| 指标 | 方向 | 定义 |
|------|------|------|
| `log_lik` | 越大越好 | Σₜ log p(yₜ \| y₁:ₜ₋₁) |
| `rmse_state` | 越小越好 | 全状态 RMSE |
| `rmse_obs` | 越小越好 | **仅可观测子空间** RMSE |
| `ess_mean` | 越大越好 | ESS = 1/Σw² |
| `smoother_rmse` | 越小越好 | 平滑轨迹 RMSE |

**为什么必须双 RMSE 口径**：部分可观测系统里，未观测维度**永远**只能
预测到条件均值，其 RMSE 只反映先验方差而与方法质量无关。
算进全状态 RMSE 会让所有方法"看起来一样差"，真正的差异被淹没。
`SequentialDataset._observed_idx` 记录 H 覆盖的维度。

## 6. 确定性设计

### 6.1 随机源隔离

所有随机性显式注入，不使用全局 `np.random` 流：

1. DGP 构造：`RandomState(seed + offset)`，offset 按 DGP 名隔离
2. 方法构造：各方法持自己的 `RandomState(method_seed)`

这样「某模块多抽一次随机数导致全局流错位」从根上不存在。

### 6.2 为什么用 legacy RandomState

`np.random.Generator`（PCG64）的流**不保证跨 numpy 版本 / BLAS 稳定**
（NEP 19 只保证 legacy `RandomState`）。粒子滤波是随机算法，
用 Generator 会让数值门禁在 CI 的 numpy 版本漂移后**随机翻转**。

因此统一用 `np.random.RandomState`（MT19937），并把
`requirements.lock.txt` 的 numpy 版本钉死。

**实测**：同 seed 两次运行，72 行 benchmark 数据**逐位一致**。

## 7. 数值稳定策略（含实测踩坑）

| 风险 | 对策 | 实测症状 |
|------|------|---------|
| 协方差失去半正定性 | `stabilize_cov`：对称化 → **SVD** → 奇异值 clip → 特征值兜底 | 秩亏协方差下 `eigh` 抛 `Eigenvalues did not converge`（N=1000 / n=30 仍触发，粒子共线） |
| Cholesky 失败 | `safe_cholesky`：stabilize + 指数递增 jitter 重试 3 次 | — |
| sigma 点精度 | 用**特征分解**而非 `cholesky(c·P)` | `c = n+λ` 在 α=1e-3 时约 4e-6，Cholesky 精度不足 |
| radar 奇异性 | `r = max(hypot(x0,x1), 0.15)`，雅可比同步 | 1/r 在 r→0 爆炸 ⇒ EKF/IEKF 协方差溢出到 inf |
| log 域下溢 | 全部权重 log 域累加（`logsumexp`） | r=0.05 的窄似然在概率域直接下溢到 0 |
| 权重偏置 | 重要性采样三件套 `log π − log q` | 缺 `log q` 项 ⇒ 偏差从 −33 变 +988 |
| IEKF 协方差更新 | `(I−KH)P(I−KH)ᵀ + KRKᵀ` | 初版写 `P @ I_KH.T @ I_KH` 而 `I_KH` 恒为 I ⇒ **完全没有协方差更新** |
| IEKF 收敛判据 | `x + K(v + H(x_k − x))` | 初版写成 `H(x − x_k)` ⇒ 线性问题上不收敛（\|Δx\|=58.3） |

## 8. 门禁策略：为什么不用统计量

实测（3 seeds，线性高斯域）：

| 量 | 观察 |
|----|------|
| `log_lik` 的 seed 间 std | 可达 0.7~1.4（KF ±222.67, APF ±144.86） |
| 方法间差值 | 仅 0.2~2.1 |
| 后果 | 「均值差 > ½(σ₁+σ₂)」在 3~5 seeds 下**随机翻转** |

粒子法的 Monte-Carlo 噪声与方法间真实差异同量级，统计门禁在 CI 上必然翻车。

**本库的分工**：

- **门禁**（`tests/test_invariants.py`）→ 22 条确定性不变量
  - 闭式解交叉验证（KF vs 独立多元高斯实现，Δ < 1e-13）
  - 代数恒等式（UKF/EKF/IEKF 线性退化 2.7e-15 / 0 / 5.6e-17；σ 点矩匹配）
  - 理论性质（bootstrap 向下有偏、RTS 协方差 Loewner 序）
- **报告**（`benchmark.json`）→ mean ± std、显著性标记、失败案例

## 9. 降级链

```
                 ┌─ statsmodels (可选 Tier-0 参照)
available_*() ───┤
                 └─ pykalman    (可选 Tier-0 参照)
                        │ 都不可用
                        ▼
              仅内置方法（numpy + scipy）
                        │
              benchmark 标 skipped + 原因
              ★不伪造任何数字
```

`test_offline_fallback_works_without_backends` 用 monkeypatch 把所有后端
探测打成 False，验证流水线仍能跑完 —— **保证兜底不是口号**。

## 10. 已知的诚实限制

1. **稀疏观测域 APF 的 ESS 低于 SIR**（n=30: 4.3% vs 37.2%）。
   根因：可观测维度太少时最优提议退化为先验提议，同时辅助重采样仍按 α
   损失多样性。这是**真实发现**，不是实现 bug。
2. **消融显示「关闭辅助重采样」略优**（偏差 −32.97 vs −35.07），
   说明真正增益来自「最优提议」而非「辅助重采样」。负结果已保留。
3. **statsmodels 0.15 移除了 `initial_state`**，无法对齐初值口径
   （实测三种初值下 llf 与自研 KF 差 −141~−59），故只作参照不作逐位基线。
4. **FFBS 的 `filter` 与 `smooth` 共用同一份前向粒子集**，
   保证口径一致，但无法评估"不同前向粒子"的平滑变体。
5. **`sparse_obs` 的 A 曾是 rank-3 低秩升维**，测的是低秩结构而非维度
   灾难，已改为全秩 A（`lowrank_transition` 保留为独立对照）。
