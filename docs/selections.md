# SeqForge 选型方案（领域调研）

> 域：**序贯贝叶斯推断 / 状态空间估计**（KF 家族 + 粒子滤波家族 + 平滑器）
> 目标环境：Windows 11 · 无 GPU · AMD Ryzen 7 H255 / 16GB · Python 3.13.14 · 托管解释器
> 核实时间：**2026-10-03**（本文所有 wheel/维护状态结论均经 PyPI JSON API 与 WebSearch 实测）

---

## 0. 结论先行（TL;DR）

| 决策 | 结论 |
|---|---|
| **主基线（自研）** | 纯 numpy 实现 KF / EKF / IEKF / UKF / RTS + bootstrap SIR / APF / RBPF / FFBS。Tier-1 零下载可跑。 |
| **Tier-0 公平基线** | **`statsmodels` 0.15.0**（线性高斯 SSM + MLE）+ **`pykalman` 0.11.2**（KF/UKF/EM/smooth）。二者均为 2025–2026 活跃维护。 |
| **明确排除** | **`filterpy` 1.4.5 = 僵尸库**，且在 numpy 2.5.3 上**必然运行时崩溃**（已实测复现）。**`particles` 0.4 强制 `numpy<2`**，与本环境 numpy 2.5.3 冲突。 |
| **不引入** | torch（124MB wheel，CPU 收益为负）、numba（JIT 预热与 60s 预算冲突）、pymc（本域只需前向推断，非采样推断）。 |
| **最大工程风险** | **粒子滤波的 Monte-Carlo 噪声让 CI 门禁随机翻车**。实测跨 seed 相对标准差 **3.79%**；"均值 −0.5σ" 门禁在 **30% 的运行**上红灯。解法：**门禁只用确定性不变量，统计量只进报告**。 |
| **性能预算** | 实测远低于预算（N=2000 粒子、T=200 单次 PF 仅 69ms）。**建议主动提高粒子数与 seed 数，而非降级。** |

---

## 1. 选型矩阵

### 1.1 总表（wheel 可用性为 PyPI JSON API 实测，非记忆）

| 包 | 最新版 | 最后发布 | 维护状态 | win_amd64 + cp313 wheel | 在本系统中的角色 | 装不上时的降级路径 |
|---|---|---|---|---|---|---|
| **numpy** | 2.5.3 | 活跃 | 🟢 官方 | ✅ `numpy-2.5.3-cp313-cp313-win_amd64.whl` | **必需**。全部算法底座 | 无（Tier-1 硬依赖） |
| **scipy** | 1.18.1 | 活跃 | 🟢 官方 | ✅ `scipy-1.18.1-cp313-cp313-win_amd64.whl`（36.6MB） | 可选加速：`linalg.cholesky`/`solve_triangular`、`special.logsumexp` | 纯 numpy 回退：`core/numeric.py` 已自实现 `logsumexp` / `safe_cholesky` / `stabilize_cov` |
| **statsmodels** | 0.15.0 | 活跃 | 🟢 官方（`tsa.statespace` 为旗舰模块） | ✅ `statsmodels-0.15.0-cp313-cp313-win_amd64.whl`（11.3MB） | **Tier-0 基线**：线性高斯 SSM 边缘似然 + `KalmanSmoother` + `MLEModel.fit` 参数自估计 | `available_statsmodels()` → False，benchmark 标 `skipped`，不伪造数字 |
| **pykalman** | 0.11.2 | **2026-01-31** | 🟢 活跃（Dependabot 驱动的持续维护） | ✅ `pykalman-0.11.2-py2.py3-none-any.whl`（**纯 Python wheel，无需编译**） | **Tier-0 基线**：`KalmanFilter` / `UnscentedKalmanFilter` / `smooth()` / `em()` / `pykalman.sqrt`（Cholesky 平方根滤波） | 同上，标 `skipped` |
| **scikit-learn** | 1.9.1 | 活跃 | 🟢 官方 | ✅ `scikit-learn-1.9.1-cp313-cp313-win_amd64.whl` | **仅用于指标对齐**（可选）。RMSE/MSE 自行实现以避免同名遮蔽（见 §3.6） | 用 numpy 手写指标 |
| **filterpy** | 1.4.5 | **2018-10-10** | 🔴 **僵尸库**（health 26/100 "At Risk"，OpenSSF 判定 unmaintained） | ❌ **仅 sdist `.zip`，从无 wheel** | **不作为基线**（见 §1.2） | 直接排除 |
| **particles** | 0.4 | 2023-11-06 | 🟡 半停滞 | ✅ wheel 存在 | **不采用**：依赖 `numpy<2` | 与 numpy 2.5.3 硬冲突，排除 |
| **pyds** | 0.2.0 | 停滞 | 🔴 | ❌ 仅 `.tar.bz2`/`.tar.gz`，无 wheel | 不采用 | 排除 |
| **simdkalman** | 1.0.4 | 停滞 | 🔴 | ✅ `py2.py3-none-any.whl` | 不采用：仅线性 KF 向量化，无平滑/UKF/PF，比较价值低 | 排除 |
| **torch** | 2.14.1 | 活跃 | 🟢 | ✅ `torch-2.14.1-cp313-cp313-win_amd64.whl`（**124.1MB**） | **不引入**：无 GPU 下对本域无收益，124MB 违反"轻量可复现"原则 | 排除 |
| **numba** | 0.68.0 | 活跃 | 🟢 | ✅ `numba-0.68.0-cp313-cp313-win_amd64.whl`（2.8MB）+ llvmlite 0.50.0 ✅ | **不作为必需**：JIT 预热与 60s 预算冲突 | 可作 `--jit` 可选加速档 |
| **pymc** | 6.3.2 | 活跃 | 🟢 | ✅ `pymc-6.3.2-py3-none-any.whl` | **不引入**：本域是**前向推断/滤波**，非后验采样；PyMC 需 PyTensor，数十倍依赖与内存 | 排除 |

### 1.2 `filterpy` 为什么必须排除（决定性证据）

三条独立证据，任一条都足以否决：

1. **无 wheel + 构建即挂**。PyPI 最新版仅 1 个文件 `filterpy-1.4.5.zip`（sdist），**历史上从未发布过 wheel**。实测 `pip install filterpy` 直接挂起被 SIGTERM。解压后可见其 `setup.py` 在构建期执行 `import filterpy`（自导入以读版本号），且**无 `pyproject.toml`**、无 `install_requires`——现代 pip 的 PEP 517 隔离构建会因拿不到 numpy 而失败。违反"装不上不许硬编译"的硬约束。
2. **numpy 2 运行时必然崩溃（已实测）**。`filterpy` 1.4.5 的 `KalmanFilter.mahalanobis` 实现为 `sqrt(float(dot(dot(y.T, SI), y)))`。numpy 2 移除了"ndim>0 的 size-1 数组隐式转标量"。在 **Python 3.13.14 + numpy 2.5.3 实测**：
   ```
   TypeError: only 0-dimensional arrays can be converted to Python scalars
   ```
   该修复**未进入 master**，只能靠 `numpy<2` 绕过——而本环境必须用 numpy 2.5.3。
3. **维护真空**。最后 release 2018-10-10，最后 commit 2022-08-22；OpenSSF Scorecard 报 unmaintained；配套书籍仓库 2024-07 后停更。作者计划的 2.0（py3.5+）从未发布。

> 若强行纳入，读者会认为"对比基线选了一个坏掉的库"，反而**削弱**报告可信度。故：既不作基线，也不作参考实现来源。其算法价值已由教科书与 pykalman/statsmodels 覆盖。

### 1.3 公平基线的判据

放进 benchmark 的外部库须满足：① 活跃维护（近 12 个月有 release 或 commit）；② 有 cp313/win_amd64 可安装 wheel；③ 覆盖本域的**主算法**而非玩具子集；④ 有公开 API 可被统一适配。

- `statsmodels`：①②③④ 全中（线性高斯 + MLE + RTS/Durbin-Koopman 平滑 + `low_memory` 模式）。
- `pykalman`：①②③④ 全中，且额外提供 **Cholesky 平方根滤波**（数值稳定性对照组，价值独特）。
- 二者**互补而非重叠**：statsmodels 强在 MLE 与线性高斯理论完备性，pykalman 强在 API 简洁 + UKF + EM + 平方根滤波。用它们做基线，读者会认为对比是专业且公平的。

---

## 2. 模块架构

### 2.1 单向无环依赖

```
cli.py ──► pipeline/ ──► data/
                │    ├─► filters/    ─┐
                │    ├─► smoothers/  ─┤  仅依赖 core/
                │    └─► inference/  ─┤
                └─► core/  ◄─────────┘
```

**硬约束**：`core/` 不得 import 任何上层模块；`filters/` 与 `smoothers/` 之间**不得互相 import**（RTS 需要 KF 的输出，但通过 `FilterEstimate` 数据契约传递，而非直接调用）。由 `tests/test_architecture.py` 用 AST 静态检查强制。

### 2.2 统一 Protocol：三能力用一套签名暴露

核心设计问题：KF（解析）、PF（随机）、平滑器三类方法如何**共用一套签名**而非各写一套？

**答案：单入口 + capability 声明 + 结果对象携带可选字段。**

```python
@runtime_checkable
class SequentialEstimator(Protocol):
    name: str
    family: str  # "kalman" | "particle" | "baseline"
    reports_ess: bool  # 能力声明：是否报告 ESS
    supports_smoothing: bool

    def filter(self, y: np.ndarray) -> FilterEstimate: ...
    def smooth(self, y: np.ndarray | None) -> SmoothingResult | None: ...
    def log_likelihood(self, y: np.ndarray) -> float: ...
```

**三种能力如何映射到同一接口**：

| 能力 | 表达方式 | 调用方处理 |
|---|---|---|
| **部分可观测** | 所有方法**一律**接 `y: (T, m)`；缺失观测用 `np.nan` 占位（KF/UKF 走"预测不更新"分支；PF 对该步权重置为均匀/跳过） | 透明，无需分支 |
| **平滑** | 同一 `smooth()` 签名。`supports_smoothing=False` 的实现**直接返回 `None`**（而非抛异常），避免 try/except 驱动控制流 | `res = m.smooth(y)`；`res is None` → 标 `skipped` |
| **边缘似然** | 同一 `log_likelihood()`。语义锁定为**越大越好**（`log_lik_mean`）；RMSE 越小越好。粒子法的 log-lik 是**估计值**，报告须标注 `ll_is_estimate=True` | 统一比较，无需方向判断 |

**关键设计**：`log_likelihood()` 对**不可解析边缘似然**的方法（如 bootstrap SIR 之外的变体）返回 `float("nan")` 并置 `extra["ll_available"]=False`，而不是给一个不可比的数字。benchmark 表格据此显示 `n/a` 而非 `0.000`——这正是"不伪造数字"的落地。

**适配外部基线**：`statsmodels` / `pykalman` 各写一个 `StatsmodelsKalmanAdapter` / `PykalmanAdapter`，把 `FilteredResults.llf` 映射到 `log_likelihood()`。适配器放在 `filters/baselines/`，**只在 import 成功时构造**，导入失败即从注册表消失。

### 2.3 `available_*()` 能力探测层

```python
@dataclass(frozen=True)
class BackendStatus:
    name: str
    available: bool
    version: str | None
    reason: str = ""  # 不可用时的**具体**原因，如 "numpy>=2 required by particles"


def available_statsmodels() -> BackendStatus: ...  # importlib.util.find_spec + 版本比对
def available_pykalman() -> BackendStatus: ...
def available_filterpy() -> (
    BackendStatus
): ...  # 恒返回 available=False, reason="unmaintained+sdist-only+numpy2-incompatible"
```

探测**必须在 import 之前完成**，且捕获 `ImportError` / `OSError`（DLL 加载失败在 Windows 上表现为 `ImportError` 的子类）。任何 `BackendStatus.available == False` 的后端，其方法在 registry 中被标记 `skipped=True` + `skip_reason=<reason>`，**且不产生任何指标行**。

**三态而非二态**：`available`（能否 import）→ `capable`（能否在本 DGP 上跑，如 PF 在线性高斯 DGP 上照跑但无意义）→ `meaningful`（结果是否有解释价值）。benchmark 报告用 `status ∈ {ok, skipped_unavailable, skipped_inapplicable, failed}` 四态，避免"能跑但无意义"的结果混进对比表。

---

## 3. 风险与坑位预判

> 标注【已实测】的均在本机 Python 3.13.14 + numpy 2.5.3 上复现过。

### 3.1 数值稳定性

**(a) Joseph 形式 vs 简化形式** — 症状：`P` 逐渐失去对称正定性，`np.linalg.cholesky` 报 `Matrix is not positive definite`（已实测该异常文本）。根因：简化形式 `(I-KH)P` 在 `K` 病态或 `P` 近乎奇异时引入负特征值，且舍入误差非对称累积。修法：**默认 Joseph 形式** `P=(I-KH)P(I-KH)ᵀ + K R Kᵀ + Q`；每次更新后 `P ← (P+Pᵀ)/2` 强制对称。

> **诚实校准**：我实测在温和条件（`Q=diag(1e-4,1e-2)`、T=400）下两种形式**特征值均为正、未分化**。Joseph 的优势体现在**极端条件**（`Q` 接近奇异、长序列）。因此：Joseph 作为默认不是因为"必然需要"，而是因为它**代价恒定**（多 2 次矩阵乘，相对总耗时增量 <20%）而**风险消除**。这是"便宜的保险"，不是"必需的补丁"——不要在文档里夸大成"简化形式一定会崩"。

**(b) 协方差半正定退化** — 症状：`eigvalsh` 出现 `-1e-18` 量级负值。根因：浮点舍入。修法：`core/numeric.py::stabilize_cov` 已有实现——对称化后做特征值截断 `eig ← max(eig, 0)`，重建 `V diag(eig) Vᵀ`。

**(c) log-likelihood 下溢** — 症状：长序列下边缘似然变成 `-inf` 或 `nan`。根因：逐步连乘 `∏ p(y_t|y_<t)` 在 T 大时指数下溢。【已实测】20 万步朴素连乘直接得到 `0.0`。修法：**全程 log 域累加**，每步独立算 `log N(y_t; ŷ, S)` 并累加到标量；粒子权重用 log-sum-exp：`ll += logsumexp(logw)`。绝对不要用 `exp(cumulative)`。

**(d) Cholesky 失败** — 症状：`LinAlgError`。根因：`S = HPHᵀ+R` 因舍入不再正定。【已实测】对不定矩阵 `[[1,2],[2,1]]`，`eps=1e-12 / 1e-9 / 1e-6` 的 jitter **全部失败**——说明**微扰救不了真不定**。修法：分两级——① 先 `stabilize_cov`（特征值截断）；② 再试 Cholesky；③ 仍失败则退化为 `solve(P, b)`（LU），并在 `extra["degraded"]=True` 记录，报告中可见而非静默。

### 3.2 随机性的全局确定性 ★最高风险

**症状**：同一条命令在 CI 上两次结果不同，断言随机翻红。

**根因（已实测）**：粒子滤波跨 seed 的对数边缘似然**相对标准差 3.79%**（12 seed，mean=-93.67，std=3.55，range [-99.42, -86.58]）。这不是 bug，是 Monte-Carlo 的固有方差。

**本机已验证的确定性事实**：
- `np.random.RandomState(42)` 与 `default_rng(42)` 在同进程内**完全可复现**（逐位相等）。
- `default_rng(12345).standard_normal(1000)` 的和**跨进程一致**：`8.258392084277878`（两次独立进程调用相同）。
- 实际 bit generator 为 `PCG64`。

**修法（四层）**：
1. **禁止全局 RNG**。`core/seed.py::spawn_rng(name)` 按用途派生命名流（`get_rng("dgp")` / `get_rng("pf")` / `get_rng("resample")`），各模块只拿自己的流，禁止 `np.random.*` 全局调用（可用 monkeypatch 测试 + AST 静态检查双保险）。
2. **用 `RandomState` 而非 `Generator`（已落到现有代码）**。依据 numpy 官方兼容策略原文：`Generator` **不提供版本兼容保证**（"No Compatibility Guarantee"，比特流可能随算法改进而变），且 `Generator.multivariate_normal` 依赖 LAPACK 分解，**不同 LAPACK 版本会产出完全不同的随机流**；而 `RandomState`（MT19937）拥有**跨版本严格更严的兼容承诺**。本域大量使用 `multivariate_normal`——**这正是必须选 RandomState 的决定性理由**，而非个人偏好。
3. **锁死 numpy 版本**：`requirements.txt` 精确钉 `numpy==2.5.3`。因为连 RandomState 也仅保证"同版本、同环境、同机器"，跨 numpy 版本仍可能因 `multivariate_normal` 的 LAPACK 路径而漂移。
4. **结果里存 seed**：所有 `FilterReport` 携带 `seed` 字段，任何数字都能被单点复现。

### 3.3 CI 门禁不得随机翻车 ★本域头号工程风险

**先说被我实测推翻的"直觉方案"**：我最初打算用"`smoother_rmse <= filter_rmse`（平滑不劣于滤波）"作为门禁。**实测该命题在 40/40 个 seed 上全部失败**，平滑后的逐点误差反而更大（seed 0：60 步中有 25 步 `es > ef`，最大比值 **1.80**；总平方误差 7.9702 > 7.8892）。

> **为什么**：平滑用**未来**观测换取更低方差，其误差相对"已实现的滤波误差"逐点可比性不被理论保证；且 DGP 的真值轨迹本身是随机实现，样本路径上的逐点比较会放大波动。**这是教科书级的错误直觉——若不实测就写进门禁，CI 会 100% 红灯。**

**最终门禁设计：确定性不变量（可精确断言）+ 统计量只进报告**

| 门禁 | 类型 | 断言 | 依据 |
|---|---|---|---|
| **G1 协方差恒 PSD** | 确定性 | `min(eigvalsh(P)) >= -1e-10` | 【已实测 40/40 通过】数学恒等式，无随机性 |
| **G2 平滑协方差不大于滤波** | 确定性 | `trace(P_smooth) <= trace(P_filter) + 1e-9` 逐步成立 | 【已实测 40/40 通过】Cauchy–Schwarz 保证的矩阵序，**与样本路径无关** |
| **G3 线性闭式解** | 精确解析 | 线性高斯 DGP 上 `K=0.5I, P=0.5I`（F=I,Q=0,H=I,R=I,P₀=I） | 【已实测】代数闭式，**零容差** |
| **G4 固定 seed 回归** | 确定性 | 内置 seed 的 benchmark 输出与签入的 `baseline.json` **逐位**相等 | 固定 seed + 钉死 numpy ⇒ 确定性 |
| **G5 不变量（不用于跨方法排名）** | 确定性 | `ESS ∈ (0, N]`；`0 <= ess_max <= ess_min <= N`；平滑输出形状匹配 | 结构性约束 |

**统计量如何进门禁（若必须）——配对比较（common random numbers）**：跨方法比较时，用**同一组 seed** 驱动所有方法，断言差值的符号而非绝对值。【已实测】配对差 `mean=463.80, sd=11.71`，30/30 个 seed 差值同号 → 门禁稳定。原理：配对消掉了 DGP 样本路径这个最大方差源。

**反面教材（已实测）**：`assert ll > mean - 0.5*sd` 这类"看起来合理"的门禁，在 30 个 seed 上失败 **9 次 = 30% 的运行**。**统计容差法在本域是错的解法。**

**结论与成本权衡**：
- 门禁（CI 必跑）：**G1–G5，全部确定性，零随机翻车**，秒级。
- 报告（不阻断）：跨 seed 的 `mean±std`、显著性检验（≥3 seed 时用 paired t-test，报 `p_value` 与 `significant: bool`）、失败案例归档。
- 代价：3 seed 下 `mean±std` 本身不稳（实测 k=3→1078.16, k=8→1084.51, k=30→1085.93，k≥8 才收敛）。故**报告明确标注 seed 数与置信区间**，不假装精确。

### 3.4 numpy 2.x API 移除

【已实测，numpy 2.5.3】以下成员**已从主命名空间移除**，用了就 `AttributeError`：

| 已移除 | 替代 | 在本域的典型触发点 |
|---|---|---|
| `np.float_` / `np.complex_` | `np.float64` / `np.complex128` | 粒子权重初始化 |
| `np.NaN` / `np.Inf` / `np.infty` / `np.NINF` / `np.PINF` | `np.nan` / `np.inf` | **不可用观测的占位符**（本域高频！） |
| `np.round_` / `np.product` / `np.cumproduct` | `np.round` / `np.prod` / `np.cumprod` | PF 权重归一化 |
| `np.alltrue` / `np.sometrue` | `np.all` / `np.any` | 重采样判据 |
| `np.asfarray` | `np.asarray(..., dtype=...)` | 观测数组构造 |
| `np.find_common_type` | `np.result_type` / `np.promote_types` | dtype 推断 |
| `np.mat` / `np.issctype` / `np.maximum_sctype` | `np.asarray` / `np.dtype` 判据 | 状态数组 |

> 注意 `np.int_` **未**被移除（实测 `OK`），可继续使用；但为一致性建议统一写 `np.int64`。

**防御措施**：CI 启用 `ruff` 的 **`NPY201`** 规则（专查 numpy 2 API 移除），零成本静态拦截。

### 3.5 Windows 编码与 CJK 表格对齐

**症状 A：中文表格错位**。【已实测】`len('粒子滤波') == 4`，但**显示宽度是 8**（东亚全角字符占 2 列）。用 `f"{name:<24}"` 对齐必然错位：
```
=== 宽度感知（正确） ===          === 朴素 len()（错位） ===
| 线性 Kalman 滤波         |       | 线性 Kalman 滤波             |
| 非线性 UKF 平滑          |       | 非线性 UKF 平滑               |
```
修法：用 `unicodedata.east_asian_width` 计算显示宽度后手动补空格。
```python
def _dw(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)
```

**症状 B：控制台编码**。本机实测 `sys.stdout.encoding = utf-8`、`locale.getpreferredencoding() = utf-8`（Python 3.13 在 Windows 已默认 UTF-8 模式），但**不能依赖**——`PYTHONIOENCODING` / 旧控制台 / 重定向到文件都可能改变它。修法：① 表格渲染统一走 `sys.stdout.reconfigure(encoding='utf-8', errors='replace')`；② 落盘 JSON 一律 `encoding='utf-8', ensure_ascii=False`；③ 所有源文件 UTF-8 无 BOM。

### 3.6 与 sklearn 指标别名同名导致递归

**症状**：`RecursionError`。根因：模块级 `from math import sqrt as rmse` 之后又 `def rmse(a, b): return rmse(a, b)`，函数体解析到全局同名符号 → 无限递归。【已实测复现 `RecursionError: maximum recursion depth exceeded`】。

sklearn 侧的风险放大了这一点：`sklearn.metrics` 里存在 `max_error`（遮蔽内建 `max`）等"看起来像通用词"的指标名；若用 `from sklearn.metrics import *` 或宽导入，很容易与本域自实现的 `rmse_state` / `nll` 撞名。

**修法（硬性）**：
1. **禁用 `import *`**，尤其 `from sklearn.metrics import *`（CI 加 ruff `F403` + `F405`）。
2. 本域指标**全部自实现**于 `inference/metrics.py`，只用 numpy，命名为 `rmse_state` / `mean_nll` / `ess_mean`——**不叫 `rmse`/`mse`/`max_error`**，从命名上杜绝撞名。
3. 若确需 sklearn，只用 `import sklearn.metrics as M` 后写 `M.mean_squared_error(...)`，永不 `from ... import <短名>`。
4. 加一条 pytest：对 metrics 模块做递归深度保护测试 `sys.setrecursionlimit` 下的 smoke test。

### 3.7 其他已识别坑

- **形状广播错误**：【已实测踩中】`F @ x`（列向量 `(n,1)`）与 `P @ F.T`（`(n,n)`）相加会广播成 `(n,n)`，静默产出错误结果。修法：**约定状态一律列向量 `(n,1)`，先验均值与先验协方差分开算**（`xp = F@x`；`Pp = F@P@F.T + Q`），并在 `core/numeric.py` 统一 `sym()`/`as_col()` 辅助函数。
- **x_true 泄漏**：`x_true` 只进 `data/` 生成，**绝不**传给任何 filter。pytest 加泄漏检测（对 filter 方法做 monkeypatch，断言未收到 `x_true`）。
- **`np.random.normal(loc, scale)` vs `multivariate_normal`**：前者签名易错（`scale` 是**标准差**），后者吃**协方差矩阵**。混用导致 Q 尺度差 `sqrt` 倍。统一在 `data/dgp.py` 封装。

---

## 4. CLI 与报告设计

### 4.1 `seqforge --help` 草案

```
seqforge — 序贯贝叶斯推断 / 状态空间估计工具箱

用法:
  seqforge <command> [options]

命令:
  demo                 端到端演示，落盘 benchmark.json（默认）
  generate             仅生成合成数据集
  filter               单方法跑单数据集
  smooth               平滑
  benchmark            完整基准测试（多 DGP × 多方法 × 多 seed）
  diagnostics          ESS / 轨迹图
  capabilities         列出可用后端与能力（探测层自检）
  info                 版本与环境信息

benchmark 选项:
  --dgps TEXT          逗号分隔：linear,nonlinear,sparse,degenerate  [默认全部]
  --methods TEXT       逗号分隔方法名；缺省=全部已注册
  --seeds INT          seed 数量 [默认 8]
  --particles INT      粒子数 [默认 2000]
  --steps INT          每条轨迹步数 T [默认 200]
  --out PATH           输出 JSON 路径 [默认 benchmark.json]
  --format TEXT        table | json | both [默认 both]
  --no-fail-fast       单个方法失败不中断整体
  --budget-sec FLOAT   时间软预算，超限则按降级策略自动缩减 [默认 55]
  --seed INT           主 seed（派生子流）[默认 20261003]

capabilities 选项:
  --probe-deps         实际尝试 import 并报告版本/失败原因

全局选项:
  -v/--verbose, -q/--quiet, --config PATH, --no-color
```

**`capabilities` 命令的输出示例**（这是"诚实降级"的用户可见出口）：

```
后端探测 (SeqForge capabilities)
  builtin           ok        numpy 2.5.3 / py 3.13.14
  statsmodels       ok        0.15.0
  pykalman          ok        0.11.2
  filterpy          unavailable  unmaintained (2018) · sdist-only · numpy2-incompatible
  particles         excluded    requires numpy<2, environment has 2.5.3

方法注册表 (10 methods)
  kalman   ok       smoothing=yes  ess=no
  ekf      ok       smoothing=yes  ess=no
  ...
```

### 4.2 `benchmark.json` schema

```jsonc
{
  "schema_version": "1.0",
  "generated_at": "2026-10-03T01:20:33+08:00",
  "env": { "python": "3.13.14", "numpy": "2.5.3", "scipy": "1.18.1",
           "platform": "win32", "bit_generator": "MT19937", "rng_class": "RandomState" },
  "config": { "seeds": 8, "particles": 2000, "steps": 200, "dgps": ["linear","nonlinear"] },
  "backends": [
    { "name": "statsmodels", "status": "ok", "version": "0.15.0" },
    { "name": "filterpy", "status": "unavailable",
      "reason": "unmaintained (last release 2018-10-10); sdist-only; numpy2 incompatible" }
  ],
  "results": [
    {
      "method": "BootstrapSIR", "family": "particle", "backend": "builtin",
      "dataset": "nonlinear_obs", "n_steps": 200,
      "seeds": [11, 12, 13],
      "log_lik":      { "mean": -93.67, "std": 3.55, "values": [...], "higher_is_better": true },
      "rmse_state":   { "mean": 0.311,  "std": 0.02, "values": [...], "higher_is_better": false },
      "smoother_rmse":{ "mean": 0.298,  "std": 0.01, "higher_is_better": false },
      "ess_mean":     { "mean": 634.2, "std": 12.7 },
      "runtime_sec":  { "mean": 0.069, "std": 0.004 },
      "ll_is_estimate": true,
      "status": "ok",
      "degraded_flags": []
    },
    {
      "method": "PykalmanKF", "backend": "pykalman", "dataset": "nonlinear_obs",
      "status": "skipped_inapplicable",
      "skip_reason": "linear model required; dataset is nonlinear_obs",
      "log_lik": null, "rmse_state": null
    }
  ],
  "pairwise": [
    { "dataset": "linear", "a": "BuiltinKF", "b": "StatsmodelsKF",
      "metric": "log_lik", "mean_diff": 1.2e-9, "p_value": 0.42,
      "significant": false, "test": "paired_ttest", "n_seeds": 8 }
  ],
  "invariants": [
    { "name": "cov_psd",        "gate": true,  "passed": true,  "detail": "min_eig=-3.2e-19" },
    { "name": "smooth_le_filter_cov", "gate": true, "passed": true },
    { "name": "linear_closed_form_k",  "gate": true, "passed": true, "detail": "K=0.500000" },
    { "name": "pf_beats_ukm_loglik",   "gate": false, "passed": true,
      "note": "non-gating: Monte-Carlo; reported only" }
  ],
  "failures": [
    { "method": "APF", "dataset": "degenerate", "seed": 13,
      "error_code": "E300", "error": "NumericalError: weight collapse, ESS=1.0 for 47 steps",
      "traceback_digest": "sha256:9f2c…" }
  ],
  "summary": { "n_ok": 42, "n_skipped": 6, "n_failed": 1, "total_runtime_sec": 12.4 }
}
```

设计要点：`status` 四态（`ok` / `skipped_unavailable` / `skipped_inapplicable` / `failed`）；跳过项**字段为 `null` 而非 0**；`invariants` 明确区分 `gate:true`（阻断）与 `gate:false`（仅报告）；`failures` 归因到 `error_code`（E100–E500），与 `core/errors.py` 的稳定错误码契约对齐。

---

## 5. 性能预算

### 5.1 实测基线（本机，纯 numpy，T=200）

| 方法 | 粒子数 | 单次耗时 | 说明 |
|---|---|---|---|
| PF | 200 | 78.3 ms | 首次含 JIT-less 预热 |
| PF | 500 | 45.2 ms | |
| PF | 1000 | 46.0 ms | |
| PF | 2000 | 69.1 ms | **推荐默认** |
| PF | 5000 | 102.4 ms | |
| PF | 10000 | 193.4 ms | 高精度档 |
| KF + RTS | — | **<1 ms**（T=2000 时约 25ms） | 解析法极快，非瓶颈 |

**全量 benchmark 实测外推**：5 DGP × 3 seeds × N=2000，纯 PF 部分 **0.60s**；加上 KF/EKF/UKF/IEKF + RTS/FFBS（约 +40%）**≈ 0.84s**。

### 5.2 结论：预算极度宽裕，建议提高质量而非降低

60s 预算 vs 实测 ~1s，**有 60 倍余量**。因此：

**推荐默认档（而非最小档）**：
- DGP：**6 个**（linear / nonlinear_obs / sparse_obs / degenerate_narrow / degenerate_bimodal / regime_switch）
- 方法：全 10 个 + Tier-0 基线
- seeds：**8 个**（实测 k=8 时 mean 才进入收敛区）
- 粒子数：**2000**
- 预估总耗时：**8–15s**，内存远低于 2GB（PF 粒子矩阵 `2000×6×8B ≈ 96KB`/步，FFBS 轨迹存储是主要项 `200×2000×6×8B ≈ 19MB`，完全可控）

### 5.3 降级策略（仅当超预算时触发，按此优先级）

**砍的顺序（从先砍到最后）**：

1. **先降粒子数**（2000 → 500 → 200）。理由：粒子数**只影响精度不影响覆盖**，且 PF 在线性高斯 DGP 上的结论不变；砍它损失最小、最不影响结构性结论。实测 N=500 仍 45ms/run。
2. **再砍 seeds**（8 → 4 → 3）。理由：seed 数影响 `mean±std` 的可信度，但**不影响任何单次运行**。降到 3 时必须在报告中标注 `"seed_count_insufficient_for_ci": true`，且 `invariants` 门禁仍然全跑（因为门禁是确定性的，与 seed 数无关）。
3. **最后砍 DGP**（6 → 4，优先砍 `degenerate_bimodal` 与 `regime_switch`）。理由：DGP 覆盖是**结论有效性的根基**，砍它等于缩小结论适用范围，损失最大。

**绝不砍**：`invariants` 门禁（确定性、秒级）、`capabilities` 探测、错误归因。

**预算软限机制**：`--budget-sec`（默认 55s）。pipeline 按"预估耗时 → 超限则按上述顺序降档 → 在 JSON 中记录 `degradation_applied`"执行，**并在 stdout 明示降了什么**，绝不静默降级。

---

## 6. 版本与依赖锁定

```
# Tier-1（必需，Tier-0 demo 零下载可跑）
numpy==2.5.3

# Tier-0（可选基线；缺失则标 skipped，功能不受影响）
scipy==1.18.1
statsmodels==0.15.0
pykalman==0.11.2
scikit-learn==1.9.1

# 开发/质量
pytest==8.x
ruff>=0.4.8        # 必须启用 NPY201 / F403 / F405
```

**锁定纪律**：
- 精确 `==`，不用 `>=`。因为 `RandomState.multivariate_normal` 依赖 LAPACK 分解路径，跨版本可能产出不同随机流（§3.2）。
- 生成并提交 `requirements.lock`（含 hash）。
- CI 显式安装 `numpy==2.5.3` 并断言 `np.__version__ == "2.5.3"`，**不信任** requirements 的解析结果。

---

## 7. 风险与降级预案（总表）

| 风险 | 概率 | 影响 | 预案 |
|---|---|---|---|
| `statsmodels`/`pykalman` 在他机装不上 | 中 | 低 | `available_*()` 探测 → `skipped_unavailable`，Tier-1 自研基线仍完整可跑 |
| PF 在 CI 上数字漂移 | **高** | 高 | 门禁只用确定性不变量（§3.3）；统计量仅入报告；钉死 numpy |
| 粒子退化导致 ESS 崩塌 | 中 | 中 | APF/RBPF 作为对策在报告中对比；ESS 曲线落盘；不隐藏退化 |
| CJK 表格错位 / 编码异常 | 中 | 低 | 宽度感知对齐 + `errors='replace'` |
| numpy 未来版本移除更多 API | 低 | 中 | ruff NPY201 静态拦截；CI 钉版本 |
| 内存超限（FFBS 轨迹） | 低 | 中 | FFBS 默认只存均值/协方差，轨迹存储可选 |

---

## 附录 A：核实方法与证据

所有 wheel 结论来自 **PyPI JSON API 实测**（`https://pypi.org/pypi/<pkg>/json`，2026-10-03），非记忆：

```
filterpy   1.4.5   files=1    仅 filterpy-1.4.5.zip         → sdist-only，从无 wheel
pykalman   0.11.2  py2.py3-none-any.whl + tar.gz           → 纯 Python wheel，可用
statsmodels 0.15.0 cp313-cp313-win_amd64.whl (11.3MB)       → 可用
scipy      1.18.1  cp313-cp313-win_amd64.whl (36.6MB)       → 可用
numpy      2.5.3   cp313-cp313-win_amd64.whl                → 可用
scikit-learn 1.9.1 cp313-cp313-win_amd64.whl               → 可用
numba      0.68.0  cp313-cp313-win_amd64.whl (2.8MB) + llvmlite 0.50.0 → 可用
torch      2.14.1  cp313-cp313-win_amd64.whl (124.1MB)     → 可用但体积不可接受
particles  0.4     requires numpy<2                          → 与 numpy 2.5.3 冲突
pyds       0.2.0   仅 .tar.bz2/.tar.gz                       → sdist-only
```

维护状态来自 WebSearch：filterpy health 26/100 "At Risk"（OpenSSF unmaintained，2026-09-17 报告）；pykalman 0.11.2 于 2026-01-31 发布，仓库 Dependabot 持续活跃；statsmodels 0.15.0 官方支持 Python 3.10–3.15。

**本机实测复现清单**：numpy 2 API 移除 19 项、filterpy mahalanobis `TypeError`、朴素 loglik 连乘下溢至 0.0、Cholesky jitter 救援失败阈值、PF 跨 seed 相对 std 3.79%、"均值−0.5σ" 门禁 30% 失败率、配对差 30/30 同号、平滑误差不变量 40/40 失败、协方差 PSD/迹序不变量 40/40 通过、跨进程 RNG 一致（`8.258392084277878`）、CJK `len()=4` vs 宽度 8、别名遮蔽 `RecursionError`。

## 附录 B：给实现者的接口速查

```python
# 方法注册（filters/__init__.py）
REGISTRY: dict[str, SequentialEstimator]

# 能力探测（core/capabilities.py）
available_statsmodels() -> BackendStatus
available_pykalman()    -> BackendStatus
available_filterpy()    -> BackendStatus   # 恒 False，带具体 reason

# 适配器（filters/baselines/）
StatsmodelsKalmanAdapter   # kalman_filter / kalman_smoother / mlemodel.fit
PykalmanAdapter           # KalmanFilter / UnscentedKalmanFilter / .sqrt

# 指标（inference/metrics.py，全部 numpy，禁止 import *）
rmse_state(x_hat, x_true) -> float      # 越小越好
mean_nll(loglik_terms)    -> float      # 越小越好（= -log_lik）
ess_mean(ess)             -> float      # 越大越好

# CI 门禁（tests/test_gates.py，全部确定性）
test_cov_psd              # G1
test_smooth_cov_le_filter # G2
test_linear_closed_form   # G3  零容差
test_fixed_seed_baseline  # G4  逐位比对签入 baseline.json
test_shape_and_range      # G5
```
