# Changelog

本文件记录 SeqForge 的所有重要变更。
格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [0.1.0] - 2026-10-04

首个版本。域：**序贯贝叶斯推断 / 状态空间估计**。

### 新增 · 核心算法

- **解析滤波族**（`filters/kalman.py`）
  - `KalmanFilter`：Joseph 形式协方差更新，Joseph 形式保PSD
  - `ExtendedKalmanFilter`：一阶泰勒线性化
  - `IEKFFilter`：迭代 EKF（Li & Kanagaraj 2017 式(8) 的重线性化修正）
  - `UnscentedKalmanFilter`：sigma 点传播（α=1, β=2, κ=0）
  - `rts_smooth`：Rauch-Tung-Striebel 固定区间平滑器
- **粒子滤波族**（`filters/particle.py`）
  - `BootstrapSIR`：Gordon-Salmond-Smith 1993，系统/多项式重采样
  - `AuxiliaryParticleFilter`：Pitt & Shephard 1999，**逐祖先最优提议**
- **粒子平滑**（`smoothers/particle_smoother.py`）
  - `FFBS`：Godsill-Doucet-Rajam 1994，O(T·N) 的后向采样
- **Tier-0 参照后端**（`filters/backends.py`）
  - `statsmodels` / `pykalman` 适配器 + `available_*()` 懒探测

### 新增 · 数据与评测

- 4 个合成 DGP（`data/dgp.py`）：`linear_gaussian` / `nonlinear_obs` /
  `sparse_obs` / `degenerate`，各暴露一类方法缺陷
- 统一指标层（`eval/metrics.py`）：双 RMSE 口径（`rmse_state` / `rmse_obs`）、
  mean ± std 聚合、显著性标记、**从实测结果派生**的失败案例
- 主流水线（`pipeline/runner.py`）与 CLI（`cli.py`，4 个子命令）
- 端到端 demo（`examples/run_demo.py`），落盘 `benchmark.json`

### 新增 · 质量资产

- **22 条可验证不变量**（`tests/test_invariants.py`）
  - 交叉验证：KF log-lik vs 独立多元高斯闭式解（Δ < 1e-13）
  - 代数恒等式：UKF/EKF/IEKF 线性退化（2.7e-15 / 0 / 5.6e-17）、sigma 点矩匹配
  - 理论性质：bootstrap 向下有偏、RTS 协方差 Loewner 序、ESS 边界
- 8 条后端与降级测试（`tests/test_backends.py`），含
  「强制关闭所有后端仍能跑完」的离线兜底验证
- 19 条 CLI / 配置 / 确定性测试（`tests/test_cli.py`）
- CI：`ubuntu + windows` × `py3.12 + py3.13` 矩阵，lint 为**硬门禁**（无 `|| true`）

### 实测结论

- **APF 的 ESS 是 bootstrap 的 2.00×**（612.8 vs 305.7，N=1000）——
  最优提议的核心价值，可复现的确定性量
- bootstrap 边缘似然对 KF 精确值的偏差随 N 单调收缩（−0.348 → −0.329 /step）
- 同seed 两次运行，72 行 benchmark 数据**逐位一致**

### 修复的 9 个真实 bug（均为实测发现，非预防性）

| # | 症状 | 根因 |
|---|------|------|
| 1 | UKF 协方差广播维度不匹配 | `dot` 误用于逐样本加权求和 |
| 2 | 秩亏协方差 `eigh` 不收敛 | 改用 SVD（对秩亏矩阵恒收敛） |
| 3 | 稀疏观测产生 NaN 污染全流程 | 未按维度跳过缺失观测 |
| 4 | APF 权重系统性偏高（偏差 +828） | 缺提议密度修正项 `log q` |
| 5 | APF log-lik 随 N 发散 | 边缘似然与后验权重口径混淆 |
| 6 | APF ESS 崩塌到 0.1% | 提议协方差用了固定名义 `P_pred` |
| 7 | IEKF 协方差溢出到 inf | `I_KH` 恒为 I，等于没有协方差更新 |
| 8 | IEKF 在线性问题上不收敛（\|Δx\|=58.3） | 重线性化项作用点写反 |
| 9 | radar 观测下 EKF/IEKF 协方差溢出 | `1/r` 奇异性未设下限 |

### 已知限制（详见 `docs/model_card.md` §5）

- 主指标在主域上**数学不可超越** KF（KF 是精确贝叶斯最优）
- 稀疏观测域 APF 的 ESS 低于 SIR（真实发现，非 bug）
- 消融显示「关闭辅助重采样」略优（负结果已保留）
- `statsmodels` 0.15 移除 `initial_state`，无法对齐初值口径

[0.1.0]: https://github.com/CJX0712/seqforge/releases/tag/v0.1.0
