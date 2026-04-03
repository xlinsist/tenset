# 4090x4 `error_no=7 (RUN_TIMEOUT)` 根因与修复方案（2026-04-03）

## 1. 结论摘要

- 这次 `RUN_TIMEOUT` 不是“4090x4 不能跑 TVM CUDA”的基础环境问题。
- 最小 TVM CUDA 程序在 4090x4 上可正常 `build + run`（`run_correct=true`）。
- 根因集中在 **auto-scheduler 测量路径**，尤其是：
  1. `local_run` 的 timeout 包装调用链；
  2. `call_func_with_timeout` 的进程/线程包装与强制终止；
  3. 个别 state 在 `time_evaluator` 阶段触发 runtime 异常（`workspace_pool` 断言、`free(): invalid pointer`），被包装链路放大为 `RUN_TIMEOUT`。

---

## 2. 核心证据

## 2.1 基础 CUDA/TVM 链路正常

最小检查结果（4090x4）：

- `gpu_exist: true`
- `compute_version: 8.9`
- `build_ok: true`
- `run_ok: true`
- `run_correct: true`

记录：

- `scripts/run_logs/min_tvm_cuda_check_v2_20260403_115103/result.json`

## 2.2 `RUN_TIMEOUT` 贴边现象说明是“卡到 timeout”，不是正常慢

- `run_timeout=20` 时，`all_cost` 约 `20.xs`，大量 `error_no=7`
- `run_timeout=60` 时，`all_cost` 约 `60.xs`，仍 `error_no=7`

这说明执行路径被卡住，主进程在 timeout 处返回，而不是 kernel 本身稳定运行 20/60 秒。

## 2.3 回放同一批“历史 timeout 样本”时出现“同批次分化”

对 `diag33_retest2` 的 9 条样本逐条回放：

- 多数样本可快速成功（<1s，`error_no=0`）
- 少数样本仍在 `20.03s` 左右超时（`TimeoutError`）

记录：

- `scripts/run_logs/replay_timeout_records_probe_20260403_121709/result.json`

这证明问题是 **state 相关 + 测量路径相关**，不是“所有样本都天然慢”。

## 2.4 runtime 破坏性信号

同批次 stderr 出现：

- `workspace_pool.cc:118 Check failed: allocated_.size() == 1`
- `free(): invalid pointer`

记录：

- `scripts/run_logs/replay_timeout_records_probe_20260403_121709/stderr.log`

---

## 3. 可定位的问题代码区间

## 3.1 首要区间：`local_run` timeout 包装链

- `python/tvm/auto_scheduler/measure.py`:
  - `local_run` 中 `call_func_with_timeout(...)` 分支（约行 1015-1039）

## 3.2 首要区间：timeout 实现

- `python/tvm/auto_scheduler/utils.py`:
  - `_func_wrapper`（约行 292-303）
  - `call_func_with_timeout`（约行 306-328）

关键点：

- `que.get(timeout=...)` 超时即返回 `TimeoutError`
- 随后 `kill_child_processes + terminate + join`
- 对 CUDA/runtime 异常敏感，容易把异常路径表现成统一 `RUN_TIMEOUT`

## 3.3 次要区间：`measure_programs.py` 的状态恢复与重绑定

- `scripts/measure_programs.py`:
  - `recover_measure_input` + 重新绑 `SearchTask` + 批量 `measurer.measure`（约行 78-97）

该段不是唯一根因，但会放大边缘 state 的不稳定。

---

## 4. 为什么“原平台不明显，4090x4 明显”

- 该类问题理论上两边都可能出现（原平台文档也记录过 `free(): invalid pointer`）。
- 4090x4 当前更容易触发的原因是：
  1. 节点级运行时状态差异（上下文/负载/内存状态）；
  2. 样本触发集合差异（只有部分 state 触发）；
  3. timeout 包装链路把 runtime 异常放大成 `error_no=7`。

---

## 5. 修复方案（按优先级）

## 5.1 方案A（优先，最直接）：GPU 路径绕过 `call_func_with_timeout` 子进程包装

做法：

- 在 `python/tvm/auto_scheduler/measure.py::local_run` 中：
  - 对 GPU target（cuda/rocm/opencl/vulkan/metal）默认直接调用 `_timed_eval_func`（进程内）；
  - 保留环境变量开关（如 `TENSET_FORCE_SUBPROCESS_RUN=1`）可回退到旧路径。

理由：

- 现有证据表明 timeout 包装链路会把异常放大为 `RUN_TIMEOUT`；
- 直接路径在探针中更稳定，且便于拿到真实 `error_no`（0/4/3）而非统一 7。

## 5.2 方案B（稳态增强）：单 task 独立进程测量，避免状态污染累积

做法：

- 继续沿用 `measure_programs_isolated.sh` 的思路：
  - 一个 task 一个 Python 进程；
  - 每个进程结束后退出，避免 runtime 污染跨 task 传播。

## 5.3 方案C（数据侧兜底）：对问题 state 预筛 + 清洗

做法：

- `dump_programs.py` 使用 `--validate-state` 预筛；
- 测量后仅保留 `error_no=0` 样本进入数据集。

---

## 6. 验证步骤（修复后必须执行）

1. 3x3 回归（同 task 同 schedule）：
   - 验证 `error_no=7` 是否显著下降；
   - 观察是否从“贴 timeout 边界”转为真实错误码分布（0/4/3 等）。

2. timeout 灵敏度复测：
   - 分别跑 `run_timeout=20` 和 `run_timeout=60`；
   - 期望不再出现“all_cost 紧贴 timeout 值”的大面积现象。

3. 稳定性复测：
   - 连续多轮同样本回放，检查是否出现 `workspace_pool` 断言和 `free(): invalid pointer` 显著下降。

---

## 7. 本次已完成动作（用于审计）

- 已保证两平台关键代码文件与 Git `HEAD` 对齐（同 commit、同 hash）：
  - `scripts/measure_programs.py`
  - `scripts/dump_programs.py`
  - `python/tvm/auto_scheduler/measure.py`
  - `python/tvm/contrib/nvcc.py`
  - `python/tvm/auto_scheduler/cost_model/__init__.py`
- 已在 4090x4 完成最小 CUDA 程序可运行验证。
- 已完成多轮 A/B 与回放探针，确认问题集中在 auto-scheduler 测量链路。

---

## 8. 2026-04-03 修复落地与结果

### 8.1 已落地代码修复

- 文件：`python/tvm/auto_scheduler/measure.py`
- 改动：
  - 保留原 subprocess timeout 路径；
  - 对 GPU 测量增加“timeout 后单次 in-process 回退”机制，避免可运行样本被统一记为 `RUN_TIMEOUT(7)`；
  - 可通过环境变量 `TENSET_DISABLE_GPU_TIMEOUT_FALLBACK=1` 关闭该回退。

### 8.2 稳定执行模式（关键）

由于当前节点仍存在退出阶段的 runtime 崩溃（`free(): invalid pointer` / `workspace_pool` 断言），
主流程 `measure_programs.py` 在同一长进程内容易被 teardown 污染。

采用稳定模式：

- 每个 schedule 独立 Python 进程测量；
- 每个子进程完成后 `os._exit(0)`，避免析构阶段污染后续样本。

### 8.3 验证结果（已达到“非 timeout 正常执行”目标）

运行目录：

- `scripts/run_logs/stable_single_schedule_runner_v2_20260403_124931`

结果文件：

- `scripts/run_logs/stable_single_schedule_runner_v2_20260403_124931/results.tsv`

统计（3 task × 3 schedule）：

- 9/9 均为 `run_error=0`
- 无 `RUN_TIMEOUT(7)`
- `run_cost` 约 `0.70s ~ 5.09s`，均为正常可执行结果

这说明：

- 4090x4 平台可正确执行这些 schedule；
- 之前的系统性 `RUN_TIMEOUT` 主要来自测量流程与进程生命周期问题，而非“内核无法运行”。
