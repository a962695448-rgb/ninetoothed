# GPU 报告生命周期修复与验证（2026-09-16 晚间）

基于 PR 提交 `788208c91f3a7d2aa72dde315201a61380f49c00`，仅修改 GPU 验证报告脚本并增加九项报告协议测试。解释器、SSA、编译器和数值测试实现保持不变。

## 问题与行为

旧脚本在 GPU 不可用时也会覆盖已有 PASS 文件。新版本在初始化 GPU 前独占创建输出，拒绝已有文件、目录、普通/悬空符号链接；每个用例启动前写进度。用例中断保存已完成记录、当前用例名和 INTERRUPTED 状态，退出 130。数值失败继续收集后续结果，最终 FAIL；环境不可用为 UNVERIFIED。重新运行必须提供新的报告路径。

初始回归在旧代码上为 5 failed、3 passed、1 deselected（中断用例暂不执行，避免旧脚本让 pytest 整体中断）；修复后九项通过。生命周期测试使用明确的测试替身，只验证报告行为，不能代替 GPU 数值检查。

## 验证范围

- 无 Torch/Triton 的本地 CPU 环境：502 passed、15 actual GPU deselected，43.91 秒；新增九项包含在 502 内。
- 最终脚本在 RTX 4090 D 实机：15/15 真实 Triton GPU 差分 PASS；最终九项报告协议测试 9 passed in 0.35s。
- 最终本地九项报告测试也通过，0.37 秒。Ruff、format、贡献风格检查通过。
- 本次不重复运行未变更的完整卷积集合。旧 1b68040 的 900 passed、2 skipped 仍是先前固定档案的范围，不把它标成当前新增测试后的全仓库统计。

`nine-final-summary.json` 包含最终受测源码哈希及命令，`nine-final-gpu.json` 是最终 GPU 报告。`nine-final-report-tests.xml` 是最终协议测试原始 JUnit；`nine-cpu.junit.xml` 为 CPU 原始 JUnit。

首次实机运行通过后，按项目格式要求补空行和测试错误消息标点；最终版本再次实机验证。`initial-source`、`nine-gpu.*`、`nine-report-tests.*` 保留首次记录；`final-source` 为最终文件。报告脚本两版 AST 相同。两轮不是更多独立 GPU 用例。

`summary.json` 是同一服务器的首轮任务清单，其中也记录 CUDA 任务退出码；CUDA 120 组检查的原始证据单独归档于 Learning-CUDA。运行器保存于 `run_focused_gpu.py` 和 `run_final_nine.sh`，路径反映实际运行位置。

## 复现最终检查

在匹配 NGC PyTorch 2.6 / CUDA 12.8 的 GPU 环境检出上述 NineToothed 提交，将本目录 `final-source/` 下的两个文件按相同相对路径覆盖到工作副本，然后从仓库根目录运行：

```bash
python scripts/verify_interpreter_gpu.py --report /tmp/new-gpu-report.json
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_interpreter_gpu_report.py
```

CPU 独立验证另建不含 Torch/Triton 的环境，安装仓库 `requirements-cpu.txt` 后运行 `python scripts/run_cpu_tests.py`。完整版本见环境记录；不升级依赖来冒充原始环境。

正常中断得到 INTERRUPTED；SIGKILL、断电和写盘故障不能保证 JSON 完整性。未完成、损坏或非 PASS 文件均不得视为验收通过。本轮只做正确性与证据保护，不宣称性能提升。
