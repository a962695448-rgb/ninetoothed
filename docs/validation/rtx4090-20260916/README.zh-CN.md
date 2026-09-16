# 当前九齿提交的 RTX 4090 验证记录

日期：2026-09-16。受测提交：`1b68040f73553de9b5929cd61067c4d4369f7f22`，已包含上游 `22e74c3` 的目标架构变更。

**按测试 ID 核对的最终结果：900 passed，2 skipped，完整覆盖同一环境收集的 902 项。** 这是分段执行后的完整测试集合覆盖，不是一次未中断的全仓库运行。15 项真实 Triton GPU 差分也通过，属于上述集合的子集，不额外累加。

## 版本与环境

- NVIDIA GeForce RTX 4090，sm89；驱动 570.124.06。
- Intel Xeon Platinum 8358P；Linux x86_64；Python 3.12.3。
- PyTorch `2.6.0a0+ecf3bae40a.nv25.01`；CUDA/NVCC 12.8/12.8.61。
- Triton 模块版本 3.1.0，发行包为 NVIDIA 的 `pytorch-triton 3.1.0+cf34004b8.internal`。
- NumPy 1.26.4、SymPy 1.13.1、pytest 8.1.1、pytest-cov 7.1.0、pytest-xdist 3.6.1、TileLang 0.1.14。

完整包版本见 [environment.json](environment.json)。源代码按 Git blob 清单核对；普通库代码、测试和断言在各阶段未变更。每阶段的源码摘要、硬件、环境与执行命令保存在对应 `run-summary.json`。

## 阶段记录

| 阶段 | 实际结果 | 说明 |
|---|---|---|
| 真实 GPU 差分 | 15/15 通过 | 9 个程序、10 类行为；浮点 rtol/atol 均为 0.001，整数/布尔精确比较 |
| 首次完整运行 | 30 分钟超时，退出 -15 | 日志保留，不计为全量通过 |
| 两个 pytest 进程 | 中断时 898 passed、2 skipped | 保留完整 JUnit；额外一个无名称的 setup 记录不计为通过 |
| 精确收集对照 | 902 项，恰有 2 项未完成 | 按 pytest 的测试地址规则匹配 JUnit，没有把其他测试替换进来 |
| 并行诊断补跑 | 两个 worker 退出 | 发生在诊断输出期间，尚未完成断言；原始失败记录保留，不将其抹除 |
| 逐项补跑 | 2 passed，1222.67 秒 | 未修改参数、精度断言或 `max_num_configs=50`；采用独立文件记录调用栈 |
| 完整集合核对 | 900 passed、2 skipped | 两阶段 ID 不重叠、无遗漏，源码和环境一致 |

两项条件跳过均需要至少两张 CUDA GPU，本轮仅一张。未完成后补齐的用例是 `tests/test_conv2d.py` 中 padding=(0,1) 与 padding=(2,0) 的 FP16 卷积；输入 N/C/H/W=4/64/16/16，输出通道 512，卷积核 3×3，rtol/atol=0.001。

采样观察到的耗时位于 Triton 编译流程。编译/测试时长不是 GPU 算子性能，不能与旧 A100 回归时长相除得到加速比。两个 worker 退出的底层原因未作未经验证的归因；后续独立日志、单进程补跑在同一源码和依赖下通过。

## 核查入口

- [900 个已完成项目的原始 JUnit](recorded/nine-parallel/full-pytest.xml)
- [两项补跑的原始 JUnit](recorded/nine-remainder-serial/remaining-pytest.xml)
- [完整收集清单与未完成项](recorded/nine-completion/remaining.json)
- [逐项核对结果](recorded/nine-completion/combined-results.json)
- [15 项真实 GPU 差分](recorded/nine-targeted/triton-differential.json)
- [中断的首次运行](recorded/nine-full/run-summary.json)
- [诊断补跑的 worker 退出记录](recorded/nine-remainder/remaining-pytest.log)

本目录保存原始文本记录。原执行脚本以 `.py.txt` 保存于 `tools/`，便于审查其原始字节且不参与仓库测试发现。`SOURCE_TREES.json` 与 `SOURCE_ARCHIVES.json` 来自同时验证两个项目的共享输入包；九齿检查只读取其中 `nine` 条目。

无需 GPU 即可重新核对完整测试集合：

```bash
python verify_reports.py --results recorded --output /tmp/nine-rtx4090-new-verification.json
```

输出必须为 `PASS_COMPLETE_COLLECTION`、902 项、900 passed、2 skipped。核对工具拒绝遗漏、重复、失败和源码/环境不一致；无名称的中断残留条目单独列出。

## 硬件复现命令

检出上述受测提交并准备匹配依赖后，在仓库根目录运行：

```bash
python scripts/verify_interpreter_gpu.py --device 0 --report /tmp/nine-gpu-new.json
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest --doctest-modules \
  -p pytest_cov.plugin --junitxml=/tmp/nine-full-new.xml \
  --cov=ninetoothed --cov-report=xml:/tmp/nine-coverage-new.xml
```

本次分段执行的具体命令以原始 `run-summary.json` 为准。多次执行同一差分入口不累计为更多独立用例；补跑阶段的覆盖率文件也不冒充完整仓库覆盖率。

## 结论边界

这些结果补充了合入新上游架构后的 NVIDIA 实机证据。它们仅属于 RTX 4090，不是新 A100 结果。历史 A100 835 passed、2 skipped 仍归属于旧计算源码 `ed33273`。CPU 顺序解释不模拟 GPU 竞争，本次验证没有扩大解释器的操作、dtype 或访存支持范围，也不代表上游 PR 已合并。
