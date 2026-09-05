# A100 squeeze fixture 受控机制诊断：`377daec`

这份记录包含实际 A100-SXM4-40GB 上的三个受控 GPU 场景，**其中一个场景预期触发原断言失败**。它不是一次正式 pytest 运行，不增加正式测试通过数，也不是对此前全量失败现场的精确回放。

- [原始 JSON](squeeze-fixture-diagnosis-20260906.json)：按原字节保存，包含输入索引、观察指标、预期失败堆栈和运行前后源码状态。
- [诊断脚本副本](a100_diagnose_squeeze_fixture.py.txt)：按原字节保存，以 `.py.txt` 提供阅读；没有在归档过程中执行。
- [archive.json](archive.json)：本目录的 POSIX 路径、大小及 SHA-256 索引，包含源码与结果范围说明。

JSON 记录时间为 2026-09-05 18:22:55 UTC，即北京时间 2026-09-06 02:22:55；设备为 NVIDIA A100-SXM4-40GB，PyTorch 为 `2.5.0+cu124`。前后 HEAD 均为 `377daec6242864a920de43a55523ac3d5f582648`，已跟踪文件无修改。JSON 中 `tests/test_generation.py` 的 SHA-256 已与该提交的 Git blob 独立核对一致。

## 执行方式与结果

脚本在每个场景手动设置 seed 2026，直接调用原 `test_squeezing_the_innermost_level(1024, 128, 10, "cuda")`。它在进程内暂时替换 `torch.empty`、`torch.randint`、`torch.allclose` 的入口以观察或控制指定输入，随后在 `finally` 中恢复；正式测试文件、原 kernel 调用和断言没有修改。

| 场景 | 输出初值与索引 | 原始 allclose | fixture 结果 | bitwise / finite mismatch | 输出 NaN：已写区 / 未写区 |
|---|---|---|---|---|---|
| `natural_allocation` | 自然分配；本次随机索引恰有 10 个不同值 | True | PASS | 0 / 0 | 0 / 0 |
| `nan_output_unique_indices` | 输出预填 NaN；索引固定为 0–9 | False | FAIL，AssertionError | 0 / 0 | 0 / 129792 |
| `finite_output_unique_indices` | 输出预填 -123.0；索引固定为 0–9 | True | PASS | 0 / 0 | 0 / 0 |

每个场景记录一次指定形状的输出分配和一次完整矩阵比较。NaN 场景的 output 与 expected 都含 129,792 个 NaN，即 `(1024 - 10) × 128`，全部位于未写行；已写行 NaN 数为 0。

`bitwise_mismatches` 比较全部 float32 元素的 int32 位模式；`finite_mismatches` 仅比较双方均为有限值的位置。NaN 场景两种 mismatch 都为 0，但原始 `torch.allclose` 返回 False，原断言因此仍失败。

三个场景中的 `equal_nan_diagnostic_only` 均为 True。这是额外观察指标：包装入口返回的始终是**原始 allclose 的结果**，没有将正式断言改成 `equal_nan=True`，也没有放宽容差来让 NaN 场景通过。

## 可以与不可以据此得出的结论

这组控制说明，在当前固定形状与 seed 下，未写区域含 NaN 时，即使 output 与 expected 逐位相同，原始 allclose 仍会失败；给未写区域有限初值并使用唯一目标行时，本次控制调用通过。

此前 [377daec 全量记录](../full_suite_a100_377daec/README.md) 仍为 **1 failed、599 passed、2 skipped，exit 1**。旧现场没有保存足以重放分配器状态、实际 pytest 随机状态或全部索引的证据；本诊断没有重建这些状态。**不能据此断言旧全量失败一定由 NaN、重复行索引或某个唯一原因造成，也不能把自然场景 PASS 说成原失败已经精确复现或消失。**本次控制使用的索引均唯一，没有测试故意重复目标行的场景。

两个 PASS 和一个预期 FAIL 仅属于机制诊断，不与正式 pytest、GPU 差分或专项计数相加。后续测试稳定化及正式回归应以新的源码 SHA、完整命令和独立结果目录记录。

## 文件使用与核验

可用 Python 标准库读取 `archive.json`，逐项检查 `artifacts` 中的文件大小和 SHA-256；无需调用项目代码或 GPU。原始 JSON 和脚本副本均未改写。

脚本依赖当时服务器的项目目录、已失败 full 的状态文件及新的输出路径，并会检查源码状态；它是实验方法的归档，不是脱离环境即可运行的旧现场回放器。若要开展新诊断，应另建实验目录并保留自己的输入、环境和结果，避免覆盖本证据。
