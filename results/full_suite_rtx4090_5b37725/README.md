# RTX 4090 完整测试归档：`5b37725`

2026-09-05，冻结源码 `5b377252cc4452b5ccc48c46ff1ae07a4e5e0e8a` 完成含 doctest 与 coverage 的完整 pytest 运行。退出码为 0，日志最终汇总为：

```text
591 passed, 2 skipped in 6036.09s (1:40:36)
```

归档时服务器 HEAD 与上述 SHA 一致，已跟踪文件的 diff 为空。[manifest.json](manifest.json) 保存完整命令、实际环境、退出状态、JUnit 汇总、跳过原因，以及压缩文件和解压原文各自的 SHA-256 与字节数。

这份证据仅属于上述冻结源码与本次 RTX 4090 运行。后续演示和默认 pass 测试改进另有 [39 项 CPU 回归记录](../interpreter_debug_replay_20260905/manifest.json)，不能把本次完整结果归于后续提交。RTX 4090 也不能替代 A100 官方验收；较早的 [SIGSEGV 记录](../rtx4090_compatibility_20260905/manifest.json)原因仍未确定，本轮成功没有证明 timeout 或 OOM 是其根因。

## 产物与一致性

| 产物 | 内容与汇总 |
|---|---|
| [pytest 原始日志](nine_full_ci_no_timeout_5b37725.log.gz) | 每项结果、最终统计和跳过说明；591 passed、2 skipped |
| [JUnit XML](nine_full_ci_no_timeout_5b37725.xml.gz) | 593 个测试条目、0 errors、0 failures、2 skipped；593 = 591 + 2 |
| [Coverage XML](nine_coverage_no_timeout_5b37725.xml.gz) | 全仓库 9178 条已覆盖行、10578 条有效行；XML 的 line-rate 为 0.8676，即 86.76% |

覆盖率属于该次全仓库运行，不是解释器专项覆盖率。XML 中 `branches-valid` 为 0，不能据此宣称分支覆盖完成。运行用时包含完整测试流程，不是解释器或 GPU 性能基准。

## 两个跳过项

| 测试 | 原始原因 |
|---|---|
| `tests.test_aot::test_add[True-45327-dtype0-bf16-cuda]` | `multi-device testing requires at least 2 devices` |
| `tests.test_built_artifact_reload::test_triton_aot_handle_is_reusable_across_cuda_contexts` | `Triton multi-context testing requires at least 2 CUDA devices` |

本机仅有一张 RTX 4090，因此没有完成上述跨设备测试。跳过不等于失败，也不等于这两个场景已经通过；准备双卡环境时仍需依照原测试的设备能力约束复验。

## 核对压缩文件与原文

下列 Python 标准库命令可从仓库根目录运行。它只在内存中解压，依次核对压缩字节数/散列和原文字节数/散列，不改写现有文件：

```bash
python - <<'PY'
import gzip
import hashlib
import json
from pathlib import Path

root = Path("results/full_suite_rtx4090_5b37725")
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
for original_name, record in manifest["artifacts"].items():
    compressed = (root / record["compressed_file"]).read_bytes()
    assert len(compressed) == record["compressed_bytes"], original_name
    assert hashlib.sha256(compressed).hexdigest() == record["compressed_sha256"], original_name
    original = gzip.decompress(compressed)
    assert len(original) == record["bytes"], original_name
    assert hashlib.sha256(original).hexdigest() == record["sha256"], original_name
    print("VERIFIED:", original_name)
PY
```

需要查看原文时，可用支持 gzip 的阅读工具解压到新的临时目录。避免自动转换换行、重新格式化 XML 或用新运行覆盖归档。manifest 中的绝对路径表示实际运行环境；迁移到其他机器时应调整路径并为新运行另建证据清单。
