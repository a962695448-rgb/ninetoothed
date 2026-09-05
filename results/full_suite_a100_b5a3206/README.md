# A100 完整测试失败归档：`b5a3206`

冻结源码 `b5a3206f8351e5a138d16ee13f6d6ef9c620044b` 于北京时间 2026-09-06 在实际 NVIDIA A100-SXM4-40GB 上完成完整 pytest（含 doctest 和 coverage）。**本轮失败，进程退出码 1；不是全量通过或官方验收通过。**

```text
16 failed, 584 passed, 2 skipped in 2737.65s (0:45:37)
```

JUnit 共 602 条目、16 failures、0 errors、2 skipped；602 = 584 + 16 + 2。原 runner 在执行前后均记录上述 HEAD，已跟踪文件无修改。pytest 汇总用时 2737.65 秒，外部 validation 进程计时 2740.628 秒，包含的启动开销不同，不能视为性能基准。

[manifest.json](manifest.json) 保存原始命令、环境、每个失败和跳过项、覆盖率、压缩包及所有原文件的字节数和 SHA-256。[raw-full.tar.gz](raw-full.tar.gz) 完整保留下载目录的 88 个文件，解压后位于 `full/`，包括原始 runner manifest、stdout/stderr、JUnit、coverage XML 和 71 个 coverage HTML 相关文件。只规范化 tar/gzip 容器元数据，原文件内容没有改写、过滤或重新排版。

## 16 个失败的分类

| 数量 | 测试参数范围 | 观察到的失败位置 |
|---|---|---|
| 8 | `test_to_padded_tensor`、`test_expand`；`jagged_dim=1`；batch 数为 2、3、7、16 | PyTorch 2.5 的参考转换调用抛出 `NotImplementedError: aten.to_padded_tensor.default` |
| 8 | 相同两个测试；`jagged_dim=2`；batch 数为 2、3、7、16 | `torch.nested.nested_tensor(..., layout=torch.jagged)` 列表工厂构造输入时抛出 RuntimeError：该工厂仅允许第二维为 ragged |

这些是日志中实际出现的参考转换与输入构造问题，不是数值断言失败。dim 2 在创建输入时已失败；dim 1 尚未完成参考值比较。因此不能据此宣称 jagged 内核已正确。后续修复与回归应以新的源码 SHA 和新目录另行记录，不能覆盖本次失败。

## 两个跳过项与覆盖率

- `tests.test_aot::test_add[True-45327-dtype0-bf16-cuda]`：`multi-device testing requires at least 2 devices`。
- `tests.test_built_artifact_reload::test_triton_aot_handle_is_reusable_across_cuda_contexts`：`Triton multi-context testing requires at least 2 CUDA devices`。

单卡 A100 没有验证这两个跨设备场景。Coverage XML 为全库 9180 / 10578 行，`line-rate=0.8678`（86.78%）；这不是解释器专项覆盖率，也不是测试成功证明。`branches-valid=0`，不宣称分支覆盖完成。

此前 [A100 已完成专项](../a100_20260906/README.md) 的 14 项 GPU 差分与 180 项专项和本轮重叠，不能累计。此前 [4090 完整测试](../full_suite_rtx4090_5b37725/README.md) 属于另一源码和环境，不能替代本轮结果。原始日志中的运行目录用于溯源，迁移机器时应调整实际路径并另建运行目录。

## 只读核验压缩包与原文

从仓库根目录运行下面的 Python 标准库命令。它只在内存中解压、验证，不调用项目代码或 GPU，也不写文件：

```bash
python - <<'PY'
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

root = Path("results/full_suite_a100_b5a3206")
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
record = manifest["archive"]
compressed = (root / record["path"]).read_bytes()
assert len(compressed) == record["bytes"]
assert hashlib.sha256(compressed).hexdigest() == record["sha256"]
payload = gzip.decompress(compressed)
assert len(payload) == record["uncompressed_tar_bytes"]
assert hashlib.sha256(payload).hexdigest() == record["uncompressed_tar_sha256"]
with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
    members = archive.getmembers()
    assert len(members) == record["file_count"]
    assert {member.name for member in members} == set(manifest["artifacts"])
    for member in members:
        assert member.isfile()
        original = archive.extractfile(member).read()
        expected = manifest["artifacts"][member.name]
        assert len(original) == expected["bytes"], member.name
        assert hashlib.sha256(original).hexdigest() == expected["sha256"], member.name
print("VERIFIED: 88 original files; preserved test outcome is FAIL, exit 1")
PY
```

需要浏览 coverage HTML 时，将压缩包解压到新的独立目录并打开 `full/coverage-html/index.html`。保留原始压缩包和 manifest，不以新运行覆盖归档。
