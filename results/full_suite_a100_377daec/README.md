# A100 完整复验归档：`377daec`

冻结源码 `377daec6242864a920de43a55523ac3d5f582648`；本轮完整测试失败，不能声明全量通过，pytest 退出码 1。此结论仅属于本次实际 A100 运行，不代表官方整体验收、上游合并或优秀评选完成。

```text
1 failed, 599 passed, 2 skipped in 752.11s (0:12:32)
```

JUnit 共 602 项，599 passed、1 failed、0 errors、2 skipped。实际日志与 JUnit 的统计、失败名称及用时均已核对；运行前后源码为上述 SHA，已跟踪文件无修改。

[manifest.json](manifest.json) 保存原始命令、环境、全部失败/错误/skip、覆盖率以及压缩前后 SHA-256 索引。[raw-full.tar.gz](raw-full.tar.gz) 内的 `full/` 完整保存 88 个原文件，含全部 coverage HTML 和原 runner manifest。原文件字节未改写，仅规范化压缩容器元数据。

## 失败、错误与跳过

唯一失败为 `tests/test_generation.py::test_squeezing_the_innermost_level[1024-128-10-cuda]`：双方输出形状均为 `[1024, 128]`，但 `torch.allclose(output, expected)` 为 False。这里仅记录断言失败，**不推断根因**；完整错误消息与堆栈保留在 manifest 和压缩包内原始 JUnit/stdout。

本轮 `tests/test_jagged.py` 的 16 项均 PASS，覆盖两个函数、`jagged_dim=1/2` 和 batch 数 `2/3/7/16`。该定向修复通过不改变本轮全量 FAIL 的结论。

两个跳过项：

- `tests.test_aot::test_add[True-45327-dtype0-bf16-cuda]`：`multi-device testing requires at least 2 devices`。
- `tests.test_built_artifact_reload::test_triton_aot_handle_is_reusable_across_cuda_contexts`：`Triton multi-context testing requires at least 2 CUDA devices`。

单卡环境没有验证这两个跨设备场景。任何后续修复需以新源码与新目录保留证据，不能覆盖本记录。

## 覆盖率与原文核验

Coverage XML 记录全库 9144 / 10578 行，`line-rate=0.8644`；不是解释器专项覆盖率，也不是测试成功或分支覆盖的替代证明。`branches-valid=0`。pytest 用时 752.11 秒，外部进程用时 754.99 秒，不能视为 GPU 性能基准。

可按 manifest 的 `archive` 字段核对压缩字节及解压 tar 的大小/SHA-256，再按 `artifacts` 的 POSIX 路径逐个核对成员原文字节；无需调用项目代码或 GPU。需浏览原文时解压到新目录，勿覆盖归档。

此前 [b5 A100 全量失败](../full_suite_a100_b5a3206/README.md) 与 [已完成 A100 专项](../a100_20260906/README.md) 保持原文；不同运行与重叠测试不能累计或互相改写结果。
