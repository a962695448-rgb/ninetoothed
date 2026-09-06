# A100 完整复验归档：`82592b8`

冻结源码 `82592b8f6de65052e4258fdd6067956d4ede18c3`；本轮完整测试通过（仍需分别解释跳过项），pytest 退出码 0。此结论仅属于本次实际 A100 运行，不代表官方整体验收或上游合并完成。

```text
600 passed, 2 skipped in 450.25s (0:07:30)
```

JUnit 共 602 项，600 passed、0 failed、0 errors、2 skipped。实际日志与 JUnit 的统计、失败名称及用时均已核对；运行前后源码为上述 SHA，已跟踪文件无修改。

[manifest.json](manifest.json) 保存原始命令、环境、全部失败/错误/skip、覆盖率以及压缩前后 SHA-256 索引。[raw-full.tar.gz](raw-full.tar.gz) 内的 `full/` 完整保存 88 个原文件，含全部 coverage HTML 和原 runner manifest。原文件字节未改写，仅规范化压缩容器元数据。

## 失败、错误与跳过

- `tests.test_aot::test_add[True-45327-dtype0-bf16-cuda]`：multi-device testing requires at least 2 devices
- `tests.test_built_artifact_reload::test_triton_aot_handle_is_reusable_across_cuda_contexts`：Triton multi-context testing requires at least 2 CUDA devices

失败与跳过均保留，不能把未执行场景标成通过。任何后续修复需另建证据，不能覆盖本记录。

## 覆盖率与原文核验

Coverage XML 记录全库 9087 / 10578 行，`line-rate=0.859`；不是解释器专项覆盖率，也不是测试成功或分支覆盖的替代证明。`branches-valid=0`。pytest 用时 450.25 秒，外部进程用时 453.04 秒，不能视为 GPU 性能基准。

可按 manifest 的 `archive` 字段核对压缩字节及解压 tar 的大小/SHA-256，再按 `artifacts` 的 POSIX 路径逐个核对成员原文字节；无需调用项目代码或 GPU。需浏览原文时解压到新目录，勿覆盖归档。

此前 [b5 A100 全量失败](../full_suite_a100_b5a3206/README.md) 与 [已完成 A100 专项](../a100_20260906/README.md) 保持原文；不同运行与重叠测试不能累计或互相改写结果。

## 外层说明修订

本 README 与外层 manifest 的 scope 于 2026-09-06 作中性叙述修订，范围仅为非技术性的结果表述。
这两个外层说明文件不再与原公开版本逐字相同；原版已准确备份到仓库外，且可从原 Git 提交恢复。
[provenance.json](provenance.json) 记录原版和当前公开副本的大小、SHA-256 及修订范围。
`raw-full.tar.gz`、其全部成员、实际命令、测试结果、失败信息、设备与真实测试 SHA 均未修改。
