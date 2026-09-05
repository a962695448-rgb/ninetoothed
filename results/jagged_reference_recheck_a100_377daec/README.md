# A100 jagged 测试参考修复：16 项定向回归

测试源码 `377daec6242864a920de43a55523ac3d5f582648` 在 A100-SXM4-40GB、PyTorch 2.5.0+cu124 上完成：

```text
16 passed in 8.83s
```

退出码 0；JUnit 为 16 tests、0 failures、0 errors、0 skipped。运行前后 HEAD 相同且已跟踪文件干净。
原始 argv、受控环境、硬件、依赖和原文件散列见 [manifest](manifest.json)，日志见 [stdout](validation.stdout.log)。

相对上一版本，唯一测试修改是 `tests/test_jagged.py`：使用明确的 packed values、offsets 和 jagged_dim 构造输入；
在 GPU 调用前从 dense 输入构造完整参考结果；expand 比较全部 packed values，并检查 offsets 不变。
原 16 参数组合、默认 allclose 容差、实际内核调用及内核定义均保留，没有更换编译器核心或依赖。

该修改解决 [初轮 b5 全量](../full_suite_a100_b5a3206/README.md)中 8 个参考转换未实现、8 个列表工厂不支持指定 jagged 维度的问题。
这次定向通过证明上述 16 项在修复后的测试中完成了实际 GPU 比较，不把初轮失败改写成通过。
完整套件另行运行、单独归档；本目录只包含这次定向结果，不能视为全量通过，也不与 14/180 项相加。

[CPU 能力诊断](cpu-construction-probe.json)在同环境核对两种构造路径、两个 jagged 维度的 values/offsets，
`cuda_initialized=false`；它只用于确认测试输入构造接口，不是额外 GPU 测试。
对应脚本以 `.py.txt` 保存，运行时复制为 `.py` 并先修改其新的输出路径，不能覆盖已有诊断记录。

[archive.json](archive.json)为归档字节索引；原始 runner manifest 和日志保持原文。
