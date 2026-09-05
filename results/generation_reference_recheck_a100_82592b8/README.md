# A100 generation 测试稳定化：77 项定向回归

测试源码 `82592b8f6de65052e4258fdd6067956d4ede18c3` 在 NVIDIA A100-SXM4-40GB、
compute capability 8.0、MIG Disabled、PyTorch 2.5.0+cu124 和 Triton 3.1.0 上完成：

```text
77 passed in 19.66s
```

退出码 0；[JUnit](junit.xml) 包含 77 tests、0 failures、0 errors、0 skipped。
19.66 s 为 [pytest stdout](validation.stdout.log) 的汇总时间；运行器记录的子进程时长为 21.199 s，
这些时间不是性能基准。运行前后 HEAD 均为上述提交，已跟踪文件状态为空。
实际 argv、受控环境、完整版本及 15 个原始产物的大小和 SHA-256 见 [原始 manifest](manifest.json)。

## 修改及验证范围

本轮执行整个 `tests/test_generation.py`。77 是 pytest 参数化测试条目数，
其中包含生成与配置检查，也包含实际 GPU 调用；不能称为 77 个全部独立执行的 GPU kernel。

相对前一提交，测试稳定化仅修改 `test_squeezing_the_innermost_level` 的两条初始化表达式：

- 输出缓冲区由 `torch.empty` 改为有限非零值 `-123.0`，使未写行也具有可比较的确定内容。
- 索引由有放回的 `torch.randint` 改为 `torch.randperm(num_rows)[:num_indices]`，
  保持随机目标行并保证互不重复，避免不同 program 对同一位置写入不同值。

原 arrangement/application、Tensor 描述、实际内核构建与调用、1024×128 输出及 10 个索引的参数组合、
参考赋值循环、完整矩阵比较和默认 `torch.allclose` 容差均保留，没有新增 skip 或升级依赖。
期望值仍在内核执行前从初始化后的输出克隆。

受控诊断已证明一个失效机制：未写区域含 NaN 时，即使 output 与 expected 位级相同，
默认 `allclose` 仍可返回 false。重复目标行另有潜在并发写竞争；本次唯一索引避免这种不确定输入。
[377 全量失败](../full_suite_a100_377daec/README.md)的原始分配器状态和索引没有保存，
因此不能把受控机制复现声称为已经还原该次失败的精确原因。

本目录只包含 82592b8 的这次定向结果。归档时另一次完整套件仍为 **RUNNING**，
没有最终汇总或退出码，其结果不包含在本目录。
77 不与 b5 的 14/180、377 的 16 项 jagged 定向或任何全量通过数相加，也不代表全仓库通过。
各轮失败、修复、源码 SHA 和验证范围分别保留。

## 原文与归档索引

原始 runner manifest、stdout/stderr、硬件与源码检查日志、JUnit 均按原字节复制，未进行脱敏改写或换行转换。
[archive.json](archive.json) 使用 POSIX 相对路径记录上述 16 个原文件及本 README 的大小和 SHA-256；
索引不包含自身散列。发布前已对归档文本检查连接地址和常见密钥格式，未发现匹配项。
执行前后的源码状态分别见 [before-head](before-head.stdout.log)、[after-head](after-head.stdout.log)
及对应 status 日志；设备信息见 [nvidia-smi](nvidia-smi.stdout.log) 和 [环境记录](environment.stdout.log)。
