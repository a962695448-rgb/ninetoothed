# 内存依赖与中间检查点验证记录

最终源码 `ed332733db28dbf16de06f166b16766760148958`：CPU **460 passed、15 deselected**；A100-SXM4-40GB完整 **835 passed、2 skipped、570.53秒**，无失败或错误。完整说明见[实施与验证报告](../../docs/memory_checkpoints_20260914.md)。

| 原件 | 内容 |
|---|---|
| [CPU清单](cpu/ed33273/manifest.json) | 精确命令、源码前后散列、依赖、wheel及各项退出码 |
| [CPU JUnit](cpu/ed33273/cpu.junit.xml) | 460项通过；取消选择的15项不记为通过 |
| [安装核验](cpu/ed33273/installed-probe.stdout.log) | 独立普通wheel、67份源码逐字验证、无Torch/Triton |
| [新故障包](cpu/ed33273/replay-cases.zip) | 4个已有接口新案例、20个矩阵乘内部案例、1个共享存储案例，均独立回放通过 |
| [分析原始样本](cpu/ed33273/analysis-benchmark/benchmark.json) | 7轮交错微基准，保留所有正负结果与诊断一致性检查 |
| [A100完整清单](gpu/ed33273/full/manifest.json) | 实际设备、源码、命令、stdout/stderr与产物散列 |
| [A100 JUnit](gpu/ed33273/full/junit.xml) | 837条目，835通过，2项双GPU条件跳过 |
| [A100完整日志](gpu/ed33273/full/validation.stdout.log) | 每项测试终态与原始耗时 |
| [覆盖率](gpu/ed33273/full/coverage.xml) | 全库行覆盖率10069/11650，86.43%；未采集分支覆盖率 |
| [GPU差分](gpu/ed33273/gpu-differential/interpreter_gpu_validation.json) | 15/15实际Triton GPU差分；包含在835中 |
| [独立CUDA dot](gpu/ed33273/cuda-dot/report.json) | NumPy、前端SSA CPU、CUDA SSA CPU、CUDA GPU四路对照 |
| [下载后核验](gpu/ed33273/verified_summary.json) | 重新核对JUnit、测试组数量、版本、退出码和覆盖率 |
| [外部ZIP收据](gpu/ed33273/external-download.json) | 散列写在ZIP外，不存在包内自引用问题 |
| [公开文件清单](archive_manifest.json) | 317份直接复制原件，以及两个ZIP内640份原件的字节数/SHA-256 |

`cpu/3312a79`与`gpu/3312a79`保留首版459/15、834/2和性能负例；没有把它们累计到最终结果。历史9c5ffba和adf658e共8个包的独立回放日志以`legacy-`开头，位于对应CPU目录。

最终GPU完整下载包1555142字节、138份原件，SHA-256为`5ac822a4859f9fcd0878fb9976806b924ded9f31979a15e8d05a5cb6fd580e08`。第一轮完整包1554991字节、138份原件，SHA-256为`f5acb88ac9f32e29484fb5f315cf982eeee9eb508cc28e27c0fdbac3a3727954`。另有21份setup与控制记录完成独立下载校验，SHA-256为`c5d86f678b56a79b185739e3ef0403dd969c1c29395ec5dda8d72e5fcec7a27d`。

公开副本省略HTML覆盖率页面、编译出的共享库、缓存与账户/SSH记录；原始运行清单仍记录完整本地包中这些产物的散列，不能把本目录说成整个服务器备份。回放ZIP由已验证原件无损打包，每个成员再次核对散列；本地未压缩副本仍保留。

最后普通wheel大小216078字节，SHA-256为`81fd40b52307ac058120872700b98c8a97ebb6a71fc1be354a80ac54db2f59b3`。它仍声明Triton依赖；隔离CPU安装显式跳过依赖，不是新的CPU-only发行包。
