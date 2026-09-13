# A100 结果定位功能回归证据

实际完整测试源码：`c180e280de7de9e2b57e4eaa56dea80373fe5d83`，设备 **NVIDIA A100-SXM4-80GB**。最终结果 **758 passed、2 skipped、851.83 秒，退出 0**。详细结论见[验证报告](../../docs/a100_value_mapping_validation_20260913.md)。

| 原件 | 内容 |
|---|---|
| [最终 JUnit](pass/full/junit.xml) | 760 个条目，0 failures/errors；55 项新增定位与15项 GPU 差分均包含在758中 |
| [完整 stdout](pass/full/validation.stdout.log) | 每项测试终态、耗时、两项双 GPU 跳过原因 |
| [运行清单](pass/full/manifest.json) | 精确命令、版本、进程退出码、运行前后源码与产物散列 |
| [覆盖率 XML](pass/full/coverage.xml) | 全库行覆盖率9811/11270（87.05%），不包含分支覆盖率 |
| [GPU 差分报告](pass/gpu-differential/interpreter_gpu_validation.json) | 15/15实际 Triton GPU 差分通过 |
| [CUDA dot 报告](pass/cuda-dot/report.json) | 四路结果、尾块和输入/输出保护检查；同目录保留 SSA、CUDA 源码、输入输出 NPZ |
| [首次失败](first-failure/full/validation.stdout.log) | adf658e上757通过/1失败/2跳过，唯一失败为缺TileLang；未覆盖 |
| [依赖修复](dependency-repair/repair.json) | 安装和源码切换命令；src/tests/pyproject未改；定向测试退出0 |
| [定向 JUnit](dependency-repair/targeted.junit.xml) | TileLang构建产物子进程重新加载通过 |
| [独立核验摘要](verified_summary.json) | 下载后重新核对源码、JUnit、覆盖率与专项结果 |
| [公开原件清单](archive_manifest.json) | 77份逐字复制原件的SHA-256、字节数和本地相对来源 |

完整最终下载ZIP为1527020字节，SHA-256：`1ec8f450448662d545ff3587b4340473eed1b56149a2e63a241e07f16421bb5b`；其中136份文件逐字核验完成。修复与runner另包24370字节、22份文件，SHA-256：`7d41ad017364f006a2b7c89e4ca2e98276f2d91204c09053e9646378c4e3bd72`。第一次失败完整包1528132字节，SHA-256：`409a20a20d8001730affaa56fc3869c03624ca1add70dd6a05fc1ec8e8f1b372`，136份文件也已核验。

公开副本只选择技术记录，省略HTML覆盖率页面、重复编译缓存、共享库二进制及锁文件。原始运行清单仍完整记录这些产物的散列，因此其中部分路径只存在于完整本地下载包；不能把77份公开文件说成完整ZIP内所有文件。未放入租赁账户、SSH、登录或付款记录。

`runner/`以`.py.txt`保存实际runner、dot探针及顺序控制脚本；根目录conftest也排除整个results目录，避免归档被pytest/doctest意外导入。runner与dot探针通过`--repo`和`--out`接受路径，dot探针冻结本次源码；顺序控制脚本保存实际服务器路径，改写其路径后的运行应另存证据，不能沿用原脚本散列。

控制脚本先打包、再把新ZIP散列写回服务器状态，因此ZIP内部`pass/sequence.json`的`download`指向上一版阶段包。最终ZIP散列以打包后另行取得的[外部状态原件](sequence-after-packaging.json)和[独立核验摘要](verified_summary.json)为准，两者均为`1ec8f450…`。内部原件保持不变；[打包脚本](runner/sequence-tilelang.py.txt)可复查这一顺序。

本次为正确性与兼容性回归，不是性能测试。旧A100703/2、CPU383/15与各专项不累加到758，也不替代双GPU验证。
