# 结果对应与依赖定位的 CPU 验证记录

实际验证源码：`adf658e6077768f22af7d3cce2e6b3efdb9dc3c0`。本目录仅记录 CPU 检查与安装包回放，没有新的 GPU 测试。

| 检查 | 实际结果 | 原件 |
|---|---|---|
| 选定 CPU 回归 | 383 passed、15 deselected，73.31秒；新增55项已包含其中 | [日志](adf658e/cpu.stdout.log)、[JUnit](adf658e/cpu.junit.xml) |
| Ruff、格式及贡献风格 | 全部退出0 | [命令及源码清单](adf658e/manifest.json) |
| 严格 Sphinx | 真实 CPU Torch 环境构建退出0 | [标准输出](adf658e/sphinx.stdout.log)、[标准错误](adf658e/sphinx.stderr.log) |
| 独立 wheel | 离线构建安装，源文件逐字核验，未安装 Torch/Triton | [安装检查](adf658e/installed-probe.stdout.log) |
| 新故障独立回放 | 4组全部通过，使用安装环境的 `python -I` | [包0](adf658e/mapped-cases/0/failure.json)、[包1](adf658e/mapped-cases/1/failure.json)、[包2](adf658e/mapped-cases/2/failure.json)、[包3](adf658e/mapped-cases/3/failure.json) |
| 历史故障兼容性 | 9c5ffba的4组真实默认管线故障包均回放通过 | 本目录 `adf658e/legacy-*.stdout.log` |

完整判断见[实施报告](../../docs/value_mapping_20260913.md)。[独立核验摘要](verified_summary.json)列出具体定位和依赖边界；[文件清单](archive_manifest.json)记录139份复制原件的字节数与SHA-256。未复制虚拟环境或编译缓存。

wheel大小208644字节，SHA-256为 `84c753ca7334571ce5f0e364d407ed8fb91f8f2fe2404e4c78577669d3acf223`。普通包仍声明Triton依赖，隔离CPU安装显式跳过依赖后运行，不等于CPU-only发行包。

本轮 NumPy 为2.5.2，SymPy为1.14.0；准确Python版本、各阶段命令和文件散列以清单为准。CPU正确性测试与新旧回放是不同证据范围，不能累计为更多独立应用。

`9783c15/`保留初次CPU验证与在线构建失败，整体FAIL没有被成功复跑覆盖。该版本的379通过不代替最终383通过，也不解释为性能提升。旧A100703/2属于源码3fc27e0，与本轮分开记录。
