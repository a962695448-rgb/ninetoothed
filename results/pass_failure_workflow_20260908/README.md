# 阶段差分与自动失败回放验证

功能与测试源码：`9c5ffbad6d75e8b7f083270f08dc4009a8fe4424`。实现、示例和支持边界见[技术说明](../../docs/pass_failure_workflow_20260908.md)。

| 范围 | 结果 | 原始证据 |
|---|---|---|
| 无 Torch/Triton 的选定 CPU 回归 | 328 passed、15 GPU deselected、57.58 s、exit 0 | [完整清单](9c5ffba/manifest.json)、[日志](9c5ffba/cpu.stdout.log)、[JUnit](9c5ffba/cpu.junit.xml) |
| Ruff、格式、项目贡献风格 | 全部 exit 0 | 同一清单及 `ruff-*`、`style`、`demo-style` 日志 |
| 普通 wheel 隔离安装与回放 | 64 个 Python 源文件逐字一致；演示、回放 exit 0 | [安装核对](9c5ffba/installed-probe.stdout.log)、[回放](9c5ffba/installed-replay.stdout.log)、同一清单中的 wheel SHA-256 |
| 真实默认 pass 管线的自动导出 | Triton/CUDA × dot/transpose 共四组；4 次隔离回放及4个修复对照通过 | [管线清单](9c5ffba/pipeline-replays/manifest.json)，目录下有四个完整复现包 |
| 全站文档与 Torch CPU 导出 | 无 mock 的严格 Sphinx exit 0，28 HTML 页；非连续 Torch CPU 输入捕获及无 Torch 环境回放通过 | [文档清单](9c5ffba/documentation/manifest.json)、[构建日志](9c5ffba/documentation/sphinx.stdout.log)、[适配器回放](9c5ffba/documentation/torch-replay.stdout.log) |
| 早一版开发冻结检查 | `7d158f3`：325 passed、15 deselected；无 Torch 环境 Sphinx exit 1，因此该轮总状态为 FAIL | [原清单](7d158f3/manifest.json)、[原文档错误](7d158f3/sphinx.stderr.log) |

这些范围与历史 GPU 数字不相加。新功能的 21 项测试已包含在 328 中。四组额外管线案例先以 NumPy 独立检查原始和正确优化输出，再注入故障、保存精确输入，并由已安装 wheel 的 `python -I` 进程回放。错误注入不能表述为新发现的上游历史缺陷。

CPU 检查环境：Python 3.12.3、NumPy 2.5.2、SymPy 1.14.0、pytest 9.1.1；不存在 Torch、Triton。文档环境单独安装 Torch 2.5.1+cpu、SymPy 1.13.1、Sphinx 7.3.7 和主题 0.17.1；不能把该环境与纯 NumPy 检查混称。文档未执行 GPU 内核或使用 mock。

## 复放

在含本次功能的 NineToothed 安装环境中，可运行任一保存的顶层 `replay.py`，例如：

```bash
python results/pass_failure_workflow_20260908/9c5ffba/pipeline-replays/cuda-dot/replay.py
```

退出 0 表示重现保存的错误，不表示错误样例计算正确。回放会核对差异类别、输出名、操作观测位置与 trace 对齐状态；候选程序修复后应失败。

所有复制文件逐字核对，见[归档清单](archive_manifest.json)。原运行 manifest 记录当时绝对路径；换机器运行需调整路径。普通 wheel 和已构建 HTML 保留于本地完整归档，公开仓库省略这两类二进制；其原运行文件散列和路径仍如实保留。Git LFS 资源按记录的 oid 与 size 验证后在隔离文档副本中使用真实内容。

首个控制器预检查曾因把 Git LFS 管理的 PNG 当成 UTF-8 文本退出，尚未运行测试；修正为校验 LFS 内容后才执行这里归档的两轮。该首次错误只有任务工具输出，无独立原始日志文件，不将后续运行伪称为该次日志。
