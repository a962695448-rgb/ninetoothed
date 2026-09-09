# A100 完整回归通过证据

实际运行源码：`3fc27e0399d82b02be5407b9542c67763c4631a7`。

**703 passed、2 skipped，929.31秒，退出0**；705个JUnit条目全部核对，无failure/error，两个跳过项均要求多GPU。

- [完整日志](default-full/full/validation.stdout.log)
- [JUnit](default-full/full/junit.xml)
- [环境、精确命令与源码前后状态](default-full/full/manifest.json)
- [全库行覆盖率原文](default-full/full/coverage.xml)：9560/11032，86.66%，不是专项或分支覆盖率。
- [独立核验摘要](verified_summary.json)：包括完整套件中15项真实GPU差分的名称。
- [归档哈希清单](archive_manifest.json)
- [此前90秒单例诊断](bounded-canary/canary.controller.log)：中断记录，不计作通过；同一用例随后在完整套件中通过。

GPU测试完成后才进行文件提取；提取阶段不执行GPU测试。完整备份包含96份原始文件，ZIP SHA-256为 `d2b2a9fde1b9324647e7c0a77b1ec1ecf848708fe5ae90dc81e4d6c89b99bbfb`。公开25份技术文件经逐字核验，HTML覆盖率页面保留在本地完整备份。

独立CUDA dot及此前15项专项JSON来自同源码的另一张A100，见[首次验证目录](../a100_validation_20260909/README.md)，不将其改记为本次同卡重跑。完整解释见[验收报告](../../docs/a100_complete_20260909.md)。
