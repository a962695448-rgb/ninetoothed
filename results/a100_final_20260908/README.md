# 2026-09-08 A100 最终源码验收

实际测试源码：`59200180717a45f173a37ee8da5356d31e02018c`。完整套件与当前 CI 使用相同的 doctest/coverage 测试范围：682 passed、2 skipped、0 failures/errors，467.20 秒。两个 skip 明确要求至少双卡。

同一源码的 A100 15 项 Triton 差分和独立 CUDA 标量 dot 验证也通过；无 Torch/Triton 的独立 NumPy CPU 选择为 307 passed、15 deselected（32.49 秒）。不同范围有重合，不相加。

- [报告](../../docs/a100_final_validation_20260908.md)
- [汇总](summary.json)、[完整运行清单](full_manifest.json)、[JUnit](junit.xml)、[coverage](coverage.xml)
- [GPU 差分报告](gpu_report.json)、[CUDA dot 报告](cuda_dot_report.json)
- [原始证据](raw.zip)、[逐文件散列](raw_manifest.json)

原始包保留 f5d57b1 的收集失败（3 errors）、d6b0864 的首次完整通过与最终 5920018 的完整通过。前者因误导入 results 下的历史实验脚本失败；根 conftest 将档案排除在活跃测试发现之外，所有原有测试与包 doctest 继续收集。Ruff 同样排除历史原件；7 个源/测试文件仅做空行机械修正，语法树等价证据与风格检查原文随包保留。

当前证据仅覆盖单 A100 和所列支持语义；不等于双卡通过、上游合并或官方评审结果。
