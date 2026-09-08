# CPU 解释器：最终 A100 验收

日期：2026-09-08。实际源码 **`59200180717a45f173a37ee8da5356d31e02018c`**；后续文档归档不视为额外硬件测试。设备为 NVIDIA A100-SXM4-40GB，sm80，驱动 550.127.05，PyTorch 2.5.0+cu124，Triton 3.1.0，NumPy 2.1.3。验证过程不与 Hadamard 性能计时并发。

## 结果

| 范围 | 结果 |
|---|---|
| 完整仓库测试、包 doctest、coverage | 682 passed、2 skipped；0 errors/failures；467.20 s |
| 实际 A100 Triton 差分 | 15/15 用例，9 个程序、10 类别 |
| 原生 CUDA 后端标量 dot | 独立缓存重新编译并运行，通过 |
| CUDA 不可见、无 Torch/Triton 的 CPU 选择 | 307 passed、15 deselected；32.49 s |
| 代码规范 | 仓库自定义风格、Ruff lint、Ruff format 均通过 |

全库行覆盖率为 86.05%（9329/10841）；这是整个仓库的行覆盖率，不是解释器专项覆盖率，也不是分支覆盖率。15 个 GPU 用例包含在完整套件中；CPU 选择、GPU runner、完整套件和独立 probe 按范围分列，不累计测试数量。

## 本轮发现与修正

原 f5d57b1 在完整 pytest 收集时出现 3 个错误：`--doctest-modules` 导入了 `results/` 中依赖旧路径和命令行参数的历史脚本。根 `conftest.py` 将这一归档目录排除，保留原 `src/`、`tests/` 及其他活跃 doctest 范围。归档原文、参数、容差、计算代码都未删除或放宽。d6b0864 完整回归为 682/2、947.77 秒；之后修正 Ruff 的归档发现范围及 7 个文件的 84 处空行，语法树保持一致，并在最终 5920018 上重跑完整硬件测试。

## 验收条款与边界

已取得五类基础 application、float32/int32/bool、masked 访存、默认 pass 前后 SSA 语义比较、实际 A100 差分、trace 与显式 unsupported 诊断所需证据。保留 softmax、M/N 多 program 矩阵乘、单步/断点/watch、逐 pass 比较、来源候选及独立复现导出。

- 单 A100 无法执行两个同机多卡测试；JUnit 已保留明确 skip 原因，不声称通过。
- dot 的每个 program 仍须包含完整 K；没有实现跨 program split-K 累积或 Tensor Core CPU 模拟。
- 结构变化时来源追踪给出已声明的 SSA 候选范围，不能保证任意变换都唯一定位因果操作。
- 上游 PR、主分支合并与官网提交仍等待用户验收。本轮技术通过不等同于官方评定。

完整命令、源码前后状态、环境、日志和失败记录见[归档索引](../results/a100_final_20260908/README.md)。历史 82592b8 的 600/2 仍独立保存，未用于替代本轮结果。
