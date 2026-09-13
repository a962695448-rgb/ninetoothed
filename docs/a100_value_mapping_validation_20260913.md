# 结果对应与依赖定位的 A100 回归

源码 `c180e280de7de9e2b57e4eaa56dea80373fe5d83` 在 **NVIDIA A100-SXM4-80GB** 完成完整 pytest、包 doctest 和覆盖率采集：**758 passed、2 skipped、0 failed/error，851.83 秒**。测试前后已跟踪源码均保持干净，验证进程和封装进程都退出 0。[原始证据索引](../results/a100_value_mapping_20260913/README.md)

这轮补齐[结果对应与执行依赖定位](value_mapping_20260913.md)的实际 GPU 回归。此前纯 CPU 验证源码为 `adf658e`；新提交 `c180e2` 仅在 `requirements.txt` 增加 `tilelang==0.1.14`，`src`、`tests` 和 `pyproject.toml` 无差异。

## 检查结果

| 范围 | 结果 | 解释 |
|---|---|---|
| 完整测试及 doctest | 758 通过，2 跳过 | 收集 760 项，无失败和错误 |
| 新增结果对应与依赖定位 | 55/55 通过 | 已包含在 758 中，不另外累加 |
| 实际 Triton GPU 差分 | 15/15 通过 | 同源码、同卡专项先运行；完整套件也包含这些用例 |
| 独立 CUDA 后端 dot | 四路对照通过 | NumPy、前端 SSA CPU、CUDA SSA CPU、实际 CUDA GPU |
| TileLang AOT 产物重新加载 | 定向及完整回归通过 | 实际构建后在子进程重新加载并执行 |
| 全库行覆盖率 | 87.05%，9811/11270 行 | 不是解释器专项覆盖率；没有采集分支覆盖率 |

两项跳过分别为 AOT 跨设备测试和 Triton 跨 CUDA 上下文复用测试，均要求同机至少两张 GPU。本次只有一张卡，没有把跳过算为通过。

独立 CUDA dot 使用 float32，输入为 `(7,3)` 和 `(3,6)`，输出 `(7,6)`，seed 为 2026。四个 M/N program 覆盖尾块，容差 `rtol=atol=1e-3`；输入及输出边界保护均未改变。前端 CPU 和实际 CUDA GPU 相对 NumPy 的最大绝对误差均为 0，CUDA SSA CPU 为 `1.1920928955078125e-07`。这是一个标量分解正确性用例，不代表 Tensor Core 性能或完整 CUDA 后端覆盖。

## 环境与首次失败

环境为 Python 3.12.7、Torch 2.5.0+cu124、Triton 3.1.0、TileLang 0.1.14、NumPy 2.1.3、SymPy 1.13.1、CUDA 12.4，驱动 550.127.05，计算能力 8.0。完整版本清单保留在 [environment.stdout.log](../results/a100_value_mapping_20260913/pass/full/environment.stdout.log)。

第一次完整运行使用 `adf658e`，结果为 **757 passed、1 failed、2 skipped，3172.34 秒**。唯一失败是 `test_aot_built_artifact_can_be_reloaded[tilelang-cuda]`：环境中没有 TileLang，报 `ImportError: TileLang is required to build this backend artifact.`。该失败保留在独立目录。

修复把所需依赖写入 `requirements.txt`，安装 TileLang 0.1.14，然后先复测失败用例，再在独立输出目录完整运行 760 项；没有删测试、修改容差、调整计算实现或覆盖失败日志。前后两次耗时受缓存与依赖环境影响，**不能解释为算法加速比**。

## 复验入口

使用具备上述 CUDA/Python 依赖的环境，从当前归档取出 [runner 源文件](../results/a100_value_mapping_20260913/runner/a100_final_runner.py.txt)，放到待测试仓库外。另准备一个检出 `c180e280de7de9e2b57e4eaa56dea80373fe5d83` 的干净工作目录，再运行：

```bash
python /path/to/a100_final_runner.py.txt \
  --mode full \
  --repo /path/to/ninetoothed-at-c180e2 \
  --out /path/to/new-evidence-directory \
  --expected-sha c180e280de7de9e2b57e4eaa56dea80373fe5d83
```

输出目录必须尚不存在。runner 检查 Git 版本、实际 GPU 和测试终态，保留完整命令、stdout/stderr、JUnit、覆盖率和源码前后状态。依赖安装应沿用记录中的核心包版本；[constraints.txt](../results/a100_value_mapping_20260913/runner/constraints.txt)保留本次保护 Torch/Triton/NumPy/SymPy 的约束。没有另设空编译缓存；原默认缓存参与本次复跑。

本报告验证新增定位功能与现有单卡执行路径的兼容性。对于没有可信结果对应关系的任意重构、内存别名历史和未映射的内部操作，仍适用[实施报告](value_mapping_20260913.md)中的边界，不能由测试通过推出“任意错误都可唯一归因”。上游 PR、主分支合并和正式提交均未执行。
