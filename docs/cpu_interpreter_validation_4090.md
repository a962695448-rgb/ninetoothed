# CPU 参考解释器：RTX 4090 验证记录

2026-09-05，在租赁的 NVIDIA GeForce RTX 4090 上完成解释器专项验证：**224 项通过**，其中包含 **14 项真实 Triton GPU 差分**。将 `CUDA_VISIBLE_DEVICES` 设为空后，CPU 可执行部分 **209 项通过、15 项取消选择**。这 15 项需要 CUDA 来执行 GPU 对照或构造 CUDA 输入，已在前述可见 GPU 的专项运行中通过；不是把失败测试跳过。

三份记录对应冻结的代码提交 **`76ca6464fc921bc1419700b22f730b4084b3035b`**。后续增加本报告和证据文件的提交不改变这一来源，也不应被写成“已对新提交重跑”。14 项 GPU 报告是 224 项的子集，不能相加宣传为 238 项。

| 验证范围 | 已有结果 | 原始证据 |
|---|---|---|
| 解释器、SSA 与 GPU 专项回归 | 224 passed，13.47 s | [专项日志](../results/interpreter_quality_rtx4090.log) |
| 隐藏 CUDA 的 CPU 回归 | 209 passed，15 deselected，9.94 s | [CPU 日志](../results/interpreter_cpu_only_rtx4090_host.log) |
| 独立真实 GPU 报告 | 14/14 PASS；8 个应用程序、9 类用例 | [GPU JSON](../results/interpreter_gpu_rtx4090.json) |
| 仓库完整测试套件 | **运行中；已收集 587 项，尚无最终汇总** | 待完成后保存独立日志和退出状态 |
| A100 官方指定验证 | **尚未运行** | RTX 4090 结果不能替代 A100 实测 |

时间是这次测试进程的运行时长，不是解释器性能基准。完整测试套件的阶段性进度不计作最终通过，也尚未向上游创建 PR。

## 环境与比较方法

GPU 为 RTX 4090、compute capability 8.9；Python 3.12.3、NumPy 1.26.4、SymPy 1.13.1、PyTorch `2.6.0a0+ecf3bae40a.nv25.01`、Triton 3.1.0、CUDA 12.8。GPU 报告逐项保存实际设备、dtype、形状、种子、pass 列表和发射前 SSA 的 SHA-256。

同一个 arrangement/application 经共享 frontend 与 SSA 管线生成程序，再分别比较：独立 NumPy 期望值、原始 frontend SSA 的 CPU 解释结果、目标默认 pass 之后的 SSA 解释结果，以及真正发射到 Triton GPU 的结果。默认 pass 包含 `ssa.canonicalize`、`ssa.analyze_effects`、`ssa.select_schedule`、`ssa.triton.optimize_schedule` 和 `ssa.decompose_linalg`；没有在失败时退回原始程序冒充 pass 后验证。

浮点采用 `rtol=1e-3, atol=1e-3`，int32 与 bool 完全相等。14 项覆盖逐元素、尾块掩码、向量/行/列广播、行归约、比较、if/for 两条分支、softmax、有符号整数向下取整除法与余数。每项检查 GPU 输出前后保护区未改写，输入未被修改。浮点最大绝对差出现在行归约，为 `1.9073486328125e-6`；整数和布尔差分为 0。

## 复现命令

在该提交的仓库根目录、项目依赖齐全的 Python 环境中运行。设置 `PYTHONPATH=src` 是为了使用本次 checkout 的源码；禁用额外 pytest 插件以保持与实测条件一致。

```bash
export PYTHONPATH=src
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

# 可见 GPU 的专项测试：包含 14 项真实 GPU 对照和 CUDA 输入拒绝测试。
python -m pytest -q --color=no tests/test_interpreter*.py tests/test_ssa*.py
python scripts/verify_interpreter_gpu.py --device 0 \
  --report results/interpreter_gpu_new_run.json

# CPU 运行：仅对本条命令隐藏 CUDA，排除恰好需要 CUDA 的 15 项。
CUDA_VISIBLE_DEVICES="" python -m pytest -q --color=no \
  tests/test_interpreter*.py tests/test_ssa*.py \
  -k 'not test_cpu_interpreter_matches_actual_triton_gpu and not test_cuda_tensor_is_rejected_by_cpu_interpreter'

# 全仓库兼容性测试；必须等待最终汇总和退出码。
python -m pytest -q --color=no
```

独立 GPU 脚本在无可用 CUDA 时报告 `UNVERIFIED` 并返回 2，有差分失败返回 1，全部通过才返回 0。CPU 测试机器上装有 PyTorch，但 CUDA 已隐藏；这份记录不能说成“未安装 PyTorch 的完整套件运行”。

## 从失败中修复了什么

- 第一轮真实 GPU 对照有 4 项失败：Triton 3.1 不支持当时生成的三个操作数 `and`，以及广播输入错误地使用全局输出 program 索引，导致矩阵除第一行外的广播错误。修复为嵌套二元条件及输入自身的地址/掩码映射，并补充同行数的单例行、单例列和尾块回归。
- 审查发现 trace 观察可能额外读取已被 mask 屏蔽的越界位置。修复为只读取活动地址，增加 trace 打开/关闭一致性与保护区验证。
- 调试器导出回放修复了数组 strides 和可写权限保留，避免把非连续视图的布局问题在回放时“变没了”。
- 有符号整数 `//` 与 `%` 统一为向负无穷取整及与除数同号的余数语义，并加入混合正负整数尾块的真实 GPU 对照。
- 旧上游 `test_aot.py` 的 stream 上下文写法改为 `torch.cuda.stream(torch.cuda.Stream(...))`，兼容当前 PyTorch。保留原测试断言，没有放宽误差阈值。

## 当前边界与下一步

直接 `linalg.dot`/`linalg.matmul` 和受限的单逻辑 program 标量分解已有 CPU 支持；多 program 分块标量分解等场景仍明确不支持。**当前 GPU runner 未纳入优化后 dot 的对照**，不能把这 14 项通过写成“所有矩阵乘法路径通过”。完整边界见 [CPU 解释器文档](source/cpu_interpreter.rst)。

下一步先补齐当前完整套件的最终汇总，再在真实 A100 上重跑专项报告和完整套件，检查原始证据后准备向 `InfiniTensor/ninetoothed:master` 提交 PR。上游评审、合并与训练营评选均尚未完成。

## 代码与文档检查

本次新增及修改的 Python 文件已通过 Ruff 0.16.6 的全库 check、format 检查（125 个文件）和仓库贡献风格检查。纯格式调整的 AST 保持一致；另按贡献规则调整了 17 条诊断消息的大小写或标点，相关回归为 78 passed、14 项 GPU 用例取消选择，计算与控制逻辑未变。服务器全量运行保持在上面记录的 `76ca646` 源码，避免运行中混合不同文件版本。

服务器 Sphinx HTML 构建成功。最初缺少系统 `python3-tk`，补齐后通过；构建报告的两个标题下划线长度警告已在本次文档中修正。源码 doctest 收集没有发现用例（退出码 5），不计为新增通过的测试。正式 PR 还需使用仓库规定的 kebab-case 分支名，并附实际 pytest 输出。

完整套件早期有一项 TileLang 重载测试因缺少可选依赖失败；保持 NumPy 1.26.4 和原有 PyTorch，补装 TileLang 0.1.14 与 ml-dtypes 0.5.4 后，该原测试单独复测 1 passed。原全量日志保留失败，不把这一复测改写成已经全量通过。

原始记录逐字节保存，SHA-256 与字节数见 [证据清单](../results/interpreter_rtx4090_manifest.json)。`results/.gitattributes` 禁止这些原始文件的自动换行转换。
