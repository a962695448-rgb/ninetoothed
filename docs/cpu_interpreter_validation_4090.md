# CPU 参考解释器：RTX 4090 验证记录

2026-09-06的后续实际A100差分与专项结果见 [A100验证记录](cpu_interpreter_validation_a100.md)。以下保留RTX 4090各轮运行及当时的验收状态。

更新日期：2026-09-05。冻结源码 `5b377252cc4452b5ccc48c46ff1ae07a4e5e0e8a` 在 RTX 4090 上完成含 doctest 与 coverage 的完整重跑：**591 passed、2 skipped，6036.09 s，退出码 0**。归档时服务器 HEAD 与该提交一致，已跟踪文件无修改。两个跳过项均要求同机至少两张 GPU。**此结论仅属于 `5b37725` 的本轮运行，不覆盖后续演示与测试改进的完整套件；A100 验证尚未运行。**旧完整套件的 11 项失败和一次 SIGSEGV 记录保留，SIGSEGV 原因仍未确定。

## 第一轮：`76ca646` 专项与完整套件

2026-09-05，在 NVIDIA GeForce RTX 4090 上完成解释器专项验证：**224 项通过**，其中包含 **14 项真实 Triton GPU 差分**。将 `CUDA_VISIBLE_DEVICES` 设为空后，CPU 可执行部分 **209 项通过、15 项取消选择**。这 15 项需要 CUDA 来执行 GPU 对照或构造 CUDA 输入，已在前述可见 GPU 的专项运行中通过；不是把失败测试跳过。

三份记录对应冻结的代码提交 **`76ca6464fc921bc1419700b22f730b4084b3035b`**。后续增加本报告和证据文件的提交不改变这一来源，也不应被写成“已对新提交重跑”。14 项 GPU 报告是 224 项的子集，不能相加宣传为 238 项。

| 验证范围 | 已有结果 | 原始证据 |
|---|---|---|
| 解释器、SSA 与 GPU 专项回归 | 224 passed，13.47 s | [专项日志](../results/interpreter_quality_rtx4090.log) |
| 隐藏 CUDA 的 CPU 回归 | 209 passed，15 deselected，9.94 s | [CPU 日志](../results/interpreter_cpu_only_rtx4090_host.log) |
| 独立真实 GPU 报告 | 14/14 PASS；8 个应用程序、9 类用例 | [GPU JSON](../results/interpreter_gpu_rtx4090.json) |
| 仓库完整测试套件 | **11 failed、574 passed、2 skipped，2813.90 s；退出码 1** | [完整日志压缩包](../results/rtx4090_compatibility_20260905/nine_full_suite_final.log.gz) |
| A100 官方指定验证 | **尚未运行** | RTX 4090 结果不能替代 A100 实测 |

旧全量失败包含缺少 TileLang 的重载用例 1 项、`debugging.addmm` 1 项、emitter capability 边界 1 项和 `jagged.expand` 8 项。缺少依赖的用例补装后单独通过，其余三类在后续提交中修复；原退出码和失败记录保留。

## 第二轮：`5b37725` 兼容性修复与全量排查

以下记录均对应 `5b377252cc4452b5ccc48c46ff1ae07a4e5e0e8a`，完整参数、退出状态和原文/压缩文件 SHA-256 见 [兼容性证据清单](../results/rtx4090_compatibility_20260905/manifest.json)。压缩包解压后的内容为原始日志。

| 验证范围 | 结果与退出状态 | 原始证据 |
|---|---|---|
| 解释器、SSA、debugging、emitter、jagged 和重载定向回归 | 254 passed、1 skipped，67.66 s；退出码 0；skip 为同机双卡条件 | [定向日志](../results/rtx4090_compatibility_20260905/nine_p0_real_gpu_retest.log.gz) |
| 含 doctest、coverage 与额外 `faulthandler_timeout=120` 的完整运行 | SIGSEGV；进程返回 -11；没有 pytest 最终汇总 | [异常日志](../results/rtx4090_compatibility_20260905/nine_full_ci_5b37725.log.gz) |
| FP8 addmm 单例，无 coverage、无额外 timeout | 1 passed，183.80 s；退出码 0 | [单例日志](../results/rtx4090_compatibility_20260905/nine_fp8_isolated_no_coverage.log.gz) |
| 同一 FP8 addmm 单例，有 coverage、无额外 timeout | 1 passed，6.68 s；退出码 0 | [coverage 单例日志](../results/rtx4090_compatibility_20260905/nine_fp8_with_coverage.log.gz) |
| 含 doctest、coverage、去掉额外 timeout 的完整重跑 | **591 passed、2 skipped，6036.09 s；退出码 0** | [完整运行清单](../results/full_suite_rtx4090_5b37725/manifest.json)、[日志](../results/full_suite_rtx4090_5b37725/nine_full_ci_no_timeout_5b37725.log.gz)、[归档说明](../results/full_suite_rtx4090_5b37725/README.md) |

完整运行的 [JUnit](../results/full_suite_rtx4090_5b37725/nine_full_ci_no_timeout_5b37725.xml.gz) 记录 593 个测试条目、0 errors、0 failures、2 skipped，与 591 passed 的日志汇总一致。跳过项为 `tests.test_aot::test_add[True-45327-dtype0-bf16-cuda]`（`multi-device testing requires at least 2 devices`）和 `tests.test_built_artifact_reload::test_triton_aot_handle_is_reusable_across_cuda_contexts`（`Triton multi-context testing requires at least 2 CUDA devices`）。单卡运行没有证明这两个跨设备场景通过。

[Coverage XML](../results/full_suite_rtx4090_5b37725/nine_coverage_no_timeout_5b37725.xml.gz) 报告全仓库 line-rate 为 **86.76%**，记录 9178 条已覆盖行、10578 条有效行；这是该次全库运行的行覆盖率，不是解释器专项覆盖率。XML 的 branch 数据没有提供有效分支覆盖计数，不据此宣称分支覆盖完成。

SIGSEGV 原因未定。本次去掉额外 timeout 的完整重跑成功是独立事实，不能据此归因为 OOM、认定 timeout 是根因或宣称 SIGSEGV 已被定位修复。两个 FP8 单例也只证明各自记录的运行条件。不同运行的编译缓存状态可能影响用时，不据这些时长给出性能结论。测试期间源码保持冻结，完整日志、JUnit、coverage 的原文与压缩散列均已归档。

## 第三轮：`5b37725` 无 Torch/Triton 的 CPU 验证

在 WSL Ubuntu 的独立 Python 环境中，确认没有安装 Torch 和 Triton，显式隐藏 CUDA 后完成 **206 passed、14 deselected，16.97 s，退出码 0**。环境为 Python 3.12.3、NumPy 2.5.2、SymPy 1.14.0、pytest 9.1.1；源码仍为 `5b377252cc4452b5ccc48c46ff1ae07a4e5e0e8a`。准确命令、环境变量和产物散列见 [CPU 证据清单](../results/cpu_no_gpu_packages_20260905/manifest.json)，输出见 [CPU 日志](../results/cpu_no_gpu_packages_20260905/pytest.log)。

该命令选择解释器应用、SSA、debugger、单步、GPU 文件中的 CPU 可执行检查及 SSA 管线回归，取消选择 14 个需要真实 GPU 的差分用例，未选择依赖 Torch 的适配测试文件。它与服务器第一轮安装了 Torch、仅隐藏 CUDA 的 **209 passed、15 deselected** 不同，源码、依赖和范围均不同，不能相加或按数量推断回归。

首次尝试广泛选择测试时得到 **207 passed、23 setup errors，21.14 s，退出码 1**，缺少 Torch/Triton 使 GPU 和 Torch 适配 fixture 无法建立。该发现的 [日志](../results/cpu_no_gpu_packages_20260905/discovery.log)和 [manifest](../results/cpu_no_gpu_packages_20260905/discovery_manifest.json)以 `discovery` 前缀保留，不是成功运行；后续按 CPU NumPy 执行合同明确选择范围。发现阶段 manifest 中的原名 `pytest.log`/`pytest.xml` 在归档目录分别对应 `discovery.log`/`discovery.xml`，内容散列保持不变。206 项通过不代表在无 Torch 环境验证了 PyTorch CPU Tensor 适配，也不代表原仓库完整套件通过。

所有时间都是对应进程的实际时长，不是解释器性能基准。各轮测试可能重叠，不累计为独立测试总数。尚未向上游创建 PR。

## 第四轮：独立副本的演示与默认 pass 验证

在 `5b37725` 基础上独立新增演示和测试，开发时保持解释器核心及当时全量运行的服务器源码不变。服务器隔离副本隐藏 CUDA 后，演示、debugger、单步和逐默认 pass 回归 **39 passed，3.80 s**；其中包含四类应用在 Triton/CUDA 两条默认管线中的 8 个 CPU 用例。Sphinx HTML 构建、脚本式单步演示及新进程独立回放均返回 0。准确修改文件的 SHA-256、命令与日志见 [本轮清单](../results/interpreter_debug_replay_20260905/manifest.json)。这些是 CPU 验证，不是新增 GPU 或 A100 对照；后续改进提交不能继承 `5b37725` 的 591 项全量通过记录。

演示现在同时保存正确 reference 和注入常量错误的 candidate；独立回放核对 NumPy 参考，并再次观察 `entry:0:arith.constant` 的差异。测试还用正确 SSA 替换 candidate，要求回放返回非零，防止只导出正确程序却宣称复现故障。[完整演示包与步骤](../results/interpreter_debug_replay_20260905/README.md)已归档。该例明确属于教学故障注入，回放不声称重新执行了 pass 定位。

新增默认管线测试直接使用 `default_pipeline()` 的 pass 对象，逐阶段与经过 NumPy 校验的原始 SSA 比较，再核对标准 `lower_for_target()` 的最终 SSA。它覆盖 int32 逐元素、float32 广播、float32 行归约和 bool 输出，不能据此声称覆盖所有 pass 分支。文档示例同时补齐 `Context` 必填参数。当前改动通过 Ruff check、130 个文件的 format check 与贡献风格检查；后续整合仍需保持测试与源码版本的对应关系。

## 4090 环境与比较方法

GPU 为 RTX 4090、compute capability 8.9；Python 3.12.3、NumPy 1.26.4、SymPy 1.13.1、PyTorch `2.6.0a0+ecf3bae40a.nv25.01`、Triton 3.1.0、CUDA 12.8。后续完整测试另使用 pytest 8.1.1、pytest-cov 7.1 和 TileLang 0.1.14。GPU 报告逐项保存实际设备、dtype、形状、种子、pass 列表和发射前 SSA 的 SHA-256。

同一个 arrangement/application 经共享 frontend 与 SSA 管线生成程序，再分别比较：独立 NumPy 期望值、原始 frontend SSA 的 CPU 解释结果、目标默认 pass 之后的 SSA 解释结果，以及真正发射到 Triton GPU 的结果。默认 pass 包含 `ssa.canonicalize`、`ssa.analyze_effects`、`ssa.select_schedule`、`ssa.triton.optimize_schedule` 和 `ssa.decompose_linalg`；没有在失败时退回原始程序冒充 pass 后验证。

浮点采用 `rtol=1e-3, atol=1e-3`，int32 与 bool 完全相等。14 项覆盖逐元素、尾块掩码、向量/行/列广播、行归约、比较、if/for 两条分支、softmax、有符号整数向下取整除法与余数。每项检查 GPU 输出前后保护区未改写，输入未被修改。浮点最大绝对差出现在行归约，为 `1.9073486328125e-6`；整数和布尔差分为 0。

## 第一轮专项复现命令

以下用于复现 `76ca646` 专项，不是所有历史运行的逐字命令。兼容性与无 Torch/Triton CPU 运行的完整命令以各自 manifest 为准；准备下一轮验收见 [验收及提交说明](cpu_interpreter_acceptance.md)。在对应提交的仓库根目录、项目依赖齐全的 Python 环境中运行。设置 `PYTHONPATH=src` 是为了使用本次 checkout 的源码；禁用额外 pytest 插件以保持与实测条件一致。

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

```

独立 GPU 脚本在无可用 CUDA 时报告 `UNVERIFIED` 并返回 2，有差分失败返回 1，全部通过才返回 0。CPU 测试机器上装有 PyTorch，但 CUDA 已隐藏；这份记录不能说成“未安装 PyTorch 的完整套件运行”。

## 从失败中修复了什么

- 第一轮真实 GPU 对照有 4 项失败：Triton 3.1 不支持当时生成的三个操作数 `and`，以及广播输入错误地使用全局输出 program 索引，导致矩阵除第一行外的广播错误。修复为嵌套二元条件及输入自身的地址/掩码映射，并补充同行数的单例行、单例列和尾块回归。
- 审查发现 trace 观察可能额外读取已被 mask 屏蔽的越界位置。修复为只读取活动地址，增加 trace 打开/关闭一致性与保护区验证。
- 调试器导出回放修复了数组 strides 和可写权限保留，避免把非连续视图的布局问题在回放时“变没了”。
- 有符号整数 `//` 与 `%` 统一为向负无穷取整及与除数同号的余数语义，并加入混合正负整数尾块的真实 GPU 对照。
- 旧上游 `test_aot.py` 的 stream 上下文写法改为 `torch.cuda.stream(torch.cuda.Stream(...))`，兼容当前 PyTorch。保留原测试断言，没有放宽误差阈值。
- `5b37725` 修复程序域重排、不规则展开与后端能力接口的兼容性问题，新增 6 项 CPU 回归；先完成 254 passed、1 skipped 定向验证，随后完整重跑得到 591 passed、2 skipped。两轮范围分别记录，不相加。

## 当前边界与下一步

直接 `linalg.dot`/`linalg.matmul` 和受限的单逻辑 program 标量分解已有 CPU 支持；多 program 分块标量分解等场景仍明确不支持。**当前 GPU runner 未纳入优化后 dot 的对照**，不能把这 14 项通过写成“所有矩阵乘法路径通过”。完整边界见 [CPU 解释器文档](source/cpu_interpreter.rst)。

演示与测试改进 `4a680a6` 已在完整测试取证之后整合。与 `5b37725` 相比，解释器核心、依赖及测试配置没有变化；修改文件的 Git 内容与先前39项定向验证的散列一致，因此不重复整轮4090运行，也不改写旧证据的源码SHA。下一步在真实 A100 上按冻结源码运行专项报告和完整套件，再结合设计说明准备向 `InfiniTensor/ninetoothed:master` 提交 PR。上游评审、合并与训练营评选均尚未完成。

## 历史代码与文档检查

`c9de134` 整理文档和风格时，新增及修改的 Python 文件通过 Ruff 0.16.6 的全库 check、format 检查（125 个文件）和仓库贡献风格检查。纯格式调整的 AST 保持一致；另按贡献规则调整了 17 条诊断消息的大小写或标点，相关回归为 78 passed、14 项 GPU 用例取消选择，计算与控制逻辑未变。当时服务器全量固定在 `76ca646`；后续已完成的完整重跑固定在 `5b37725`。各历史检查不自动覆盖后续提交。

服务器 Sphinx HTML 构建成功。最初缺少系统 `python3-tk`，补齐后通过；构建报告的两个标题下划线长度警告已在本次文档中修正。源码 doctest 收集没有发现用例（退出码 5），不计为新增通过的测试。正式 PR 还需使用仓库规定的 kebab-case 分支名，并附实际 pytest 输出。

完整套件早期有一项 TileLang 重载测试因缺少可选依赖失败；保持 NumPy 1.26.4 和原有 PyTorch，补装 TileLang 0.1.14 与 ml-dtypes 0.5.4 后，该原测试单独复测 1 passed。原全量日志保留失败，不把这一复测改写成已经全量通过。

初轮原始记录的 SHA-256 与字节数见 [初轮证据清单](../results/interpreter_rtx4090_manifest.json)；后续记录分别见 [兼容性清单](../results/rtx4090_compatibility_20260905/manifest.json)、[无 GPU 包 CPU 清单](../results/cpu_no_gpu_packages_20260905/manifest.json)和 [`5b37725` 完整运行清单](../results/full_suite_rtx4090_5b37725/manifest.json)。`results/.gitattributes` 禁止原始记录的自动换行转换。每项结论只适用于相应的提交、依赖与命令。
