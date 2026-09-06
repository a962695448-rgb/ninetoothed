# CPU 参考解释器：验收与提交说明

当前功能源码为 **`f35fb51b16a52392e7ee92b3a3c15622305d428b`**，新增多 M/N 输出 tile 标量 dot 和显式 SSA 来源记录，修改了 runtime、默认 pass、发射器、IR 与测试。预备 CPU 组合为 **171 passed、23.72 s**；广范围 CPU 选择已得到 **1 failed、294 passed、15 deselected，35.42 s，退出码 1**，原测试对整个 metadata 的字符串检查误命中 origins 中保留的操作名，兼容修正需独立复验。**15 个 GPU 用例已准备，但新 GPU 尚未运行，也没有新的 A100 结果。**本轮功能变更不能继承历史 A100 600 项通过结论。完成全部约定优化及验证后先由用户验收，之后才处理上游 PR 与官网提交；当前不创建或发布 PR，不执行官网提交。

历史验证：源码 **`82592b8f6de65052e4258fdd6067956d4ede18c3`** 曾在实际 A100-SXM4-40GB 上完成完整测试：**600 passed、2 skipped，450.25 s（0:07:30），退出码 0，无 failures/errors**。两个 skip 均要求同机至少双卡，详见 [该轮完整清单](../results/full_suite_a100_82592b8/manifest.json)与 [归档说明](../results/full_suite_a100_82592b8/README.md)。b5/377 失败历史及 14/180/16/77 各范围按原源码保留，不累计，成功重跑不确定旧 squeeze 失败的唯一根因。

项目通过共享 arrangement/frontend/SSA 管线执行 NumPy 参考语义，帮助检查编译变换后的结果。设计、接口、支持操作和限制见 [CPU 解释器文档](source/cpu_interpreter.rst)，最新硬件记录见 [A100 报告](cpu_interpreter_validation_a100.md)，历史分轮结果见 [4090 报告](cpu_interpreter_validation_4090.md)，后续实现与验证见 [实施与优化计划](implementation_optimization_plan.md)。

## 验收要求与证据入口

下表依据 2026-09-06 重新只读核对的官方任务内容，按功能、验证条件与支持边界组织。测试文件与用例名称是审查入口，列出实现或 CPU 测试不等于当前版本已在 A100 上验收。

| 要求 | 实现或自动化测试入口 | 现有证据与边界 |
|---|---|---|
| 复用实际 SSA，NumPy CPU 执行 | `interpret(..., backend="triton")`、`interpret_program`；`test_high_level_backend_option_executes_the_real_pass_pipeline` | 执行目标默认管线输出；不支持时显式失败，不回退 frontend SSA 或 GPU |
| 五类 application | [应用测试](../tests/test_interpreter_applications.py)：`test_elementwise_and_nondivisible_tail`、`test_broadcast_reuses_bias_for_each_row`、`test_row_reduction_ignores_padded_lanes`、`test_nested_if_and_for_carry_values` | 分别覆盖逐元素、广播、非整除尾块、行归约、分支/循环；尾块也是独立验收类别 |
| float32、int32、bool | 同一应用测试文件中的 dtype 参数与 `test_bool_results_are_exact`；[GPU 用例](../tests/test_interpreter_gpu.py) 的 `GPU_CASES` | float32 使用 `rtol=1e-3, atol=1e-3`；整数与布尔完全相等 |
| 无可见 CUDA 的独立运行 | `test_cpu_path_does_not_import_gpu_backends`；[无 Torch/Triton CPU 清单](../results/cpu_no_gpu_packages_20260905/manifest.json) | `5b37725` 的 206 passed、14 deselected；未选择 Torch 适配文件，不等于全仓库无依赖运行 |
| 至少三个程序默认 pass 前后一致并与 A100 差分 | `test_gpu_fixtures_match_oracle_before_and_after_lowering_on_cpu`、[逐阶段测试](../tests/test_interpreter_default_pipeline.py)、`run_gpu_case`、[GPU runner](../scripts/verify_interpreter_gpu.py) | 历史 b5 的 8 程序、14 项 A100 四方比较和 180 项专项分别留存；f35 已准备 15 个用例，新的真实 GPU 及 A100 对照尚未完成 |
| mask 与 trace 正确；unsupported 显式失败 | [SSA 测试](../tests/test_interpreter_ssa.py) 中 masked load/store、trace 一致性、unsupported operation/dtype 用例 | mask 排除地址不解引用；诊断包含 operation 位置。支持边界见接口文档 |
| 单步、断点、watch、program/opcode 过滤 | [单步测试](../tests/test_interpreter_step_debugger.py)、`StepDebugger`、[演示](cpu_interpreter_demo.py) | 暂停发生在操作完成之后；交互观察不能代替计算正确性检查 |
| 首个错误 pass/operation 与差分复现 | [来源测试](../tests/test_interpreter_provenance.py)、`check_passes`、`compare_programs`、`Operation.origins`、`export_reproducer` | 结构和完整 trace 次序对齐时定位对应 operation；结构变化给已声明的原 SSA 候选集合，不是 Python 行号或唯一因果点。未知映射保持未知，故障注入与真实历史缺陷分别标注 |
| 扩展接口与完整应用 | `handlers`、softmax、[多 program dot 测试](../tests/test_interpreter_matmul.py) | CPU 已实现 M/N 输出 tile 内完整 K 的 rank-2 标量 dot，覆盖 F32/I32 和独立对齐 strides；split-K、别名、字节重叠及混合副作用等拒绝写前。新 float32 GPU dot 待跑，无 Tensor Core 或自动样例缩减承诺 |
| 原测试、风格、文档、主分支合并 | 完整 pytest、Ruff、贡献风格检查、Sphinx、用户验收及后续外部审查 | 825 的 600 passed、2 skipped 是历史结果；171 只是预备 CPU 组合，f35 广范围 CPU 的 1 项失败保留并待兼容修正复验。两项双卡场景未测，上游合并未完成 |

官方 CPU-only 阶段要求 CUDA 不可见且解释器不导入或调用 CUDA 执行路径；GPU 阶段使用 A100。原生 CPU 代码生成、CPU 性能、warp/block 调度、shared memory 和 GPU race 模拟不属于解释器目标。Atomics、间接指针、多设备、随机数和 float8 等未支持语义必须显式报错，不能静默回退或把原仓库 GPU 测试能力当作 CPU 解释器支持。

## 复查已完成的运行

同一测试可能在多个范围中出现。`224` 包含 `14` 项 GPU 对照，`209` 为旧提交在安装 Torch 的服务器上隐藏 CUDA 的子集，`206` 为 `5b37725` 在无 Torch/Triton 环境的选定范围，`254` 为该提交的兼容性定向回归，`591` 为该提交完整运行的通过数（另有 2 skipped），`39` 为独立副本的 CPU 回归，不能累计。

- [最终 A100 完整清单](../results/full_suite_a100_82592b8/manifest.json)与 [原文归档](../results/full_suite_a100_82592b8/raw-full.tar.gz)：825，600 passed、2 skipped、450.25 s、退出码 0；JUnit 602 tests、0 failures/errors、2 skipped，前后源码一致且 tracked clean。完整成功是独立运行结论，不覆盖此前失败或给出旧 squeeze 的唯一根因。
- [A100 真实 GPU 清单](../results/a100_20260906/gpu-report/manifest.json)与 [GPU JSON](../results/a100_20260906/gpu-report/interpreter_gpu_validation.json)：`b5a3206`，14/14、8 程序、9 类别、退出码 0；设备与计算能力确认实际 A100。JSON 中旧静态模板提示的含义见 [A100 说明](cpu_interpreter_validation_a100.md)，原文及散列保持不变。
- [A100 专项清单](../results/a100_20260906/specialist/manifest.json)：180 passed、7.95 s、退出码 0，包含 GPU、应用、SSA、调试、逐 pass、回放与 Torch 适配范围。首次 [smoke 失败](../results/a100_20260906/smoke/manifest.json)与 [修正 libcuda 路径后 1 passed](../results/a100_20260906/smoke-libcuda/manifest.json)分别保留；smoke 属于 14 项，14 项又包含于 180 项，不相加。
- [A100 初轮完整清单](../results/full_suite_a100_b5a3206/manifest.json)与 [原文归档](../results/full_suite_a100_b5a3206/raw-full.tar.gz)：`b5a3206`，16 failed、584 passed、2 skipped、2737.65 s、退出码 1；JUnit 602 条目。8 项 dim 1 在 PyTorch 参考转换抛 NotImplementedError，8 项 dim 2 在 nested 输入构造抛 RuntimeError，不能把未完成比较的原用例视为 kernel 正确证明。
- [jagged 修复后定向清单](../results/jagged_reference_recheck_a100_377daec/manifest.json)：`377daec`，16 passed、8.83 s、退出码 0；原参数与容差保留，输入使用 packed values/offsets/dim，dense 参考在 kernel 前独立生成，expand 检查完整 values 与 offsets。这 16 项在第二轮全量中也通过，但该全量另有 squeeze 失败。
- [A100 第二轮完整清单](../results/full_suite_a100_377daec/manifest.json)：`377daec`，1 failed、599 passed、2 skipped，752.11 s、退出码 1；唯一失败为 `test_squeezing_the_innermost_level[1024-128-10-cuda]` 的完整矩阵 allclose。原失败没有保存分配器状态及全部索引，不能从后续控制实验确定其唯一根因。
- [squeeze 受控机制诊断](../results/squeeze_fixture_diagnosis_a100_377daec/README.md)：原 fixture 源码未改，人工 seed 2026；自然分配 PASS、NaN/唯一索引 FAIL、有限值/唯一索引 PASS。NaN 控制逐位差异为 0，129792 个 NaN 全在未写区；这不是旧现场精确重放，不增加正式测试通过数。
- [generation 文件复验清单](../results/generation_reference_recheck_a100_82592b8/manifest.json)：`82592b8`，77 passed、19.66 s、退出码 0；只用有限非零输出初值与 randperm 唯一随机目标稳定输入，kernel、src、参数、全矩阵比较和原容差不变。其后最终全量 600 passed、2 skipped 单独归档，两个范围不累计。
- [初轮 4090 清单](../results/interpreter_rtx4090_manifest.json)：`76ca646` 专项、隐藏 CUDA 子集与独立 GPU JSON。
- [后续 4090 清单](../results/rtx4090_compatibility_20260905/manifest.json)：旧全量失败、新提交定向回归、SIGSEGV、两个 FP8 单例；各条含源码 SHA、命令参数、进程返回值、原文与压缩文件散列。
- [`5b37725` 完整运行清单](../results/full_suite_rtx4090_5b37725/manifest.json)：591 passed、2 skipped，退出码 0；JUnit 593 条目、0 errors、0 failures、2 skipped。归档时 HEAD 与测试 SHA 一致、已跟踪文件无修改。全库 coverage XML 的 line-rate 为 86.76%（9178/10578 行），不是解释器专项覆盖率；跳过项和核验方法见 [归档说明](../results/full_suite_rtx4090_5b37725/README.md)。
- [独立副本改进清单](../results/interpreter_debug_replay_20260905/manifest.json)：基于 `5b37725` 的修改文件散列、39 项 CPU 回归、Sphinx、单步演示与独立差分回放；它不等于后续改进提交的完整 GPU 测试。
- [无 GPU 包 CPU 清单](../results/cpu_no_gpu_packages_20260905/manifest.json)：`5b37725` 的包存在性、版本、明确测试选择、日志及 JUnit 散列。[首次范围发现清单](../results/cpu_no_gpu_packages_20260905/discovery_manifest.json)与 [日志](../results/cpu_no_gpu_packages_20260905/discovery.log)记录 207 passed、23 setup errors、退出码 1；这些错误来自缺失的 GPU/Torch 依赖，不删除、不计入成功范围。发现阶段 manifest 中原名 `pytest.log`/`pytest.xml` 分别归档为 `discovery.log`/`discovery.xml`，内容散列保持不变。

复查压缩日志时先核对压缩文件 SHA-256，再解压并核对原文 SHA-256 和字节数。manifest 内的绝对路径记录当时环境；在另一机器复跑应调整路径并形成新的 manifest，不能覆盖旧记录。

无 Torch/Triton 的 CPU 复现选择如下；这是可移植命令写法，当次准确命令见 manifest。须在相应源码版本的仓库根目录运行，使用含 NumPy、SymPy、pytest 的独立环境。

```bash
export PYTHONPATH=src
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export CUDA_VISIBLE_DEVICES=""
python -m pytest -q --color=no -ra --tb=short \
  tests/test_interpreter_applications.py \
  tests/test_interpreter_debugger.py \
  tests/test_interpreter_gpu.py \
  tests/test_interpreter_ssa.py \
  tests/test_interpreter_step_debugger.py \
  tests/test_ssa_application_lowering.py \
  tests/test_ssa_first_backend_lowering.py \
  tests/test_ssa_pass_pipeline.py \
  tests/test_ssa_program_domain_regressions.py \
  tests/test_ssa_validation.py \
  -k 'not test_cpu_interpreter_matches_actual_triton_gpu'
```

`tests/test_interpreter_gpu.py` 同时包含 CPU 可执行的发射/布局验证和真实 GPU 测试，因此保留该文件并明确取消选择后者。包元数据仍依赖 Triton；此处是 source checkout 的 CPU 验证路径，不是已发布的 CPU-only wheel 安装说明。

## A100 最终通过与提交对应关系

`5b37725` 的 RTX 4090 全量与原文产物已归档，结果为 591 passed、2 skipped、退出码 0。两项跨设备测试因只有一张 GPU 而跳过，不构成跨设备验证。演示与测试改进 `4a680a6` 已在完整测试取证之后整合，其核心、依赖和测试配置未变，修改文件与39项CPU验证记录的散列一致，因此不重复整轮4090运行；39项CPU检查与 `5b37725` 完整GPU结果分别保留。早先SIGSEGV的原因仍未确定，本轮成功不解释该异常。

实际设备为 A100-SXM4-40GB、compute capability 8.0、MIG Disabled。`b5a3206` 的 14 项 GPU 差分与 180 项专项通过，但同一源码完整测试最终失败；这些范围分别记录。最初项目内 libcuda 链接修正指向真实 ELF64 驱动库并设置库路径，该阶段失败也保留；它与后续 jagged 测试参考修复是两个不同问题。环境、原始 argv、结果及来源见 [A100 报告](cpu_interpreter_validation_a100.md)。

`377daec6242864a920de43a55523ac3d5f582648` 只改 jagged 输入和参考，16 项定向及全量中的同一子范围均通过；但第二轮完整结果仍是 FAIL，唯一 squeeze allclose 失败保留原始记录。受控诊断说明未写区 NaN 可导致逐位一致而原 allclose 失败，却没有重建旧现场的分配器和索引，不能确定旧失败的唯一根因；详见 [A100 报告](cpu_interpreter_validation_a100.md)。

`82592b8f6de65052e4258fdd6067956d4ede18c3` 仅改两条 generation 测试输入语句及注释：输出初始化为有限非零值 -123，目标行用 randperm 保证随机且唯一。kernel、`src/`、参数、全矩阵比较、原容差和依赖不变，未改为 `equal_nan=True`。generation 文件 77 项和后续完整测试 600 passed、2 skipped 均已取得独立证据；这些成功不确定旧失败的唯一根因。当前工作是优化后 dot、跨结构 operation 来源定位等约定实现及验证，完成后先交用户验收。

归档提交 `086f148b40a7ac057f9184ecfbfccef84eb4037e` 仅修改当时的 `docs/`、`results/`，其代码、测试、依赖与 CI 和 825 相同，所以该批资料引用 825 的原运行；086 本身不是另一轮实测。当前 f35 则有 runtime/pass/emitter/provenance 功能与测试变更，必须独立完成广范围 CPU、实际 GPU 和新的 A100 验证，不能继承 825 的 600 项结果。新 GPU 的预备用例及合同见 [实施与优化计划](implementation_optimization_plan.md)。

## PR 与官网提交材料

以下仅为材料准备要求。完成全部约定优化与验证并通过用户验收后，才处理上游 PR 和官网提交；当前不创建或发布 PR，不执行官网提交。

正式 PR 使用已推送的 kebab-case 分支 `add-cpu-reference-interpreter`，目标为 `InfiniTensor/ninetoothed:master`。候选标题为 `Add a CPU reference interpreter and differential debugger`。中文 commit 保留既有历史；提交前重新核对本地和远端源码 SHA。

PR 正文应按以下顺序组织：

1. 说明难以核对优化前后语义的问题，以及现在如何执行实际 SSA、观察差异和导出复现材料。
2. 链接设计、支持操作、输入合同、限制和可运行演示；故障注入示例明确标注，真实缺陷附原始失败与修复证据。
3. 列出每次验证的源码 SHA、实际设备、依赖、命令、范围与原始证据，直接粘贴最终 pytest 输出。正文必须有 `` `pytest` output: `` 后接非空 fenced code block。
4. 单独写清 A100、完整套件、双卡跳过、优化后 dot、跨结构定位和合并状态；未完成项不得用计划或参数化测试数量替代。

训练营官网的实际提交字段需按提交时页面填写。先备好 fork/正式分支链接、准确源码 SHA、上游 PR 链接（创建后才填写）、设计与使用文档、证据索引和简短演示步骤。公开材料不包含租赁实例标识、私有服务 URL 或个人求职材料。

当前工作流的检查范围如下：

| 工作流 | 实际检查 | 提交时的解释 |
|---|---|---|
| `contributing.yml` | PR title、head branch、非空 pytest 输出区块 | 不核查日志真假，不遍历历史 commit；元数据通过不等于测试通过 |
| `pytest.yml` | self-hosted NVIDIA runner 上运行含 doctest 和 coverage 的套件 | fork 来源的 PR job 被条件跳过；push 需要对应 runner。附实测证据，由维护者安排其 runner，不能把 skipped 记为 GPU 通过 |
| `ruff.yml` | 贡献风格、Ruff check 和 format check | 对准备提交的实际改动重新检查 |
| `sphinx.yml` | push 时构建文档并上传产物；master 才部署 | 当前提交的文档构建应留结果，旧构建成功不自动覆盖新增内容 |

只有实现与证据齐全后才将相应项标为完成，并交由用户验收。后续外部审查与主分支合并由相应维护者处理，按实际状态记录。
