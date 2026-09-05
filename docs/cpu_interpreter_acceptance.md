# CPU 参考解释器：验收与提交说明

本说明整理截至 2026-09-05 的已有证据和剩余验收步骤。源码 `5b377252cc4452b5ccc48c46ff1ae07a4e5e0e8a` 在 RTX 4090 完整运行得到 **591 passed、2 skipped，6036.09 s，退出码 0**，已保存日志、JUnit 与 coverage。独立副本演示改进的 39 项 CPU 回归另行记录；后续提交须重新标注源码 SHA 与验证范围，不能沿用 `5b37725` 全量通过结论。当前没有 A100 验证记录，也尚未创建上游 PR。

项目通过共享 arrangement/frontend/SSA 管线执行 NumPy 参考语义，帮助检查编译变换后的结果。设计、接口、支持操作和限制见 [CPU 解释器文档](source/cpu_interpreter.rst)，分轮结果与环境见 [验证报告](cpu_interpreter_validation_4090.md)，加分项计划见 [差距计划](excellence_gap_plan.md)。

## 验收要求与证据入口

下表依据训练营九齿任务的 2026-09-05 阅读快照。测试文件与用例名称是审查入口，列出实现或测试不等于该项已在 A100 上验收。

| 要求 | 实现或自动化测试入口 | 现有证据与边界 |
|---|---|---|
| 复用实际 SSA，NumPy CPU 执行 | `interpret(..., backend="triton")`、`interpret_program`；`test_high_level_backend_option_executes_the_real_pass_pipeline` | 执行目标默认管线输出；不支持时显式失败，不回退 frontend SSA 或 GPU |
| 五类 application | [应用测试](../tests/test_interpreter_applications.py)：`test_elementwise_and_nondivisible_tail`、`test_broadcast_reuses_bias_for_each_row`、`test_row_reduction_ignores_padded_lanes`、`test_nested_if_and_for_carry_values` | 分别覆盖逐元素、广播、非整除尾块、行归约、分支/循环；尾块也是独立验收类别 |
| float32、int32、bool | 同一应用测试文件中的 dtype 参数与 `test_bool_results_are_exact`；[GPU 用例](../tests/test_interpreter_gpu.py) 的 `GPU_CASES` | float32 使用 `rtol=1e-3, atol=1e-3`；整数与布尔完全相等 |
| 无可见 CUDA 的独立运行 | `test_cpu_path_does_not_import_gpu_backends`；[无 Torch/Triton CPU 清单](../results/cpu_no_gpu_packages_20260905/manifest.json) | `5b37725` 的 206 passed、14 deselected；未选择 Torch 适配文件，不等于全仓库无依赖运行 |
| 至少三个程序默认 pass 前后一致并与 A100 差分 | `test_gpu_fixtures_match_oracle_before_and_after_lowering_on_cpu`、[逐阶段测试](../tests/test_interpreter_default_pipeline.py)、`run_gpu_case`、[GPU runner](../scripts/verify_interpreter_gpu.py) | 初轮 4090 JSON 记录 8 个程序、14 个用例的四方比较；新逐阶段测试在 CPU 上执行四类应用 × 两个后端的真实默认管线。A100 仍待运行 |
| mask 与 trace 正确；unsupported 显式失败 | [SSA 测试](../tests/test_interpreter_ssa.py) 中 masked load/store、trace 一致性、unsupported operation/dtype 用例 | mask 排除地址不解引用；诊断包含 operation 位置。支持边界见接口文档 |
| 单步、断点、watch、program/opcode 过滤 | [单步测试](../tests/test_interpreter_step_debugger.py)、`StepDebugger`、[演示](cpu_interpreter_demo.py) | 暂停发生在操作完成之后；交互观察不能代替计算正确性检查 |
| 首个错误 pass/operation 与差分复现 | [调试器测试](../tests/test_interpreter_debugger.py)、`check_passes`、`compare_programs`、`export_reproducer` | 结构和执行次序可对齐时定位对应 operation；结构重写缺少来源映射时不猜位置。故障注入与真实历史缺陷必须分别标注 |
| 扩展接口与完整应用 | `handlers` 扩展测试、softmax 与直接/受限分解 dot 测试 | 多 program 分块标量 dot 仍拒绝；GPU runner 排除优化后 dot。没有自动样例缩减承诺 |
| 原测试、风格、文档、主分支合并 | 完整 pytest、Ruff、贡献风格检查、Sphinx、上游审查 | `5b37725` 4090 全量 591 passed、2 skipped；两个跳过项需双卡。独立副本另有 39 项 CPU 回归与风格/文档检查。旧失败及未定原因的 SIGSEGV 均保留；上游合并未完成 |

## 复查已完成的运行

同一测试可能在多个范围中出现。`224` 包含 `14` 项 GPU 对照，`209` 为旧提交在安装 Torch 的服务器上隐藏 CUDA 的子集，`206` 为 `5b37725` 在无 Torch/Triton 环境的选定范围，`254` 为该提交的兼容性定向回归，`591` 为该提交完整运行的通过数（另有 2 skipped），`39` 为独立副本的 CPU 回归，不能累计。

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

## 剩余硬件验收

`5b37725` 的 RTX 4090 全量与原文产物已归档，结果为 591 passed、2 skipped、退出码 0。两项跨设备测试因只有一张 GPU 而跳过，不构成跨设备验证。演示与测试改进 `4a680a6` 已在完整测试取证之后整合，其核心、依赖和测试配置未变，修改文件与39项CPU验证记录的散列一致，因此不重复整轮4090运行；39项CPU检查与 `5b37725` 完整GPU结果分别保留。早先SIGSEGV的原因仍未确定，本轮成功不解释该异常。

获得实际 A100 后，在冻结源码上记录 GPU 名称/计算能力、软件版本、源码 SHA、完整命令、输入种子、dtype、布局、默认 pass、发射 SSA 散列、误差与 skip 原因，再运行解释器专项、独立 GPU 报告及完整兼容性套件。复现参考命令如下，`RUN_DIR` 必须指向本次运行的新目录，依赖包含 pytest-cov，且 CUDA 可见：

```bash
export PYTHONPATH=src
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pytest -q --color=no tests/test_interpreter*.py tests/test_ssa*.py
python scripts/verify_interpreter_gpu.py --device 0 \
  --report "$RUN_DIR/interpreter_gpu.json"
python -m pytest -p pytest_cov --color=no -v --tb=short -ra \
  --doctest-modules --junitxml="$RUN_DIR/full-suite.xml" \
  --cov=ninetoothed --cov-report="xml:$RUN_DIR/coverage.xml"
```

上述命令是未来验收步骤，不是已完成的 A100 结果。单卡导致的双卡用例跳过应逐项说明；A100 分区的接受情况须依据训练营要求另行核对。dot、跨结构 operation 来源定位和上游合并等缺口不会因更换 GPU 自动关闭。

## PR 与官网提交材料

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

只有证据齐全后才将相应项标为完成。主分支合并和优秀学员评选由维护者及训练营决定。
