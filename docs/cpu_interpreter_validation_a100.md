# CPU 参考解释器：NVIDIA A100 验证记录

更新日期：2026-09-06（Asia/Shanghai）。初轮源码 **`b5a3206f8351e5a138d16ee13f6d6ef9c620044b`** 的 A100 完整测试已结束：**16 failed、584 passed、2 skipped，2737.65 s，退出码 1**。随后提交 **`377daec6242864a920de43a55523ac3d5f582648`** 只修正 jagged 测试的输入构造与独立参考，A100 定向回归为 **16 passed、8.83 s、退出码 0**；该提交的新完整测试已启动，仍为 **RUNNING，没有最终结论**。

初轮 `b5a3206` 已完成的 **14/14 真实 Triton GPU 差分**和 **180 passed 解释器专项**仍是该源码上的有效 PASS 记录，不改写到 `377daec`，也不与定向 16 项或完整测试数字相加。以下结果不表示整项目官方验收、上游合并或优秀学员评选完成。历史 RTX 4090 结果见 [4090 报告](cpu_interpreter_validation_4090.md)，验收对照见 [提交说明](cpu_interpreter_acceptance.md)。

## 按运行保留结果

原目录按 UTC 命名为 `a100-20260905T165639Z`，对应北京时间 2026-09-06。首次失败与环境修正后的结果分别归档，没有覆盖旧日志。

| 范围 | 实际结果 | 原始证据 |
|---|---|---|
| 首次逐元素 smoke | 1 failed，2.18 s；退出码 1；Triton 编译驱动辅助模块时无法链接兼容的 `-lcuda` | [manifest](../results/a100_20260906/smoke/manifest.json)、[失败日志](../results/a100_20260906/smoke/validation.stdout.log) |
| 修正 libcuda 路径后，同一 smoke | 1 passed，2.31 s；退出码 0 | [manifest](../results/a100_20260906/smoke-libcuda/manifest.json)、[日志](../results/a100_20260906/smoke-libcuda/validation.stdout.log) |
| 独立真实 GPU 差分报告 | 14/14 PASS；8 个程序、9 类用例；退出码 0 | [manifest](../results/a100_20260906/gpu-report/manifest.json)、[GPU JSON](../results/a100_20260906/gpu-report/interpreter_gpu_validation.json) |
| 解释器专项 | 180 passed，7.95 s；退出码 0 | [manifest](../results/a100_20260906/specialist/manifest.json)、[日志](../results/a100_20260906/specialist/validation.stdout.log)、[JUnit](../results/a100_20260906/specialist/junit.xml) |
| `b5a3206` 初轮完整仓库测试 | 16 failed、584 passed、2 skipped，2737.65 s；退出码 1 | [完整清单](../results/full_suite_a100_b5a3206/manifest.json)、[原文归档](../results/full_suite_a100_b5a3206/raw-full.tar.gz)、[说明](../results/full_suite_a100_b5a3206/README.md) |
| `377daec` jagged 定向回归 | 原 16 个参数用例全部通过：16 passed，8.83 s；退出码 0 | [定向验证清单](../results/jagged_reference_recheck_a100_377daec/manifest.json)、[说明](../results/jagged_reference_recheck_a100_377daec/README.md) |
| `377daec` 新完整仓库测试 | RUNNING；尚无最终退出状态与汇总 | 完成后单独归档，不用 16 项定向通过替代全量 |

smoke 是 14 项 GPU 差分中的一个用例；180 项专项又包含这 14 项，以及应用、SSA、调试、逐默认 pass、回放和 Torch 适配测试。初轮完整测试与这些专项重叠，新提交的 16 项又是 jagged 定向复验；各版本和范围均不能相加。这里的时长是各次 pytest 汇总时长；外部 runner 计时还包含进程启动等开销，二者不能混用，也不是性能基准。

## 初轮 16 项失败与测试参考修复

`b5a3206` 完整运行的 JUnit 为 602 条目、16 failures、0 errors、2 skipped，与 **584 + 16 + 2 = 602** 一致。原始 stdout/stderr、runner manifest、JUnit、coverage XML/HTML 均保留在 [raw-full.tar.gz](../results/full_suite_a100_b5a3206/raw-full.tar.gz)；原文字节数与 SHA-256 见 [完整清单](../results/full_suite_a100_b5a3206/manifest.json)，失败没有被后续定向通过覆盖。

| 数量 | 原有参数范围 | 实际失败位置 |
|---|---|---|
| 8 | 两个 jagged 测试，`jagged_dim=1`，batch 数 2、3、7、16 | PyTorch 2.5 的参考转换 `to_padded_tensor` 抛出 `NotImplementedError: aten.to_padded_tensor.default` |
| 8 | 同样两个测试，`jagged_dim=2`，同样 batch 参数 | 使用列表创建 jagged nested tensor 时抛出 RuntimeError，输入构造阶段即不支持所需 ragged 维度 |

这 16 项是输入构造/参考计算失败，不是已经完成比较后的数值断言失败；仅凭初轮日志不能宣称 jagged 内核正确。初轮另有两项跨设备 skip，分别要求至少两张设备的 AOT 测试和至少两张 CUDA 设备的 Triton 上下文复用测试；单卡 A100 未验证这两个场景。

修复提交 `377daec6242864a920de43a55523ac3d5f582648` **只修改 `tests/test_jagged.py`**：

- 由原始 batch 显式拼接 packed values、累计 offsets，并传入 `jagged_dim` 构造 nested tensor，使两个 jagged 维度都能建立测试输入。
- `to_padded_tensor` 的 dense 期望值在调用被测 kernel 前，由原始 batch 和 padding 独立填充，不再调用当前环境未实现的参考接口，也不依赖被测输出。
- `expand` 预先生成完整 expected values 和 offsets 副本；运行后比较全部 values，并检查 offsets 未变，不再按被测结果的非零位置筛选比较。

原有 16 个参数组合、比较容差、被测 kernel、`src/` 实现和依赖版本均保留。A100 定向结果为 **16 passed、8.83 s、退出码 0**，证明这些修正后的用例已完成数值比较；新版全量仍需等待最终结果。本次修复没有把 jagged 运行能力新增到 NumPy CPU 解释器，也不改变优化后 dot 等限制。

## 硬件与依赖

硬件信息由 PyTorch runtime 和 `nvidia-smi` 分别记录，见 [GPU 运行环境](../results/a100_20260906/gpu-report/environment.stdout.log)、[设备输出](../results/a100_20260906/gpu-report/nvidia-smi.stdout.log)及 [环境检查](../results/a100_20260906/setup/preflight.json)。

| 项目 | 实测值 |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB，单卡，compute capability 8.0 |
| MIG | Disabled，未使用 MIG 分区 |
| NVIDIA driver | 550.127.05 |
| Python / OS | Python 3.12.7，Linux x86_64 |
| PyTorch | 2.5.0；实际导入版本 `2.5.0+cu124`，CUDA build 12.4 |
| CUDA toolkit | 12.4，nvcc 12.4.99 |
| Triton | 3.1.0 |
| NumPy / SymPy | 2.1.3 / 1.13.1 |
| pytest | 8.1.1 |
| 专项阶段的附加依赖 | pytest-cov 7.1.0、coverage 7.16.0、TileLang 0.1.14；完整清单以该次 manifest 为准 |

初始 smoke/GPU 报告与后续专项准备阶段的可选包不同。完整安装记录见 [packages.txt](../results/a100_20260906/setup/packages.txt)；[pip-check.log](../results/a100_20260906/setup/pip-check.log)记录依赖一致性检查通过。不要将后续添加的包倒写为首次 smoke 的环境。

## 首次 libcuda 链接失败与修正

首次 smoke 已识别 A100，但在 Triton 构建驱动辅助模块时失败。日志显示链接器跳过不兼容的 `/lib/i386-linux-gnu/libcuda.so`，随后报 `cannot find -lcuda`。这是驱动库链接路径问题；该次没有取得 smoke 通过结果，不能只因 `torch.cuda.is_available()` 为真就视为 GPU 执行成功。

本次检查确认真实驱动库 `/usr/lib/x86_64-linux-gnu/libcuda.so.550.127.05` 为 ELF64，在项目目录的 `.driver-libs` 中创建指向它的 `libcuda.so` 符号链接，并将 `TRITON_LIBCUDA_PATH`、`LIBRARY_PATH` 指向该目录。记录见 [环境修正 JSON](../results/a100_20260906/setup/cuda-link-environment-fix.json)。修正未改 NineToothed 源码，没有使用 CUDA stub、替换系统驱动库、调用 sudo 或关闭 TLS 验证。随后同一 smoke 在同一源码 SHA 上通过。

## 差分比较了什么

每项测试复用相同 arrangement/application、dtype、布局与随机种子 2026，比较以下四方：NumPy/Python 独立参考值、原始 frontend SSA 的 CPU 解释结果、实际发射 SSA 的 CPU 解释结果，以及真正运行在 A100 上的 Triton 输出。整数 floor_divide 与 remainder 使用 Python int 精确运算作为参考，其余使用 NumPy。`run_gpu_case` 核对生成器没有使用 Python fallback，并从实际编译产物的 metadata 读取目标 SSA；不是另写一份参考应用替代目标程序。

GPU JSON 的每条记录保存发射 SSA 的 SHA-256 和默认 Triton pass 列表：`ssa.canonicalize`、`ssa.analyze_effects`、`ssa.select_schedule`、`ssa.triton.optimize_schedule`、`ssa.decompose_linalg`。这里验证原始与最终目标 SSA；专项中的逐默认 pass 测试另逐阶段执行真实 pass 对象。两种检查共同提供证据，但不能把 JSON 中记录 pass 名称本身说成所有 pass 分支都已遍历。

另在无Torch/Triton的WSL环境中，复用同一源码的现有fixture和默认管线，独立重建14份SSA原文；每份SHA-256与已记录的A100报告逐项匹配，pass trace也一致。重建前核对了122个相关源码文件与b5 Git blob，文本只做CRLF→LF规范化。见 [重建清单与原文](../results/a100_20260906/reconstructed-ssa/manifest.json)、[匹配日志](../results/a100_20260906/reconstructed-ssa/reconstruction.stdout.log)和 [归档脚本](../results/a100_20260906/reconstructed-ssa/export_a100_ssa_proof.py.txt)。这是独立CPU重建后的哈希核对，不是新增GPU运行，也不是运行当时导出的原文；不增加14个GPU用例的计数。

| 程序 | 用例范围 | GPU 对 NumPy/Python 独立参考的最大绝对差 |
|---|---|---|
| `vector_add` | float32 对齐与尾块、int32 尾块 | 0 |
| `broadcast_add` | 向量、单例行、单例列广播及非整除尾块 | 0 |
| `row_sum` | float32 / int32 行归约 | float32：`1.9073486328125e-6`；int32：0 |
| `comparison` | bool 精确比较输出 | 0 |
| `control_flow` | if/for 的两条分支 | `2.384185791015625e-7` |
| `row_softmax` | float32 softmax，含尾块 | `8.940696716308594e-8` |
| `vector_floor_divide` | int32 混合正负输入及尾块 | 0 |
| `vector_remainder` | int32 混合正负输入及尾块 | 0 |

这 8 个程序形成 9 类用例，包括题目要求的逐元素、广播、非整除 masked load/store、行归约、分支/循环五类 application。float32 使用 `rtol=1e-3, atol=1e-3`，int32 与 bool 完全相等；全报告最大绝对差为 `1.9073486328125e-6`。每项还检查输出前后保护区未改写及输入未修改，JSON 中 `guard_lanes_unchanged` 均为 true。

原 GPU JSON 的 `limitations` 保留一条旧脚本静态模板句：`Results describe the reported GPU; they do not prove A100 validation.`。它不是硬件检测结果，也不是本次检测到非 A100 的结论：同一 JSON 的 `gpu_name`、`compute_capability`，加上 runtime 与 `nvidia-smi`，记录的都是真实 A100。为保持原文及 SHA-256 可核验，不修改该句。本报告据实际硬件字段确认上述用例在 A100 上运行，同时保留“专项通过不等于整项目官方验收”的边界。

## 复现 `b5a3206` 已完成的专项范围

在独立 checkout 中使用测试源码 `b5a3206f8351e5a138d16ee13f6d6ef9c620044b` 和上述依赖。以下是初轮专项 manifest 对应的可移植写法，替换的仅是机器相关路径；原始 argv 与环境变量以各运行 manifest 为准。正在运行的新版全量使用 `377daec`，不要在那个 checkout 中切回旧源码或重复启动全量。

```bash
# 在对应 checkout 的仓库根目录，使用已准备的项目 Python 环境。
PROJECT_ROOT="$(cd .. && pwd)"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
RUN_DIR="$PROJECT_ROOT/runs/a100-replay-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$PROJECT_ROOT/runs"
mkdir "$RUN_DIR"
git rev-parse HEAD
git status --porcelain --untracked-files=no

export PYTHONPATH="$PWD/src:$PWD"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONUNBUFFERED=1
export TRITON_INTERPRET=0
export OMP_NUM_THREADS=2
export MPLBACKEND=Agg
unset PYTEST_ADDOPTS PYTEST_PLUGINS PYTEST_XDIST_WORKER NINETOOTHED_BACKEND

# 使用本次已核对的真实 ELF64 驱动库；已有正确链接时不重复创建。
export TRITON_LIBCUDA_PATH="$PROJECT_ROOT/.driver-libs"
export LIBRARY_PATH="$TRITON_LIBCUDA_PATH"

"$PYTHON" -m pytest --color=no --tb=short -ra -q \
  'tests/test_interpreter_gpu.py::test_cpu_interpreter_matches_actual_triton_gpu[elementwise_float32_aligned]' \
  --junitxml="$RUN_DIR/smoke.xml"

"$PYTHON" scripts/verify_interpreter_gpu.py --device 0 \
  --report "$RUN_DIR/interpreter_gpu_validation.json"

"$PYTHON" -m pytest --color=no --tb=short -ra -q \
  tests/test_interpreter_applications.py \
  tests/test_interpreter_debugger.py \
  tests/test_interpreter_default_pipeline.py \
  tests/test_interpreter_demo.py \
  tests/test_interpreter_gpu.py \
  tests/test_interpreter_ssa.py \
  tests/test_interpreter_step_debugger.py \
  tests/test_interpreter_torch.py \
  --junitxml="$RUN_DIR/specialist.xml"
```

这些命令要求 CUDA 可见，并显式设置 `TRITON_INTERPRET=0`。若在同样缺少 unversioned 驱动链接的环境重建 `.driver-libs`，先确认真实 driver 文件为 ELF64，再以 `mkdir -p "$PROJECT_ROOT/.driver-libs"` 创建项目内目录，并执行 `ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.550.127.05 "$PROJECT_ROOT/.driver-libs/libcuda.so"`；不同机器的真实驱动文件名可能不同，不能改用 toolkit stub。各命令退出状态、stdout/stderr 和源码前后检查应分别保存在新目录，避免覆盖原始证据。

## 修复后定向复现与剩余范围

在独立的 `377daec6242864a920de43a55523ac3d5f582648` checkout 中复验 jagged 时，使用相同 A100 依赖与上述受控环境，令 `RUN_DIR` 指向本次新建的结果目录，然后运行：

```bash
python -m pytest --color=no --tb=short -ra -q tests/test_jagged.py \
  --junitxml="$RUN_DIR/jagged.xml"
```

本次准确 argv、stdout/stderr、前后源码检查及退出状态见 [定向 manifest](../results/jagged_reference_recheck_a100_377daec/manifest.json)。这条命令只代表 jagged 子范围。

初轮 `b5a3206` 的完整结果已经确定为失败；修复后的 `377daec` 完整 A100 套件仍为 RUNNING，尚无最终汇总。等待新版进程结束后再按实际输出记录失败、skip 和覆盖率。单卡环境不能证明双卡测试通过；4090 的 `5b37725` 完整结果、A100 的 `b5a3206` 专项结果和 `377daec` 的定向 16 项均不能替代新版全量。

GPU runner **未包含优化后 dot/matmul**，多 program 分块标量分解仍有明确限制；softmax 通过不填补这一缺口。跨结构 pass 的 operation 来源追踪、真实历史缺陷演示等加分增强以及上游 PR/主分支合并仍按 [差距计划](excellence_gap_plan.md)推进。归档只确认对应源码、环境与范围内的真实结果。
