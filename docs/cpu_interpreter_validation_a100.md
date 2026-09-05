# CPU 参考解释器：NVIDIA A100 验证记录

更新日期：2026-09-06（Asia/Shanghai）。冻结源码 **`b5a3206f8351e5a138d16ee13f6d6ef9c620044b`** 在实际 NVIDIA A100-SXM4-40GB 上完成 **14/14 真实 Triton GPU 差分**和 **180 passed 的解释器专项回归**，两次退出码均为 0。四个已结束运行的 manifest 均记录执行前后 HEAD 与该 SHA 一致、已跟踪文件无修改。

**A100 完整仓库测试仍在运行，已收集 602 项，没有最终汇总。** 以下结果证明所列程序的硬件差分与专项范围，不表示全部原测试、整项目官方验收、上游合并或优秀学员评选已经完成。历史 RTX 4090 结果见 [4090 报告](cpu_interpreter_validation_4090.md)，验收对照见 [提交说明](cpu_interpreter_acceptance.md)。

## 按运行保留结果

原目录按 UTC 命名为 `a100-20260905T165639Z`，对应北京时间 2026-09-06。首次失败与环境修正后的结果分别归档，没有覆盖旧日志。

| 范围 | 实际结果 | 原始证据 |
|---|---|---|
| 首次逐元素 smoke | 1 failed，2.18 s；退出码 1；Triton 编译驱动辅助模块时无法链接兼容的 `-lcuda` | [manifest](../results/a100_20260906/smoke/manifest.json)、[失败日志](../results/a100_20260906/smoke/validation.stdout.log) |
| 修正 libcuda 路径后，同一 smoke | 1 passed，2.31 s；退出码 0 | [manifest](../results/a100_20260906/smoke-libcuda/manifest.json)、[日志](../results/a100_20260906/smoke-libcuda/validation.stdout.log) |
| 独立真实 GPU 差分报告 | 14/14 PASS；8 个程序、9 类用例；退出码 0 | [manifest](../results/a100_20260906/gpu-report/manifest.json)、[GPU JSON](../results/a100_20260906/gpu-report/interpreter_gpu_validation.json) |
| 解释器专项 | 180 passed，7.95 s；退出码 0 | [manifest](../results/a100_20260906/specialist/manifest.json)、[日志](../results/a100_20260906/specialist/validation.stdout.log)、[JUnit](../results/a100_20260906/specialist/junit.xml) |
| 完整仓库测试 | 运行中；602 是收集数，不是通过数 | 完成后单独保存退出状态、完整日志、JUnit 与 coverage |

smoke 是 14 项 GPU 差分中的一个用例；180 项专项又包含这 14 项，以及应用、SSA、调试、逐默认 pass、回放和 Torch 适配测试。不能相加为 15、194 或 195 个独立用例。这里的时长是各次 pytest 汇总时长；外部 runner 计时还包含进程启动等开销，二者不能混用，也不是性能基准。

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

## 复现已完成的范围

在独立 checkout 中使用测试源码 `b5a3206f8351e5a138d16ee13f6d6ef9c620044b` 和上述依赖。以下是与 manifest 对应的可移植写法，替换的仅是机器相关路径；原始 argv 与环境变量以各运行 manifest 为准。不要在本次仍运行完整测试的 checkout 中切换源码或重复启动同一全量。

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

## 仍未完成的范围

完整 A100 套件仍在运行；收集 602 项不表示 602 项通过，最终失败、skip 和覆盖率须等进程结束后按实际输出记录。单卡环境不能证明双卡测试通过，4090 的 `5b37725` 完整结果也不能重标为本次 `b5a3206` 的 A100 全量结果。

GPU runner **未包含优化后 dot/matmul**，多 program 分块标量分解仍有明确限制；softmax 通过不填补这一缺口。跨结构 pass 的 operation 来源追踪、真实历史缺陷演示等加分增强以及上游 PR/主分支合并仍按 [差距计划](excellence_gap_plan.md)推进。归档只确认对应源码、环境与范围内的真实结果。
