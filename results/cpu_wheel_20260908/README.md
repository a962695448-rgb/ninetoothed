# 纯 CPU wheel 安装验证（2026-09-08）

**结论：真实 wheel 安装、解释器、单步调试、导出回放和选定 CPU 回归通过。**
默认 GPU 安装元数据及计算核心均未改动。本次改进是安装和运行入口，不代表算法加速。

## 来源与安装方式

- 起点：`d3f7b7629d138e071d3deacfada7ee3c2e2d7559`。
- 实测来源：`d9acc12711e378ad045629959a8639d991fa2f43`。
- 该提交只改 README、安装文档、CPU 使用文档和演示脚本说明字符串。
  `src/`、`tests/`、`pyproject.toml` 与起点相同。
- 使用 `git archive` 从该提交提取构建输入；全部输入文件 SHA-256、精确命令、
  工作目录、退出码、日志哈希见 [run2/manifest.json](run2/manifest.json)。
  归档文件使用 Git 原始字节，不依赖 Windows 工作树换行符。
- 在仓库外新建、不继承系统 site-packages 的 venv；用 pip 构建普通 wheel，
  先安装 NumPy/SymPy，再以 `pip install --no-deps` 安装该 wheel。
- wheel：`ninetoothed-0.26.0-py3-none-any.whl`，198532 字节；
  SHA-256：`42c0ed739949659bd558df9bd41d1ed1358a755acac5a7abc75d6312c8f1e4c7`。
  65 个包文件均逐字匹配 Git 提取的源文件；执行后再次逐字匹配安装文件与 wheel。
- [wheel 元数据](run2/wheel-METADATA.txt) 仍含 `Requires-Dist: triton>=3.0.0`。
  这是一条明确绕过 GPU 依赖的 CPU 使用路径，不是独立发布或依赖完整的 CPU 包。

## 实际环境与结果

Ubuntu/WSL，Python 3.12.3；uv 0.12.10；pip 26.2.1；Hatchling 1.32.0；
NumPy 2.5.3；SymPy 1.14.0；mpmath 1.3.0；pytest 9.1.1。
构建依赖版本见 `02-build-wheel.stderr.log`，完整运行环境见
[已安装包列表](run2/10-installed-packages.stdout.log)。这些是本次实测版本，未改项目依赖下界。

| 检查 | 结果及证据 |
| --- | --- |
| 独立 wheel 安装 | 构建和安装退出码均为 0；构建不下载运行时 Triton |
| NumPy 解释器及调试 | `x * 2 + 1` 数组断言通过；11 个元素、15 条 trace；断点到达 program `(2, 0, 0)`；定位人为注入的 `injected_bad_constant` |
| 导出后的独立回放 | 正确参考结果与 NumPy 一致；复现候选常量 2→3 引起的 `out` 差异；首次差异为 `arith.constant` |
| CPU 回归 | **307 passed, 15 deselected, 21.06 s**；0 failure/error；[JUnit](run2/junit.xml)、[完整日志](run2/09-cpu-regression.stdout.log) |
| 包来源和 GPU 隔离 | demo、replay、pytest 三个进程执行前后均断言所有已加载 `ninetoothed.*` 文件位于该 venv 的 site-packages；Torch/Triton 均无 distribution、无可导入 spec、未出现在已加载模块中 |
| pip check | **预期退出 1**：仅报告 `ninetoothed 0.26.0 requires triton, which is not installed.`；[原始输出](run2/07-pip-check.stdout.log) |

演示/回放在 `python -I` 下经归档 runner 的 `runpy.run_path` 执行原样脚本，
并在执行前后记录 `WHEEL_PROVENANCE`。这些记录包含 `sys.path`、包位置、
distribution 位置、完整依赖版本与已加载文件路径。

回归将原样 `tests/` 和 demo 复制到无 `src/` 的独立工作目录，并清除父进程
`PYTHONPATH`。不使用源码目录或 editable install。demo 回归自身给子进程设置的
`PYTHONPATH=<副本>/src` 指向不存在的目录，因此也无法覆盖已安装包。
15 个排除项是实际 Triton GPU 差分用例；没有运行 GPU 或 Torch 专属测试。
测试耗时是回归墙钟时间，不是性能基准。此前 A100 验收仍以
[最终 A100 报告](../../docs/a100_final_validation_20260908.md) 为准。

## 失败、边界与复现

首次调用系统 `python3 -m venv` 因缺 `ensurepip` 退出 1，尚未构建或安装包。
其 `/tmp` 原始文件在归档时已不可用；不将重建内容冒充原始日志。
在持久目录重新执行同一环境创建操作，保留了新的
[失败命令](venv-failure-reproduced/command.json) 和
[原始错误输出](venv-failure-reproduced/stdout.txt)。成功 run2 改用现有
`uv venv --seed --python /usr/bin/python3`，未更改系统 Python。
安装文档已补充这条替代方式。

提交前首次 Ruff format 检查发现演示脚本编辑后的混合换行符；只格式化该文件后，
全量 Ruff 格式、lint 和项目 style 检查通过。计算语句无变更。

run2 原始命令从 `/tmp/ninetoothed-cpu-wheel-20260908-run2` 执行。
归档保留原路径与哈希；临时 venv 不作为持久运行入口。
完整日志、wheel 和 replay 另存于本机 WSL 的
`/home/xxl/ninetoothed-cpu-wheel-20260908-run2`。仓库不提交构建产物 wheel。

冻结的 [验证程序](run2/validate-original.py) 可用 Python 3.12+ 在 Linux 再现。
从本仓库运行，指定一个尚不存在的**持久**输出目录：

```bash
python3 results/cpu_wheel_20260908/run2/validate-original.py \
    "$PWD" "$HOME/ninetoothed-cpu-wheel-recheck" "$(command -v uv)"
```

它验证调用时的 HEAD，重新记录版本/命令/哈希，不覆盖任何旧运行。
若系统 Python 自带 venv/pip，可省略最后一个 uv 参数。
用户安装入口见 [installation.rst](../../docs/source/installation.rst)。
本次后续归档提交仅补失败处理说明和统一已安装包运行命令，未修改 wheel 内的计算代码。
