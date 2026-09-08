# 逐阶段差分定位与失败复现

本轮功能源码为 `9c5ffbad6d75e8b7f083270f08dc4009a8fe4424`，新增相邻 pass 差分、带比较依据的 SSA 观测位置，以及通用失败自动导出和独立回放。选定 CPU 回归 **328 passed、15 deselected，57.58 秒**；15 项为明确排除的 GPU 测试。原始命令、源码哈希及日志见[证据索引](../results/pass_failure_workflow_20260908/README.md)。

## 如何使用

```python
from ninetoothed.interpreter.debugger import check_passes

report = check_passes(
    original_program,
    named_passes,  # A sequence of named Program-to-Program transforms.
    inputs,
    tensors=tensors,
    symbols=symbols,
    failure_dir="/tmp/nine-failure-001",  # Use a directory that does not exist.
    seed=2026,
)
print(report.first_bad_pass, report.localization)
print(report.reproducer, report.export_error)
```

首次失败立即停止执行后续 pass，并自动创建复现材料；全部通过时不创建目录。`failure_dir=None` 保持默认无文件写入行为。单独比较两个 SSA 的 `compare_programs` 同样接受 `failure_dir` 和 `seed`。

```bash
python /tmp/nine-failure-001/replay.py
```

回放直接读取保存的 SSA 和数值输入，不依赖原来的 frontend、Python pass 函数或固定演示公式。它要求重现同类输出差异、操作观测位置和 trace 对齐状态，或相同类型与内容的执行异常。错误消失、变成另一种错误都会返回非零；`python -O` 不会关闭这些检查。

## 定位依据

每个 pass 都分别与原始程序及上一轮程序比较：原始基准避免多轮小误差逐渐累积，相邻比较避免正负误差相互抵消。原始程序先执行成功，再允许将失败归于某个 pass。旧的 `difference` 字段保留原始基准结果，新字段 `adjacent_difference` 保存相邻比较。

`localization` 包含比较基准、候选和参考位置、program ID、循环迭代、标量 lane，以及发生差异的 input/result/mask。其 `basis` 有三种明确含义：

| 依据 | 可以报告的内容 | 边界 |
|---|---|---|
| `full_trace` | 完整对齐 trace 中最早出现差异的操作 | SSA 结构及所有事件对应 |
| `aligned_prefix` | 控制流分歧前，公共执行前缀中的差异 | 不将分歧之后的事件按位置强行配对 |
| `retained_boundary` | 重构前后明确保留的操作，其输入、mask 或结果首次出现差异的位置 | 验证 pass 指纹、操作一致性、唯一 preserve 映射及完整过滤事件序列；只对保留的观察点排序 |

保留操作处观察到错误，不等于证明该操作本身就是错误变换的唯一根因。没有可信对应关系的重构仍返回 `localization=None` 和已声明的来源候选；不会把 split/merge 来源 ID 当作中间值等价证明。

真实默认管线已做以下端到端验证，故障均为明确标注的人工注入：

| 管线 | 程序与故障 | 实际差异观测位置 |
|---|---|---|
| Triton / CUDA 默认 pass | `(7,3) @ (3,6)`；标量分解中的乘法被改成加法 | `entry:6:scf.for/region0:3:arith.add` 的输入 `%8`；program `(0,0,0)`、iteration `(0,)`、lane `(0,0)` |
| Triton / CUDA 默认 pass | 3×3 非对称矩阵转置；提取行列索引互换 | `entry:3:mem.store` 的输入 `%3`；program `(0,0,0)`、lane `(0,1)` |

四个故障包均由独立安装的 wheel 使用 `python -I` 成功回放；四个修复后的对照均通过且不生成失败包。这里的 Triton/CUDA 指真实目标的 SSA 优化管线，执行仍在 CPU 上，不能计作 GPU 实测。

## 自动导出的内容

- `reference/`、`candidate/`，pass 检查另加 `previous/`：各含 JSON SSA、可读 SSA、NPZ 数值输入和 manifest。
- 输入的 shape、dtype、正/负/零 strides、可写属性、完全相同对象的别名关系、grid、symbols 和调用方提供的 seed。
- `failure.json`：首次失败 pass、已检查阶段、两个基准的差分报告、定位依据、容差、异常和环境版本。
- 通用 `replay.py`：核对保存的错误；无 pickle 或任意 Python pass 代码的反序列化。

保留给定问题的全部复现输入，没有声称自动压缩程序或缩小 shape。独立重叠视图仍按原合同拒绝。seed 未提供时记为 `null`，不推测随机状态。

Python pass 抛异常、返回错误对象或无效 SSA 时保存明确标注为不可执行回放的诊断快照，不伪装成可重放的 Python 变换。目录已存在或磁盘写入失败时，原始差分/异常保留，额外填写 `export_error`；部分导出保留 `INCOMPLETE` 标记，回放拒绝使用。

## 验证与版本边界

- `9c5ffba`：无 Torch/Triton 的 328 项 CPU 回归通过，含新增 21 项失败工作流测试；Ruff、格式与项目风格通过。
- 同源码普通 wheel 在新环境 `--no-deps` 安装，64 个 Python 源文件与冻结源码逐字核对；演示和通用回放通过。包元数据仍依赖 Triton，未新增独立 CPU 发行包。
- 四组真实默认管线的自动导出、隔离 wheel 回放和四组修复对照通过，单独记录，不与 328 相加。
- 文档环境使用真实 `torch==2.5.1+cpu`，无导入 mock、无 Triton，Sphinx 严格构建退出 0，生成 28 个 HTML 页面；Torch CPU 非连续输入自动导出，再由无 Torch 的安装环境回放通过。
- 早一版 `7d158f3` 的 325 项 CPU 检查通过，但同轮全站文档因缺少 Torch 失败，原日志和失败状态保留；最终文档使用单独补齐依赖的环境验证。

相对 A100 验证源码 `5920018`，本轮 `src/` 只改 `interpreter/debugger.py` 并新增 `interpreter/failure.py`；数值执行器、SSA pass、GPU 发射器没有变化。本轮没有启动 GPU，不能将历史 A100 的 682 项全量结果标成 `9c5ffba` 的新实测。上游 PR、合并和官网提交仍待用户验收。
