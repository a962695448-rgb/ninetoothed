# 解释器表达式求值优化：测量记录

基线提交：`d7e779642eab70164be849b2dcf52a5d02a7858a`。本轮仅修改解释器内部表达式求值，新增十项回归；SSA、编译器与 GPU 生成代码保持原样。

## 方法

先用 cProfile 与 tracemalloc 找热点，再进行三轮独立旧/新进程比较，运行顺序交替。每个用例含预热和五个无剖析器计时样本，单独测量内存；前端构建时间另列，正确性及轨迹序列化时间不计入执行计时。固定的三项程序均由实际前端与默认 Triton 目标管线生成 SSA。

预设门槛：矩阵乘、softmax 在每轮中位数上至少快 10%，对照程序不得慢 5%，开启轨迹的峰值内存不得增加 5%。所有输出符合独立 NumPy 公式，完整轨迹的序列化指纹在旧/新之间一致。结果：三轮全部满足门槛。

| 程序 | 开启轨迹 | 三轮中位数加速比范围 |
|---|---|---|
| elementwise_loop_16384 | False | 1.066–1.113× |
| elementwise_loop_16384 | True | 1.012–1.081× |
| softmax_32x128 | False | 1.513–1.550× |
| softmax_32x128 | True | 1.427–1.448× |
| matmul_12x16x12 | False | 1.624–1.661× |
| matmul_12x16x12 | True | 1.457–1.487× |

上述是 CPU 解释器执行耗时，不能解释为 Triton GPU 内核或模型端到端加速。轨迹内存整体持平，未压缩或丢弃调试记录。三个固定程序的结果不代表所有 SSA 程序；首次构建、极小程序与未支持语义保持各自边界。

## 实现与语义

对不可变 IndexExpr 的算术结构建立可复用求值计划，符号值仍在每次调用时读取。缓存只保留最近 256 个根表达式计划；简单常量和符号直接求值，避免挤占缓存。对象身份作为键，区分结构相等的 True、1、1.0 以及正负零；不可哈希常量继续原求值路径，不全局保留数组。未优化的表达式仍按原受支持操作求值，不执行 Python eval。

回归覆盖：常量类型与负零、符号与数组原地变化、输入释放、不可哈希常量、错误先后顺序、旧表达式缓存释放；原完整 CPU 选择范围通过 512 项，15 项实机测试按规则排除。

## 原型记录

首次使用结构相等作为缓存键的原型会混淆布尔、整数与浮点常量，已用明确反例证实并弃用。第二个按所有节点身份缓存的原型在不带轨迹的矩阵乘中出现缓存频繁淘汰，速度回退，亦弃用。最终只缓存根计划并直接处理简单表达式，既保留语义又通过预设测量门槛。原型与原始剖析结果保留在个人实验档案中。

## 最终回归与实机记录

- 无 Torch/Triton 的 macOS arm64 / Python 3.12.14 / NumPy 2.3.5 / SymPy 1.14.0 环境：512 passed、15 actual GPU deselected，35.10 秒；新十项包含在 512 内。
- RTX 4090 D / NGC Torch 2.6 / CUDA 12.8 / NumPy 1.26.4：15/15 真实 Triton GPU 差分通过；新增十项表达式测试 10 passed in 1.55s。
- Ruff、format、贡献风格检查通过。最终表达式实现 SHA-256 与三个确认轮次和实机上传记录一致。

初次实机 pytest 因实验驱动漏设 PYTHONPATH 而在收集时退出 2，数值差分本身已通过。补上指向受测源码的 PYTHONPATH 后十项测试通过；初始失败的日志/JUnit、修复后的日志/JUnit，以及 `summary-final.json` 均保留。未修改断言或库实现来解决这个启动配置问题。

## 复现

取基线提交 d7e7796 为 `control/nine`，复制为 `candidate/nine`，将本目录 `source/` 两个文件按原相对路径放到 candidate。将本目录 `profile_interpreter.py`、`confirm_interpreter.py` 放入同级 `tools/`，PROTOCOL.json 放在根目录，创建空的 results 目录：

```bash
python tools/confirm_interpreter.py
```

该入口要求 CPU 环境具备 NumPy/SymPy，逐次创建独立进程并保存六份原始记录和最终判定。可单独用 `profile_interpreter.py --source /path/to/nine --output /tmp/new-profile.json` 查看 cProfile 热点和 tracemalloc 内存。确认阶段使用 `--measure-only`，计时阶段无 cProfile/tracemalloc。

从候选仓库运行回归：

```bash
python scripts/run_cpu_tests.py
# 下列命令在 GPU 环境运行：
python scripts/verify_interpreter_gpu.py --report /tmp/new-gpu-report.json
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_interpreter_expression_plans.py
```

单机固定工作负载不能保证所有程序同等提升。当前轮次没有重新运行未变化的完整卷积 GPU 集合；旧 902 项集合保持其原版本范围。本轮 CPU 解释器优化也不改变 Triton GPU 内核的性能结论。
