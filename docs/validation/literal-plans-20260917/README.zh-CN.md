# 表达式计划两项实验：未采用

两项独立实验均以已发布的 `4d50b0424d963ac7ea3fd67fb06ff2c828fb3d92` 为对照。生产求值器不变；新增 41 项语义测试用于后续优化回归。

## 固定门槛与结果

三个独立进程对照轮次，轮换新旧运行顺序；每项 7 组、每组 3 次。目标为公开 interpret_program 的三种 prepared matmul；控制为向量、padding、softmax 和小 matmul，包含 trace 开关。要求每轮目标几何平均加速比至少 1.10×，全部稳态测试项不得慢过 5%。冷启动和 tracemalloc 单独记录。

| 方案 | 第一轮目标几何平均 | 第二轮 | 第三轮 | 结论 |
|---|---:|---:|---:|---|
| 仅折叠有界整数常量子树 | 1.014061× | 1.002081× | 1.004981× | 未达 1.10× |
| 常量叶节点直接持值、算子仍实时执行 | 1.039771× | 1.090193× | 1.039057× | 未达 1.10× |

直接持值方案另有 `vector_4099`（trace=False，第 2 轮）耗时增加 5.667%，超过 5% 门槛。未缩减测试集、修改门槛或混合轮次掩盖该结果。

两项方案各有 33 条旧新配对和 7 种前端完整轨迹的逐轮摘要核对。数值与轨迹通过，但不能据此认定存在足够性能收益。全部原始时间、内存和源码摘要保留于 results；前一项折叠试验位于 experiments/constant-fold。

## 正确性与保留内容

- 折叠候选：770 passed / 15 GPU deselected，34.84s；其中表达式专项 51 passed / 0.55s。
- 直接持值候选：770 passed / 15 GPU deselected，34.68s。
- 最终生产求值器逐字节保持基线；加入测试后：`770 passed, 15 deselected in 35.40s`。可用 production/prepare_inputs.py 独立复现。
- 新增 41 项测试覆盖 Python 精确类型、signed zero、固定种子的随机表达式树、异常消息、动态数组/符号、NumPy 警告策略、自定义算子与整数子类、超大整数和对象释放。随机测试中每个种子生成 60 棵树、各有 4 种输入，不能把重复计算简单称为独立测试数。
- 保留初稿随机参考求值触发布尔减法异常的 8 failed / 43 passed 日志，并修正为核对两条路径的异常类型和信息，没有删去失败输入。
- 本轮没有启动或运行 GPU，不能把此前 GPU 成绩归于这两项候选。

## 复现

先取得上述固定基线源码，再分别运行对应目录的准备脚本：

```bash
python prepare_inputs.py --baseline /path/to/ninetoothed-baseline --work /tmp/literal-replay
cd /tmp/literal-replay
python candidate/nine/scripts/run_cpu_tests.py --junitxml results/cpu-suite.xml
python tools/confirm_nine.py
```

折叠方案用 experiments/constant-fold/prepare_inputs.py 和新的 work 目录。准备脚本按 Git blob 摘要核对全部基线文件，覆盖记录的实验源码，并复制固定协议/计时脚本；不会自动安装依赖或改变基线。依赖为 Python 3.12.14、NumPy 2.3.5、SymPy 1.14.0 和 pytest 9.1.1。采用其他版本时需独立记录环境，不能直接引用此处的比例。

本地后续 CUDA 候选位于 paired-hadamard-20260917，目前仅准备源码，未经 GPU 编译/验证，不在本次已发布实现范围。
