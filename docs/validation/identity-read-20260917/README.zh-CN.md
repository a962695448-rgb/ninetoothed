# 恒等布局读取的配对验证

基线：a962695448-rgb/ninetoothed 的 385a4a5c5b900e53e0ec724b76ebdbe68590e3d3。
source 是通过最终验证的完整源码覆盖；initial 保存 whole_tensor 初版。首轮一项 trace 对照退化约 5.5%，判为 REJECT。调整后按同一门槛得到 ACCEPT：三轮目标几何平均 7.611/7.147/7.198 倍，48 条记录完整轨迹一致，最大退化 1.754%。

目标是公开 interpret_program 的未安排布局、读取为主且不包含输出 mem.store 的 SSA 程序，不把收益外推为通用 GPU 或模型速度。对照包含同形状写入、trace 开关、前端 affine/softmax，内存数据为 tracemalloc 而不是 RSS。

## 复现

检出上述基线，然后在无 Torch/Triton、具有 NumPy/SymPy/pytest 的环境执行：

```bash
python prepare_inputs.py --baseline /path/to/ninetoothed --work /tmp/nine-read-replay
python /tmp/nine-read-replay/tools/confirm_nine.py
```

准备脚本逐个检查基线 Git blob，然后应用 source。输出目录必须不存在。checks 包含初版 607 项和最终 609 项 CPU 回归；最终增加了两个数组子类协议用例。gpu 保存同一冻结源码的 15 项真实 GPU 差分以及新旧 74 项边界测试。两轮均保留原始数据，不累计重复测试为独立用例。
