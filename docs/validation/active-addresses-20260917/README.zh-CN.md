# 活动地址记录的固定验证档案

基线为 a962695448-rgb/ninetoothed 的 f68c88f162c09974ef59686929dd3d89633cb5a1。source 保存实际测量的代码与 53 项新回归；measurements 保存全部三轮新进程配对记录；checks/gpu 分别保存 CPU、文档及真实 GPU 兼容性证据。

12 个记录层微基准的三轮几何平均为 1.2484 / 1.2476 / 1.2398 倍，57 条记录的事件或完整轨迹指纹一致。稀疏 mask 的分配峰值降低约 40%–74%；密集 mask 峰值近似持平。实际前端 trace 程序仅观测约 1%–5% 的整体收益，不把记录层指标写成全解释器或 GPU 的速度。

本地：662 passed, 15 deselected。RTX4090 上同一源码的 15 项真实 GPU 差分通过；88 项地址/依赖测试在 NumPy 1.26.4 通过。后续 CUDA 复测未改动这些 Python 源码。

检出固定基线，在无 Torch/Triton 且安装 NumPy/SymPy/pytest 的环境中执行：

```bash
python prepare_inputs.py --baseline /path/to/ninetoothed --work /tmp/active-address-replay
python /tmp/active-address-replay/tools/confirm_nine.py
```

准备脚本逐项验证基线 Git blob，应用 source 覆盖，要求输出目录不存在。计时排除 recorder 构造和预存输入/mask；内存为 tracemalloc，不是 RSS。原始记录没有删除负例或换算为不同平台的收益。
