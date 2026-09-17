# 布局调用语义修复的固定证据

基线为 a962695448-rgb/ninetoothed 的 9de14a3d8effad34799467faa90bef936434c315。原版会静默丢弃 IndexExpr 调用的关键字，并将零 next_power_of_2 及命名 padded shape 算成 2，导致公开空读取失败。

同一新增测试在原版得到 21 failed / 46 passed；修复后 67 passed。完整 CPU 范围 729 passed / 15 GPU deselected。参考 Triton v3.1.0 固定提交 cf34004b8a67d290a962da166f5aa2fc66751326，其整数工具的零/负输入返回 0；REFERENCE.json 区分源码验证与实际 GPU 运行。

修复拒绝无法表示的调用关键字，位置调用保持兼容；逻辑形状取整共享私有 helper。没有修改 GPU 启动块大小或调度默认值，没有新 GPU 运行或性能加速声明。

检出固定基线后，在无 Torch/Triton、具有 NumPy/SymPy/pytest 的环境中执行：

```bash
python prepare_inputs.py --baseline /path/to/ninetoothed --work /tmp/layout-call-replay
python /tmp/layout-call-replay/candidate/nine/scripts/run_cpu_tests.py --junitxml /tmp/layout-call-replay/results/pytest.xml
```

准备工具逐项验证 Git blob 并应用 source 覆盖。输出目录必须不存在。checks 保存原版失败、修复回归和文档日志，测试代码保留独立位扩散参考及真实 interpret_program 空张量用例。
