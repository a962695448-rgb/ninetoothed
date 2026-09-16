# dtype 描述符缓存与更广形状验证（2026-09-17）

基线 `64bd6958a610a0e5e61807aa141bf1cbc7881426`。重复 dtype 规范化位于解释器逐操作路径。新增至多 64 项的描述符缓存，仅按规范化后的字符串保留不可变 dtype；None/fallback、字节序、原始错误文本和用户对象变化保持原语义。

## 配对结果

实际前端和默认管线产生 SSA，前端准备不计时。连续 4096 元素、跨步 16387 元素 FP32 循环，以及反向 8193 元素 INT32 循环作为目标；带尾部 softmax 17×257、matmul 9×16×7 及开启轨迹的版本作为对照。三轮独立旧/新进程交替，每配置七个无剖析器样本。

三轮目标几何平均加速分别为 1.386、1.329、1.383×；各目标至少快 5%、目标几何平均至少快 10%、对照不得慢 5% 的预设门槛全部通过。完整轨迹指纹、独立 NumPy 参考及输入不可变性均一致。开轨迹对照的收益较小，不把目标循环收益推广至所有程序，也不与上一轮不同形状的加速比相乘。

另测热缓存和冷缓存执行期 tracemalloc 峰值。冷缓存测量前清除表达式计划和 dtype 缓存，因此包含本次新建缓存；同时记录结果释放后及再次清缓存后的分配。数据不是进程 RSS，完整记录见各轮 JSON，不宣称整体进程内存按同一比例减少。

## 回归

- 无 Torch/Triton 的 CPU 选择范围：535 passed、15 actual GPU deselected，36.18 秒。
- 新增 23 项 dtype 回归：别名、字节序、错误原始拼写、fallback/symbol 区分、动态用户对象与释放；包含在 535 内。
- RTX 4090 D 实机：15/15 真实 GPU 差分，23 项新回归通过（0.42 秒，后者在主机 CPU 执行）。
- Ruff、format、贡献风格检查通过。设备 GPU 核函数和生成器未更改。

## 复现

检出固定基线后，以不含 Torch/Triton 的 CPU 环境执行：

```bash
python prepare_inputs.py --baseline /path/to/ninetoothed --work /tmp/dtype-repro
python /tmp/dtype-repro/tools/confirm_nine.py
```

该脚本校验所有基线 Git blob，再应用 source/ 两个文件。绝对时间对应 macOS arm64、Python 3.12.14、NumPy 2.3.5、SymPy 1.14.0；换环境应重新配对，不能直接比较跨机器时间。GPU 环境记录为 NGC Torch 2.6 / CUDA 12.8 / NumPy 1.26.4。

候选仓库回归入口为 scripts/run_cpu_tests.py。实机执行 scripts/verify_interpreter_gpu.py，并用 PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_interpreter_dtype_cache.py 运行新增测试。

535 是当前 CPU 选择范围；旧全库 GPU 集合仍保留其受测源码范围。此次改善 CPU 解释器开销，不是 Triton GPU 内核或模型端到端加速。
