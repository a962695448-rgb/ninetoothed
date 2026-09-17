# 布局调用及表达式语义修订的 RTX4090 验证

以 07062137e599cfead66d510cede3c0a3c1bbdf6b 为精确源码，本次 15 项真实 Triton GPU 差分全部通过，108 项布局/表达式语义回归通过（0.77s）。108 项测试使用服务器上的解释器，不代表 108 个 GPU 内核。原无 Torch/Triton 的 770 项完整 CPU 选择范围属于此前已保存的本地与 CI 运行，本次没有重复执行它。

GPU 为 NVIDIA GeForce RTX4090，计算能力 8.9；Python 3.12.3、NumPy 1.26.4、SymPy 1.13.1、Torch 2.6.0a0+ecf3bae40a.nv25.01、Triton 3.1.0、CUDA 12.8。15 项涵盖 float32/int32 尾块、广播、行归约、布尔比较、分支/循环、softmax、整数 floor division/remainder 与一个 float32 多输出 tile 的矩阵乘案例。浮点 rtol=atol=1e-3，整数/布尔精确比较；详细输入、seed、形状和边界逐项保存于 nine-gpu.json。

这批证据覆盖近期关键字调用拒绝、空逻辑形状修复及表达式语义测试所对应的当前计算源码。它没有扩展成完整库 GPU 套件、所有 dtype/layout、split-K 或 Tensor Core 验证，也不作新的 GPU 加速声明。旧 A100 全套结果继续保留其旧源码范围。

SOURCE_MANIFEST.json 将当前提交的全部 196 个文件同时对应到 Git blob 和远端冻结输入 SHA256，含全部 70 份 src 文件。运行前后校验相同输入，原始结果回收到本地逐文件核对后关闭实例。ENVIRONMENT.json 是同一实机会话记录；其中 CUDA 编译器信息也服务于另一项 Hadamard 实验，不代表 NineToothed 使用该编译器生成每个 Triton 内核。

用 verify_checkout.py 先核对本地检出，再按 commands.txt 在对应 CUDA 环境执行。命令使用新的输出路径，保留旧日志。GitHub 自托管 GPU 自动任务仍为排队状态；本页成功指的是已归档的手动实机验证。
