# 九齿解释器实施与优化计划

更新日期：2026-09-07。本轮功能源码已冻结为 **`6ecce58da28bb9709aa35fc6c25c1f361aff736f`**。它修改了运行时、默认 pass、发射器、SSA 来源记录和测试，并把标量存储保护统一到 single/multi dot 与 transpose；相对历史代码有功能变化，不能继承旧 GPU 结果。

最新冻结 CPU 选择为 **307 passed、15 deselected，32.68 s，退出码 0**；它是无 Torch/Triton 的 NumPy CPU 范围。独立 **RTX 4090** 运行已完成 **15/15 Triton 差分（9 程序、10 类别）及一个 CUDA 标量 dot probe，均 exit 0**，见[实机归档](../results/interpreter_optimization_20260906/gpu-6ecce58/archive_manifest.json)。这些范围不累计；新源码的 A100、完整仓库与双卡验证仍未完成。两次历史 Sphinx 构建均因缺少依赖失败，服务器Sphinx实际exit0/28页；控制器因14个正常autosummary输出记FAIL，独立复核确认原有输入未变。HTML内logo仍是LFS指针，静态资产不完整，正式发布前须恢复真图，见[最终记录](../results/interpreter_optimization_20260906/sphinx-final-20260907/delivery_limitations.json)。当前只收尾已有实现与证据；用户验收前不创建 PR 或执行官网提交。

## 要求依据与验证层次

依据 2026-09-06 重新只读核对的[官方任务](https://gxtctab8no8.feishu.cn/wiki/GxhWwiz0iiAhKkk7CFhcLTCyn2b)，按功能、验证条件和排除范围组织实施。接口细节见 [CPU 解释器文档](source/cpu_interpreter.rst)，验收记录见 [验收与提交说明](cpu_interpreter_acceptance.md)。

| 功能或验证条件 | 当前实现与入口 | 需要保留的边界 |
|---|---|---|
| 清晰 CPU 入口，复用 arrangement/frontend/SSA | `interpret`、`interpret_program`，直接解释共享布局和 SSA；已加载的 PyTorch CPU Tensor 可共享 NumPy 存储 | 6ec 无 Torch/Triton 的 NumPy CPU 选择已通过；未选择 Torch 适配文件，不把该结果解释为全部 CPU 适配路径通过 |
| 最小操作集合与程序状态 | 常量、算术、比较、cast、select、broadcast、mask load/store、sum/max、if/for、value environment、program ID | 非活动地址不解引用；未支持的 operation/dtype/访存形式明确报错并给出 SSA 位置 |
| 五类应用与三种必需 dtype | elementwise、broadcast、非整除尾块、row reduction、分支/循环；float32、int32、bool | float32 对独立参考用 `rtol=1e-3, atol=1e-3`，整数和布尔完全相等；参数化数量不替代功能范围 |
| 默认优化前后及实际 GPU 差分 | CPU 比较 frontend 与默认目标管线；GPU runner 比较独立参考、两份 SSA 的 CPU 结果与实际 GPU 输出 | 官方 GPU 条件使用 A100；6ec 的 15 个 Triton 用例和一个 CUDA probe 已在 RTX 4090 通过；新 A100 仍未完成，旧记录不迁移 |
| dot/matmul 与 softmax | 本轮增加完整 K 的多 M/N 输出 tile 标量 dot；softmax 已有解释器实现 | 仅下述 rank-2 合同；未实现 split-K 累积、Tensor Core 验证或任意矩阵布局 |
| trace 与交互调试 | program/opcode 过滤、单步、断点、watch、独立快照和回放 | 来源候选不是 Python 源码行；完整事件序列不同不能强行对齐中间值 |
| pass 差分与来源记录 | `check_passes`、`Operation.origins`、显式 pass 关系与 `source_candidates` | 结构一致且完整 trace 对齐时可报告操作位置；结构变化只给已声明的原 SSA 候选集合，不证明唯一因果点 |
| 复现材料与局部扩展 | SSA/输入/shape/dtype/seed/布局导出与回放；独立 operation handlers | 当前教学故障注入不等于真实历史缺陷回放；不宣称自动缩减到最小 shape 或最少操作 |
| 外部审查与合并 | 可准备可复查的实现、文档和证据 | 必须在全部约定优化、验证和用户验收后处理；当前不创建 PR 或执行官网提交 |

## 本轮已实现的具体合同

### 多 program 标量矩阵乘法

默认 Triton/CUDA 管线输出的标量 SSA 和 K 循环由 CPU 实际执行，输入为 rank-2 的 `(M,K)` 与 `(K,N)`，输出为 `(M,N)`。M/N 可以跨多个输出 tile，但每个 program 必须包含完整 K；K 尾块补零，输出有效坐标完整覆盖且每个位置只写一次。操作使用 value-space 局部行列坐标，再通过共享布局解析到存储位置。

[矩阵乘法回归](../tests/test_interpreter_matmul.py)覆盖 float32/int32、非方阵、M/N/K 尾块、非连续输入，以及元素对齐的独立正/负 stride 输出和保护区。K 跨独立 program 却无累积、计算得到的操作数、额外 value 层级、潜在输入输出别名、部分字节重叠、输出重复写及额外 store/atomic 副作用均在支持合同外；现有负向测试明确检查 K 分拆、别名、重叠输出和嵌套额外 store 在写入前被拒绝且缓冲区保持不变。额外副作用递归检查嵌套 region，避免每个输出 lane 重复执行它们。

6ec 的公共存储保护在 single/multi dot 与 transpose 的标量 lane 执行前统一检查：输入输出潜在别名、部分字节重叠、造成重复写的 zero stride，以及同一 `out` 绑定又作为 lhs 等值读取均被拒绝；形状/stride/offset 元数据读取仍允许。正常 untiled 与独立、元素对齐的正负 stride 输出继续支持。新增 12 项单 program 回归修正前为 8 failed、4 passed；修正后相关组合 137 passed，正式广范围结果另按 6ec 的 307/15 保存。

当前 GPU 新例只覆盖 float32 标量 lowering：`(7,3) @ (3,6)`、4×4 tile、四个 M/N 输出 program。CPU 的 int32 和丰富 strides 测试不能推断为这些情况也已完成 GPU 验证。[GPU fixture](../tests/test_interpreter_gpu.py)含发射坐标检查和完整 origins 往返检查，代码生成检查本身不是 GPU 运行。

### 显式来源记录与保守定位

[来源实现](../src/ninetoothed/ir/provenance.py)为原始 SSA 分配稳定 ID，记录 preserve、replace、split、merge、delete 及未知关系；默认管线和实际 linalg decomposition 已接入。临时名字分配保留整个 Program 的现有名字，包括嵌套 region，避免生成变量冲突。

未知 pass 不能仅凭复制的 ID、名字或结构获得来源；混合已知/未知来源的关系不返回貌似完整的已知子集。来源候选描述已声明的变换范围，不是值等价、Python 行号或唯一故障原因。`ProgramComparison.first_operation` 只有结构与完整 trace 序列一致时才成立；否则保留输出差异和候选范围。旧 JSON 缺少 origins/来源元数据时仍可读，默认保持未知。

[来源回归](../tests/test_interpreter_provenance.py)覆盖显式拆分/合并/删除、未知映射、分支与循环 trace 分歧、真实默认 dot/transpose pass、故障注入及正确负向控制。它们验证解释器和变换记录的合同，不能冒充新发现的上游缺陷。

## 证据版本分别记录

| 版本与范围 | 真实状态 |
|---|---|
| b5 A100 专项 | 14/14 GPU、180 项解释器专项；180 包含 14，见[历史 A100 报告](cpu_interpreter_validation_a100.md) |
| b5 / 377 A100 全量 | 分别为 16 failed、584 passed、2 skipped；1 failed、599 passed、2 skipped；原 FAIL、命令和诊断边界保留 |
| 377 jagged 定向 / 825 generation 定向 | 分别 16 passed、77 passed；不与专项或全量相加 |
| 825 A100 全量 | 600 passed、2 skipped、450.25 s、exit 0；见[原运行清单](../results/full_suite_a100_82592b8/manifest.json) |
| 086 资料归档 | `086f148b40a7ac057f9184ecfbfccef84eb4037e` 仅归档 docs/results，运行源码仍为 825；不是另一轮实测 |
| f35 CPU 选择 | 1 failed、294 passed、15 deselected、35.42 s、exit 1；原断言对全部 metadata 做字符串检查，误命中 origins 中保留的操作名，见[原失败](../results/interpreter_optimization_20260906/cpu-f35fb51/manifest.json) |
| 56 CPU 复验 | `56f091eb585e94b725a08989e44a63b222b1e3f0`：295 passed、15 deselected、34.18 s、exit 0，见[复验清单](../results/interpreter_optimization_20260906/cpu-56f091e/manifest.json) |
| 6ec CPU 阶段 | 307 passed、15 deselected、32.68 s、exit 0，见[最新清单](../results/interpreter_optimization_20260906/cpu-6ecce58/manifest.json)；无 Torch/Triton，15 个 GPU 用例在该 CPU 命令中取消选择；独立 4090 运行另记 |

171 passed、23.72 s 的预备组合，以及新增存储回归的 8 failed/4 passed 与后续 137 passed 属于开发检查；不与三次冻结 CPU 运行累计，也不改写对应失败记录。

历史 4090 的 224/209/206/254/591/39 等也分别归于其源码、依赖和选择范围，见[历史报告](cpu_interpreter_validation_4090.md)。CPU、GPU、完整套件和诊断场景不累计。旧 squeeze 控制实验没有恢复原全量分配器与索引现场，后续成功仍不能确定旧失败的精确原因。

## 当前交付与后续验收边界

1. f35/56/6ec 三轮 CPU 原文与各自源版本保留；最新 307/15 不扩大为 Torch 适配或全仓库通过。
2. 6ec 的 15 项 Triton 差分及一个 CUDA 后端标量 dot 已在 4090 完成。新的 A100、完整库与同机多卡不在本次证据范围，不继承历史 825 的 600/2。
3. 保留两次 Sphinx 原 FAIL；静态绘图的 GUI 依赖改为只在交互入口导入，服务器Sphinx实际exit0/28页；控制器因14个正常autosummary输出记FAIL，独立复核确认原有输入未变。HTML内logo仍是LFS指针，静态资产不完整，正式发布前须恢复真图，见[最终记录](../results/interpreter_optimization_20260906/sphinx-final-20260907/delivery_limitations.json)。
4. 来源候选、故障注入与导出/回放是明确的现有能力；唯一根因定位、自动最小样例缩减、任意布局/split-K 和性能最优不作为已完成承诺。
5. 源码、文档和原始证据同步到个人交付分支供用户验收；上游 PR、官网提交及维护者合并仍未执行。

## 排除范围与提交约束

解释器不生成原生 CPU 代码，不以 CPU 执行性能为目标，也不模拟 warp/block 调度、shared memory 或 GPU race。Atomics、间接指针、多设备、随机数、float8 等平台相关语义不是本轮支持承诺；不支持时显式报错。其他项目的国产平台适配不作为九齿实现或验证证据。

正式 PR 的分支命名、标题、正文中非空 pytest 输出区块和工作流要求以现有 CONTRIBUTING 与[提交说明](cpu_interpreter_acceptance.md)为准；中文 commit 历史保留，不改上游规则。材料可以先准备，但必须先完成全部约定优化与验证并由用户验收，之后才能处理上游 PR 与官网提交。外部合并状态只按真实结果记录。
