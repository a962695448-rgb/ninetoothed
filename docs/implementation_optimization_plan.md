# 九齿解释器实施与优化计划

更新日期：2026-09-06。本轮功能源码已冻结为 **`f35fb51b16a52392e7ee92b3a3c15622305d428b`**。它修改了运行时、默认 pass、发射器、SSA 来源记录和测试，相对历史 `82592b8` 有功能变化，不能继承旧 A100 的 600 passed、2 skipped 结果。

本轮预备 CPU 组合为 **171 passed、23.72 s**，仅代表当次选择范围。随后冻结 f35 的广范围 CPU 选择得到 **1 failed、294 passed、15 deselected，35.42 s，退出码 1**：原测试把整个 SSA 元数据中的来源记录名称误当成尚未消除的执行 opcode；原失败保留，测试兼容修正需独立复验。GPU runner 已准备 **15 个用例**，包括新增的多 M/N 输出 tile、完整 K 的 float32 dot，**新版本 GPU 尚未运行，也没有新的 A100 结果**。本阶段完成全部约定优化与验证后先交用户验收；验收后才处理上游 PR 与官网提交，当前不创建或发布 PR，不执行官网提交。

## 要求依据与验证层次

依据 2026-09-06 重新只读核对的[官方任务](https://gxtctab8no8.feishu.cn/wiki/GxhWwiz0iiAhKkk7CFhcLTCyn2b)，按功能、验证条件和排除范围组织实施。接口细节见 [CPU 解释器文档](source/cpu_interpreter.rst)，验收记录见 [验收与提交说明](cpu_interpreter_acceptance.md)。

| 功能或验证条件 | 当前实现与入口 | 需要保留的边界 |
|---|---|---|
| 清晰 CPU 入口，复用 arrangement/frontend/SSA | `interpret`、`interpret_program`，直接解释共享布局和 SSA；已加载的 PyTorch CPU Tensor 可共享 NumPy 存储 | 不调用 GPU 代算，不另写 application 到 NumPy 的转换器；新版本 CPU-only 检查须独立留证 |
| 最小操作集合与程序状态 | 常量、算术、比较、cast、select、broadcast、mask load/store、sum/max、if/for、value environment、program ID | 非活动地址不解引用；未支持的 operation/dtype/访存形式明确报错并给出 SSA 位置 |
| 五类应用与三种必需 dtype | elementwise、broadcast、非整除尾块、row reduction、分支/循环；float32、int32、bool | float32 对独立参考用 `rtol=1e-3, atol=1e-3`，整数和布尔完全相等；参数化数量不替代功能范围 |
| 默认优化前后及实际 GPU 差分 | CPU 比较 frontend 与默认目标管线；GPU runner 比较独立参考、两份 SSA 的 CPU 结果与实际 GPU 输出 | 官方 GPU 条件使用 A100；f35 的 15 个用例尚未实际运行，旧 b5/825 的结果不迁移 |
| dot/matmul 与 softmax | 本轮增加完整 K 的多 M/N 输出 tile 标量 dot；softmax 已有解释器实现 | 仅下述 rank-2 合同；未实现 split-K 累积、Tensor Core 验证或任意矩阵布局 |
| trace 与交互调试 | program/opcode 过滤、单步、断点、watch、独立快照和回放 | 来源候选不是 Python 源码行；完整事件序列不同不能强行对齐中间值 |
| pass 差分与来源记录 | `check_passes`、`Operation.origins`、显式 pass 关系与 `source_candidates` | 结构一致且完整 trace 对齐时可报告操作位置；结构变化只给已声明的原 SSA 候选集合，不证明唯一因果点 |
| 复现材料与局部扩展 | SSA/输入/shape/dtype/seed/布局导出与回放；独立 operation handlers | 当前教学故障注入不等于真实历史缺陷回放；不宣称自动缩减到最小 shape 或最少操作 |
| 外部审查与合并 | 可准备可复查的实现、文档和证据 | 必须在全部约定优化、验证和用户验收后处理；当前不创建 PR 或执行官网提交 |

## 本轮已实现的具体合同

### 多 program 标量矩阵乘法

默认 Triton/CUDA 管线输出的标量 SSA 和 K 循环由 CPU 实际执行，输入为 rank-2 的 `(M,K)` 与 `(K,N)`，输出为 `(M,N)`。M/N 可以跨多个输出 tile，但每个 program 必须包含完整 K；K 尾块补零，输出有效坐标完整覆盖且每个位置只写一次。操作使用 value-space 局部行列坐标，再通过共享布局解析到存储位置。

[矩阵乘法回归](../tests/test_interpreter_matmul.py)覆盖 float32/int32、非方阵、M/N/K 尾块、非连续输入，以及元素对齐的独立正/负 stride 输出和保护区。K 跨独立 program 却无累积、计算得到的操作数、额外 value 层级、潜在输入输出别名、部分字节重叠、输出重复写及额外 store/atomic 副作用均在支持合同外；现有负向测试明确检查 K 分拆、别名、重叠输出和嵌套额外 store 在写入前被拒绝且缓冲区保持不变。额外副作用递归检查嵌套 region，避免每个输出 lane 重复执行它们。

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
| f35 本轮功能 | 171 passed、23.72 s 为预备 CPU 组合；广范围 CPU 为 1 failed、294 passed、15 deselected、35.42 s、exit 1，修正后复验另记；GPU 15 用例待跑，新 A100 未验证 |

历史 4090 的 224/209/206/254/591/39 等也分别归于其源码、依赖和选择范围，见[历史报告](cpu_interpreter_validation_4090.md)。CPU、GPU、完整套件和诊断场景不累计。旧 squeeze 控制实验没有恢复原全量分配器与索引现场，后续成功仍不能确定旧失败的精确原因。

## 接下来必须完成的验证与演示

1. 保留 f35 广范围 CPU 的 1 项失败；执行 opcode 的断言应递归检查 blocks/regions，不能对包含 origins 的整个 metadata 做字符串包含判断。测试兼容修正后冻结新 SHA 并独立复验，保留确切命令、依赖、真实退出码和选择范围；171 项仍只作预备组合记录，不能称为全仓库通过。
2. 使用相同输入、布局、dtype、seed 和默认 pass，在可用 NVIDIA GPU 上实际运行 15 项差分并归档。若先用 RTX 4090，只记为 4090；新的 A100 对照仍需单独完成，不能继承 825 的 600 项结果。
3. 新增功能后的完整兼容性、格式、文档及示例验证按新提交留证。双卡场景继续单独注明，单卡不能代替同机多设备验证。
4. 核对真实历史缺陷演示及失败后导出/独立回放流程；现有故障注入、显式导出 API 和自动样例缩减是不同能力。新 operation 局部扩展及更丰富动态 shape 的证明也须有具体用例，不能只写计划或增加 dtype 名称。
5. 汇总全部约定实现、验证结果、仍不支持的输入和可复查演示，交用户验收。用户验收通过后，再按[贡献规范](../CONTRIBUTING.md)准备外部提交；当前不创建或发布 PR，不执行官网提交。

## 排除范围与提交约束

解释器不生成原生 CPU 代码，不以 CPU 执行性能为目标，也不模拟 warp/block 调度、shared memory 或 GPU race。Atomics、间接指针、多设备、随机数、float8 等平台相关语义不是本轮支持承诺；不支持时显式报错。其他项目的国产平台适配不作为九齿实现或验证证据。

正式 PR 的分支命名、标题、正文中非空 pytest 输出区块和工作流要求以现有 CONTRIBUTING 与[提交说明](cpu_interpreter_acceptance.md)为准；中文 commit 历史保留，不改上游规则。材料可以先准备，但必须先完成全部约定优化与验证并由用户验收，之后才能处理上游 PR 与官网提交。外部合并状态只按真实结果记录。
