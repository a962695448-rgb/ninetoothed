# 九齿优秀项目差距与执行计划

审查日期：2026-09-06。源码 **`82592b8f6de65052e4258fdd6067956d4ede18c3`** 已在实际 A100-SXM4-40GB 上完成完整测试：**600 passed、2 skipped，450.25 s（0:07:30），退出码 0，无 failures/errors**。两个 skip 均要求同机至少双卡，见 [最终完整清单](../results/full_suite_a100_82592b8/manifest.json)。A100 完整运行的工程证据已取得，当前重点是上游审查、官方确认和既有可选加分项。b5/377 失败历史及 14/180/16/77 等各范围不累计、不重标版本；成功重跑不确定旧 squeeze 的唯一根因。

规则依据：训练营[九齿 CPU 参考解释器与差分调试器任务](https://gxtctab8no8.feishu.cn/wiki/GxhWwiz0iiAhKkk7CFhcLTCyn2b)的 2026-09-05 本地阅读快照；实施依据为本仓库源码、测试、[A100 证据报告](cpu_interpreter_validation_a100.md)、[4090 历史报告](cpu_interpreter_validation_4090.md)与 [贡献规范](../CONTRIBUTING.md)。

## 当前提交重点

`377daec` 的 jagged 测试参考修复已通过定向和第二轮全量中的全部 16 项，但第二轮仍因 squeeze allclose 失败退出 1。后续在原 fixture 上、人工 seed 2026 的受控诊断为自然分配 PASS、NaN/唯一索引 FAIL、有限 -123/唯一索引 PASS；NaN 控制中两矩阵逐位一致，129792 个 NaN 全在未写区域。旧现场未保存分配器状态和索引，因此这只是机制证据，不能认定唯一根因或称为原失败精确重放。

`82592b8` 仅用有限非零输出初值与 randperm 唯一随机目标稳定两条测试输入语句，并添加说明注释；kernel、`src/`、参数、全矩阵比较、原容差与依赖均不变。generation 文件 77 项和随后完整测试 600 passed、2 skipped 均已取得证据。此前两次全量失败、libcuda smoke 失败和受控诊断分别保留；最终通过不改变其原始结果或诊断边界，详见 [A100 报告](cpu_interpreter_validation_a100.md)。

本轮使用 Python 3.12.7、PyTorch 2.5.0+cu124、Triton 3.1.0、NumPy 2.1.3、SymPy 1.13.1，修复没有升级依赖。最终完整运行源码为 `82592b8`。后续只有 `docs/`、`results/` 归档的提交，若代码、测试和依赖均与该版本相同，应说明对应关系并引用原运行，不把资料提交描述为另一次全量实测。

单 A100 不会自动解决多 program 标量分解和跨结构 pass 追踪；这两项主要需要开发时间和可审查设计。

可选再安排一次两张相同计算能力 NVIDIA 卡的短时验证，例如同机 2×4090 或 2×A100，用于原仓库的跨设备 AOT/缓存重载回归。`test_aot.py` 要求至少两张卡，`test_built_artifact_reload.py` 还检查计算能力一致。两台各一张卡不能等同于这一环境，4090 与 A100 混搭也不能覆盖要求计算能力相同的那一项。双卡不是 CPU 解释器题目的明确必需条件，也不要求 NVLink。

国产服务器不是本题快照中明确列出的验收设备或加分条件。若另一 Hadamard 项目需要国产适配，可单独准备；不能将该适配算作九齿已完成的优秀项。

## 逐项对照

| 标准 | 当前证据 | 差距与合格证据 |
|---|---|---|
| 原有测试、风格、文档保持可用 | 825 A100 最终完整测试 600 passed、2 skipped、exit 0；历史 b5/377 失败、14/180 专项、16/77 回归与文档检查分别归档 | 两个跨设备 skip 仍未验证；按最终资料提交检查格式、文档和证据对应关系，再交上游审查。受控诊断与旧失败边界保留 |
| CPU-only 与五类应用、三种必需 dtype | `76ca646` 隐藏 CUDA 后 209 passed；`5b37725` 的 WSL 无 Torch/Triton 环境 206 passed、14 deselected；应用与 dtype 对照见验收说明 | 两轮依赖、源码和选择范围不同。新 CPU 记录未验证 PyTorch CPU Tensor 适配；广泛收集时的缺包错误另保留，不能将 206 与 209 相加或直接比较 |
| 默认 pass 前后至少三个程序一致与 A100 差分 | b5 A100 独立报告有 8 程序、14 项四方比较和实际默认 5 pass；825 最终完整测试已通过，原报告仍按 b5 保存 | 硬件验证证据已取得；原 GPU JSON 的旧静态提示保持原文，由 A100 报告解释。进入官方与上游审查，不冒称审查结论已获得 |
| dot/matmul、softmax 和扩展能力 | softmax 已有 A100 真实 GPU 对照，最大绝对差约 `8.94e-8`；直接 dot 与单 program 分解有 CPU 测试；已支持 strided 视图等 | 多 program 分块 dot 被明确拒绝，A100 GPU runner 仍排除优化后 dot；补齐真实默认管线中的代表性矩阵乘法 |
| trace 过滤、单步、断点、watch | 有实现与自动化测试；新增 reference/candidate 独立回放、负向控制和公开演示包，服务器 CPU 定向 39 passed | 现有完整演示是故障注入；继续补真实历史缺陷的前后对照，保持跨 program/循环作用域及 trace 一致性验证 |
| 自动定位首个错误 pass 和 operation | 每个 pass 可与原始程序比对；结构与事件一致时能定位 operation | 重构 pass 后缺少来源映射，operation 位置会明确为空；不能宣传普遍精确定位，需要限定合同或补来源追踪 |
| 导出 SSA、输入、shape、dtype、seed | JSON/NPZ 和回放代码已有实现；保存 strides、权限与同对象别名 | 用一个真实历史缺陷导出并独立回放，证明输入布局问题没有在复制时消失；自动缩减不是规则明确必需项 |
| 可扩展、复用 | operation handlers 与 frontend/运行时接口已分离 | 用一个独立新增 operation 的局部实现示例验证扩展接口，并保留 unsupported 报错 |
| 合并主分支 | 已有用户 fork 与分支 | 尚未创建上游 PR；合并取决于维护者，必须预留设计讨论、CI、修改与审查时间 |

224 项已经包含历史 14 项 GPU 对照；209、206、254、591、39 各属前述不同范围。A100 smoke 包含于 b5 的 14 项 GPU，14 又包含于 180；b5 全量 602 = 584 passed + 16 failed + 2 skipped，377 全量 602 = 599 passed + 1 failed + 2 skipped。377 定向 16、825 generation 77 和诊断控制均不与这些数字累计。准确命令和证据入口见 [A100 报告](cpu_interpreter_validation_a100.md)、[4090 报告](cpu_interpreter_validation_4090.md)与 [验收及提交说明](cpu_interpreter_acceptance.md)。

## P0：整理证据并进入上游审查

1. `5b37725` 完整运行已完成并归档：591 passed、2 skipped、退出码 0，归档时 HEAD 正确且已跟踪文件无修改。[完整运行清单](../results/full_suite_rtx4090_5b37725/manifest.json)已核验，演示改进 `4a680a6` 已整合；核心、依赖和测试配置未变，修改文件与39项定向验证的散列一致，故不重复整轮4090测试。旧失败、SIGSEGV与FP8单例保留；本轮成功没有确定SIGSEGV根因。
2. 825 最终 A100 全量为 600 passed、2 skipped、450.25 s、退出码 0，完整原文见 [最终归档](../results/full_suite_a100_82592b8/README.md)。b5/377 的 [初轮失败](../results/full_suite_a100_b5a3206/manifest.json)、[第二轮失败](../results/full_suite_a100_377daec/manifest.json)、14/180 专项、16/77 回归及 [受控诊断](../results/squeeze_fixture_diagnosis_a100_377daec/README.md)全部保留，成功结果不确定旧失败唯一根因。
3. 提交材料应直接链接实际设备、依赖、测试 SHA、原始命令、日志/JUnit/coverage 和散列；两项同机双卡 skip 逐项说明。后续资料提交与 825 的代码、测试和依赖对应关系必须明确，实际测试 SHA 不变。
4. 将已具备证据的主线提交上游审查和官方确认，回应实际 review；多 program dot、跨结构来源追踪等可选增强按后续独立改动推进，不把合并或评选结果写成已完成。

当前状态：真实 A100 专项与最终完整运行证据已取得。提交阶段的完成条件是材料与代码/测试版本可核对、第三人能按记录复验、上游和官方审查得到实际处理；审查、合并及评选结论仍由相应维护者和训练营决定。

## P1：让矩阵乘法成为真实的优秀项

目前 `runtime.py` 对分解后的标量输出要求单个输出 store、单逻辑 program；默认目标管线生成的多 program dot 会被提前拒绝。这是正确的边界防护，但不是完整分块矩阵乘法支持。

建议先做一个边界清楚的完整合同：二维输出、多 program M/N 分块、一个矩阵结果、显式 K 归约，覆盖非方阵和 M/N/K 的非整除尾块。先写明 program 坐标、tile 内坐标、源全局坐标和 reduction 坐标的关系，再实现共享布局映射，避免复制一套与发射器不同的地址规则。每个有效输出坐标必须只写一次，inactive 地址不能读取；不能通过取消默认分解或回退前端 SSA 绕过失败。

证据至少包含：

- 对齐分块、M/N 尾块、K 尾块、非方阵和多个 program；选择少量能区分错误类型的形状，不堆积重复尺寸。
- 原始 SSA 的 CPU 结果、每阶段默认管线后的 CPU 结果、实际发射 SSA 对应 GPU 结果、独立 NumPy 参考四方比较。
- 固定 K 与符号/动态 K 的运行合同分别验证；int32 先验证 CPU 精确语义，GPU 精确整数 dot 只有后端确实支持时才承诺。
- 保护区、输入未修改、trace 开关一致、program/lane 可定位；失败时自动导出材料。
- 至少一个 A100 上真实运行的多 program float32 dot 用例。明确 GPU 后端的矩阵乘法精度策略，不能以放宽题目阈值替代修复。

停止标准：合同内的典型布局与尾块全部通过，合同外仍明确拒绝；不为拿“更多功能”引入 batched matmul、任意复杂别名或所有 dtype。

## P1：让首次错误定位经得起重构 pass

现有 `compare_programs` 只有在 `_structure` 相等且事件序可对齐时比较对应中间结果。正常优化可能删除、合并或展开 operation，此时不猜位置是合理的。提升点应是保留来源信息，并解释“首次观察到数值差异”和“最初造成错误的源码位置”的区别。

建议给 frontend SSA operation 分配稳定来源标识，由主要变换 pass 显式传播一对一、一对多、多对一映射。在每阶段输出比较仍与原始参考保持一致的基础上，用来源标识和 program/region/iteration/lane 对齐可比语义边界；无法一对一证明时返回最小相关 operation 集合及原因，不伪造唯一位置。

先覆盖 canonicalize 和 linalg decomposition 等项目确实使用的 pass，保留不支持来源映射的明确降级。验证包括：一个结构不变的坏常量、一个合法结构重写、一个结构变化后产生错误的 pass，以及循环/分支实例不相同的情况。故障注入应标成故障注入，不能包装为新发现的真实上游缺陷。

停止标准：首个错误 pass 稳定可复现；结构变化的代表性缺陷能追到有来源证据的 operation 或最小集合；合法重写没有误报。真实历史广播或掩码缺陷应另附复现，展示工具的实际价值。

## P2：按收益选择加分增强

优先补“同一个已构建解释器 handle 运行多个动态 shape”与非连续/负 strides 的端到端用例，或一项实际后端能比较的 dtype 扩展。新增一种数据类型必须有输入合同、算术舍入/溢出语义和独立参考，不能仅在 dtype 白名单中增加名称。

自动失败样例缩减可作为后续增强；官方快照要求可复现材料，未强制自动缩减。若做缩减，应以失败谓词保持为停止条件，并保存缩减前后 shape/操作数变化，不以“导出已有样例”冒称最小化。

CPU 执行速度不是本题目标，不优先做多线程 CPU kernel、GPU 并发模拟、原生代码生成或 GPU race 模拟。工程时间优先投入解释语义、真实失效定位和可审查性。

## 提交流程与优秀评选风险

贡献规范要求 kebab-case 分支；合规提交分支 `add-cpu-reference-interpreter` 与开发分支 `feat/cpu-reference-interpreter` 均已建立并推送，已整合演示改进与 4090 完整测试归档。正式 PR 使用前者，发布前以远端引用核对确切提交。4090 完整运行记为 `5b37725`；A100 初轮专项及失败全量记为 b5，jagged 定向 16 和第二轮失败全量记为 377，generation 文件 77 和最终第三轮完整通过记为 `82592b8`。后续只有资料的提交也不改写这些实际运行 SHA。

用户要求中文 commit 备注，后续保留该偏好，不重写已推送的中文提交。`.githooks/commit-msg` 是需在本地显式启用的 hook，其英文首字母要求不能直接解释为远端对历史 commit 的检查。实际远端 `.github/workflows/contributing.yml` 通过 `--event` 检查 PR 的 title、head branch 和正文中的 pytest 输出，没有遍历历史 commit messages。PR 标题按英文大写开头、命令式、无结尾标点的规则准备，正文可用中文并附实际 pytest 输出；不修改上游规则，也不以本地 hook 惯例覆盖用户语言偏好。

建议先形成可审查主线 PR，再将已确认独立的编译器 bug 修复按依赖拆为小补丁，避免为了拆分制造相互不兼容的分支。每个 PR 必须说明问题触发方式、修改后的行为、测试环境及尚未覆盖范围。

`.github/workflows/pytest.yml` 会跳过来自 fork 的 PR GPU job，push 触发的任务则要求带有 `nvidia`、`ninetoothed` 等标签的 self-hosted runner。必须附可复查的真实测试证据，并由维护者决定如何在其 runner 验证；GitHub 上该 job 的 skipped 状态不代表 GPU 测试通过。提交内容与各工作流的实际检查范围见 [验收及提交说明](cpu_interpreter_acceptance.md)。

合并和优秀评选由训练营及维护者决定。可控的完成条件是准确实现、完整证据、及时回应 review、实际修复 CI 和真实用户可用性；不是用服务器数量或测试数量替代这几项。
