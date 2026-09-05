# A100 已完成阶段的原始证据

测试源码固定为 `b5a3206f8351e5a138d16ee13f6d6ef9c620044b`。
完整结论、环境和复现步骤见 [A100验证报告](../../docs/cpu_interpreter_validation_a100.md)。

- `smoke/`：首次链接失败，退出1；保留原始记录。
- `smoke-libcuda/`：修正实际64位驱动库链接后，同一用例1 passed。
- `gpu-report/`：实际A100差分14/14通过，8个程序、9类用例。
- `specialist/`：解释器专项180 passed，包含上述GPU用例，不能累计。
- `setup/`：实机预检、安装包清单、依赖检查及项目内驱动链接修正。
- `reconstructed-ssa/`：在无Torch/Triton环境独立CPU重建的14份SSA，与已记录A100哈希逐项匹配；不是新增GPU运行或当时导出的原文。

每次运行的 `manifest.json` 保存精确argv、实际设备与依赖、运行前后源码SHA、
受控环境、子进程退出码及文件散列。[archive.json](archive.json) 是本归档的字节与SHA-256索引。
原始JSON的旧静态提示按原文保留，由验证报告解释，不修改数据来匹配新叙述。

本目录不包含仍在运行的A100全量结果。应在该进程结束后另行归档，不能将602个收集条目当作通过数。
归档中的导出脚本以 `.py.txt` 保存；需要复现CPU重建时复制成 `.py` 文件，使用 `--help` 指定独立b5 checkout、GPU报告和新输出目录。
