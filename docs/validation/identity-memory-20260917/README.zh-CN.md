# 恒等布局内存与标量掩码验证档案

基线：a962695448-rgb/ninetoothed 的 `3c0ceccf540233fbe2f23656f71b69268dd811e1`。
`measurements` 保存三轮独立进程的完整记录和判定；`checks` 保存 577 项 CPU 回归、原版标量掩码失败以及风格/文档检查；`gpu` 保存 15 项真实 GPU 差分及 42 项新增边界测试。

`source` 是实际测量的字节版本，`final-source` 只按项目规范补齐空行。二者 Python AST 相同，`checks/format-only.json` 记录对应关系；最终格式版另通过完整 577 项 CPU 回归。首轮部分计时与本地回归重叠，报告仅将 36%–42% 的分配峰值下降作为主要收益，且不称为 RSS 或 Triton GPU 算子收益。

## 复现

先检出上述基线提交，然后在无 Torch/Triton、具有 NumPy/SymPy/pytest 的环境中，从本档案目录执行：

```bash
python prepare_inputs.py --baseline /path/to/ninetoothed --work /tmp/nine-identity-replay
python /tmp/nine-identity-replay/tools/confirm_nine.py
```

准备工具逐个验证基线 Git blob，再复制实测的 source 覆盖，输出目录必须不存在。若检查最终格式版，在准备完成后以 final-source 覆盖 candidate/nine，再执行该项目的 scripts/run_cpu_tests.py。

原始 JSON/log/XML 内容保留，绝对路径仅是运行环境记录。全站 Sphinx 在本地缺少 Torch 的 CPU 环境中失败；CPU 指南与安装说明的专项严格构建通过，不把两者混记。
