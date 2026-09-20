# 截止前最终针对性 A100 验收

受测代码150f5268977174164db383a75254f541aad14ffa，基于已同步官方#218的f44ca621。额外修复tensor.cast的字符串dtype属性引号解析：原版28项失败、2控制通过；修复后新增30项全部通过，NumPy-only完整范围829通过、15实际GPU用例排除。

实际A100-SXM4-80GB单卡，Torch2.5.0+cu124、Triton3.1.0、NumPy2.1.3、SymPy1.13.1、Python3.12.7、CUDA12.4、驱动550.127.05：

- 正式GPU差分15/15，通过原始/目标SSA、Triton GPU与独立参考对照。
- 强制trace目标3/3通过。
- 整数别名64配置、96值检查全部通过，覆盖8种signed/unsigned宽度、标量参数和显式转换、Triton/CUDA、JIT/AOT；32个AOT配置另完成48次重载值检查。
- 相关pytest输出：`127 passed in 3.81s`，无跳过，含先前Mac上排除的CUDA张量拒绝测试和新增字符串转换回归。

这是针对性GPU验收，不是当前版本全库GPU回归。9月19日1206通过/2多卡跳过仍对应原a86bea9，不累计或冒充本轮全库数量。

207候选文件及工具共215份冻结输入在运行前后SHA一致；12份输出199904字节，SHA256 c400dfb605eb0b550d4568d3e59196a6c7ccfcf176f48e4ef5760a11a2adcdac，回收后已逐分片/gzip/文件校验并独立核对全部测试集合。GPU不可纠正ECC当前/累计均0；历史可纠正重映射为1、无pending/failure，60GiB双模式读写预检通过，不称完整硬件认证。CUDA生成代码的unused-variable警告保留原样，不影响数值通过。

## 恢复与核验

```bash
python3 recover_inputs.py inputs-restored
python3 recover_export.py EXPORT_RECEIPT.txt . results-restored
python3 audit_results.py results-restored/results
```

恢复器验证ZIP和215个输入；输入包含全部源码、验收编排和独立别名检查。别名离线预检在原版的失败也保留，`CPU_ONLY`只表示CPU预检，不冒充GPU通过。实际GPU文件以EXPORT_RECEIPT恢复的结果为准。JUnit和硬件诊断仅做所列隐私字段脱敏，原始哈希保留。

数据备份核验后，GPU在2026-09-20 16:03:18停止，临时云盘随后释放。后续仅整理验收文档，不继续性能优化。
