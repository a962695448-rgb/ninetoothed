# 最终版本 A100 验收

最终结果：**1206 passed, 2 skipped in 1431.71s (0:23:51)**，另有15/15正式GPU差分和3/3强制跟踪目标通过。两项跳过均要求至少两张GPU，本次为单卡A100-SXM4-40GB。全部1208个collected node ID与逐阶段结果、JUnit集合已独立核对，未遗漏用例。

源码：a86bea9d32a43e122322acd52c0168b19682ffa2。203个GitHub blob逐项匹配；包括验收工具共209个冻结输入在运行前后SHA一致。最终结果11文件1711669字节，SHA256 eb8adb6afedcc7447b782708eb01b987514523524090eb2c889785a6089dd486。

## 运行与失败记录

- initial-linker-failure：镜像只有64位libcuda.so.1，缺少对应libcuda.so链接，Triton编译失败；保留失败输出和中止的原全库运行。
- ecc-failure：以任务私有目录补齐原驱动链接后，15差分和3跟踪通过；全库在卷积中遇到Xid48不可纠正显存双比特错误及Xid64重映射备用行耗尽，1失败并连带1094初始化错误，111通过、2跳过。不能将其记为通过。
- final-pass：平台迁移到另一台同型号A100，ECC累计与当前均0、行重映射正常；30GiB两种固定模式的显存读写检查通过，然后以同一源码和测试完整重跑。迁移保留磁盘和编译缓存，本轮是正确性验收，不是性能基准。最终ECC仍为0。

环境：Ubuntu22.04、Python3.12.7、Torch2.5.0+cu124（distribution元数据2.5.0）、Triton3.1.0、NumPy2.1.3、SymPy1.13.1、CUDA12.4、驱动550.127.05。实际依赖清单包含在ENVIRONMENT.json，初始pip约束错误与修正后的安装日志分别保留。没有调整代码、测试、容差或计数来获得通过。

## 恢复与复算

```bash
python3 recover_inputs.py restored-inputs
python3 recover_export.py final-pass/EXPORT_RECEIPT.txt final-pass restored-results
python3 audit_results.py restored-results/results --output independent-audit.json
# 两次失败可用相同recover_export.py分别恢复到全新目录。
```

源码ZIP包含203源文件及验收工具，PACKAGE_RECEIPT和INPUT_MANIFEST提供完整性验证；最终JSON可回溯每个测试ID、状态、跳过原因、环境和原始pytest日志。硬件诊断仅移除设备UUID/板卡序列号，REDACTIONS.json保留原始SHA。

全部输出回收校验后，GPU实例已在2026-09-19 11:13:11停止；云端临时磁盘的处理不影响此独立证据包。

PACKAGE_RECEIPT.json 是运行前的打包收据，其中 NOT_RUN 记录生成包时的状态；最终验收结论以 final-pass/SUMMARY.json（恢复后）和 FINAL_AUDIT.json 为准。
