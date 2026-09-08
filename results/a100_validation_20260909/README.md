# A100 验证原始证据

源码：`3fc27e0399d82b02be5407b9542c67763c4631a7`。本轮 **GPU 差分和 CUDA dot 通过，完整套件中断，整体不能标为完整验收通过**。

- [GPU 差分](gpu-differential/interpreter_gpu_validation.json)：15/15，9个程序、10个类别。
- [CUDA dot](cuda-dot/report.json)：四路对照通过、输入和保护区未变。
- [完整套件原文](full/validation.stdout.log)：收集705项，中断前84 passed、2 skipped、2645.26秒，exit 2。
- [完整套件清单](full/manifest.json)：原 FAIL 与缺失产物记录保留。
- [JUnit](full/junit.xml)：suite 完成计数86；另有1个匿名中断元素，不计为通过。
- [独立核验摘要](verified_summary.json)：`verified_pass=false`，不能将部分结果读成完整PASS。
- [归档清单](archive_manifest.json)：公开53份技术文件的字节数及哈希。

两个跳过项要求多GPU。运行从独立编译缓存开始，后在执行窗口结束前由控制器中断，保存原始日志。输入、内核、测试范围和容差没有为获得通过而修改。完整过程及后续步骤见[验证说明](../../docs/a100_validation_20260909.md)。

本地完整ZIP含55份原始文件，SHA-256为 `d5497b8d58cfa3079332524ba9779d4a945c389c0669339e5bbb71e7eeebf41c`。公开文件是经过逐字核验的子集，另附实际运行控制器；编译产物和资源操作细节留在本地完整备份。manifest 中的绝对路径是当时环境，换机器复跑需调整路径。
