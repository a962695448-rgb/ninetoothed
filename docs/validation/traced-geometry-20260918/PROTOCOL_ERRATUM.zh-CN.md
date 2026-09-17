# 协议文案更正（采样期间发现）

原PROTOCOL.json的extra_limits继承了上一阶段的“or tracing”禁用描述，与本轮change、trace=True目标及已固定源码的范围不一致。它是陈旧说明，不参与bench/confirm的配置或数值判定。

原文件及其SHA保持不变，所有轮次仍使用原配置。另存PROTOCOL.prose-corrected.json，仅修正extra_limits文字；源文件、目标与控制、7×3样本、顺序、三轮、1.10倍/5%退化及新增5%峰值门槛均未变更。不删数据、不补跑到通过。

此更正在采样期间作出，报告需明确披露，不能称原协议的所有说明自始一致。恢复与复算仍以原PROTOCOL.json为实际运行输入。
