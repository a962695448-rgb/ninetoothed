首轮使用 whole_tensor 标志在读取 helper 内判断恒等布局。目标三轮几何平均 7.476/6.757/7.268 倍，但第二轮 vector trace 对照为 0.94774 倍，超过 5% 退化门槛，判为 REJECT。保留源码和 results/nine-first 原始数据；后续把判断移到 read 入口，并让索引提取与映射读取走轻量分支，保持同一数值和时间门槛。
