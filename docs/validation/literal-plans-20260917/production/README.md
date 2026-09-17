# 已保留的生产变更

仅新增表达式语义测试，所有 src 文件逐字节等于 4d50b0424d963ac7ea3fd67fb06ff2c828fb3d92。

`python prepare_inputs.py --baseline /path/to/base --work /tmp/production-replay` 校验基线并加入测试；随后运行 `python /tmp/production-replay/candidate/nine/scripts/run_cpu_tests.py`。最终结果 770 passed / 15 GPU deselected / 35.40s，日志在上级 results/production-cpu-suite.*。
