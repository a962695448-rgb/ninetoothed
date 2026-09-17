# NineToothed 布局重复计算剖析

使用已发布90197b0的源文件，仅在独立本地进程中包裹TensorRef._access计数，结束恢复；没有修改生产文件。向量、softmax和带尾块的matmul均与NumPy结果核对。插桩耗时不作为性能成绩。

- vector4099：66次访问、66个引用对象、66个基本整数绑定签名，无重复。
- softmax13×129：39次访问、26对象、26签名，最大复用2次。
- matmul9×32×11：6471次访问、54个引用对象；其中6372次采用基本整数绑定及布尔额外mask，只有15个几何签名，单个签名最多出现1411次。另99次使用非标量mask，未计入候选签名。此计数只表示重复，不证明可以安全缓存。

源码显示scalar matmul初始化已验证矩阵坐标与掩码的关系，但后续extract仍会再次构造整个逻辑tile的地址图。后续应围绕解释器单次执行内的几何复用验证，不缓存数组数值、不以NumPy matmul替代实际SSA循环。

实现前必须处理：符号/shape变化、任意IndexExpr常量与自定义数字对象、NumPy警告/错误顺序、输出/输入别名、trace与memory observer语义、坐标可写性及对象释放。已有primitive signature相同不足以证明这些条件。可以考虑只对可证明纯整数且无溢出/除零的表达式建立执行内缓存；不应直接全局缓存TensorRef结果。

结果和源码SHA256在geometry-counts.json。复现命令：`python profile_geometry.py --source /path/to/ninetoothed-90197b0 --output /tmp/new-geometry-counts.json`。脚本先按SOURCE_MANIFEST.json核对70份源码，再执行计数；输出路径必须不存在。当前只有剖析证据，尚未实现缓存或宣称加速。

本轮在无Torch/Triton的Python3.12.14、NumPy2.3.5、SymPy1.14.0环境中复现相同计数。没有调用GPU或修改现有解释器。
