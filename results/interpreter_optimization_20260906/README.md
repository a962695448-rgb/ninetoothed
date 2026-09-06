# 九齿6ec解释器复验与资料收尾

计算源码：`6ecce58da28bb9709aa35fc6c25c1f361aff736f`。后续仅将交互可视化的Tk/调试依赖移到交互入口，并更新文档和本归档；GPU结果不冒充新提交的另一轮运行。

| 独立范围 | 实际结果 |
|---|---|
| f35 NumPy CPU选择 | 1 failed、294 passed、15 deselected，exit1；原失败保留 |
| 56 NumPy CPU复验 | 295 passed、15 deselected，34.18s，exit0 |
| 6ec NumPy CPU选择 | 307 passed、15 deselected，32.68s，exit0；无Torch/Triton |
| 6ec RTX4090 Triton | 15/15，9程序、10类别，exit0；见gpu-6ecce58/triton-report.json |
| 6ec RTX4090 CUDA dot | 一个float32标量多program尾块probe，exit0；见gpu-6ecce58/cuda-dot/report.json |
| Sphinx两次历史尝试 | 均exit2，分别缺matplotlib与tkinter |
| 本地无GUI修复后首次Sphinx | exit2，进一步暴露缺Torch；原日志保留，未记为成功 |

这些范围不相加。15个Triton用例已经包含同类标量dot，CUDA probe是独立后端验证，不是新增程序、完整库或性能基准。新版A100、同机双卡、完整仓库重跑、Tensor Core、任意布局/split-K仍未验证；上游PR与官网提交等待用户验收。

GPU传输包37文件、717096字节，SHA256 `42f4dd0f21087b6bfaa6cb8112435c812f822fd9b40f9a1c21bae981a3b8bb42`。全部对象已核size/SHA；公开保留26份原始源码/报告/数组/日志，另11份编译二进制、缓存与锁保留在私有原包，详见gpu-6ecce58/archive_manifest.json。CUDA二进制SHA已对原包实际重算。公开原文件没有改字节；.gitattributes关闭换行转换。

CPU、Sphinx与GPU各目录的manifest记录命令、环境、源码/工作文档摘要和原始日志。独立复验不会覆盖旧FAIL；此处不宣称全局最优或自动通过训练营评选。

## 最终CPU文档构建

服务器既有环境运行原Sphinx `-W --keep-going`：exit0、28页、5.392s、stderr空，源码为 `cc190a1653eb99878cd21a59eac8a69ab8302684`。所有原有构建输入SHA已与该Git提交逐一核对，Sphinx只正常新增14份autosummary RST；控制器误把新增文件当作输入漂移的原FAIL保留，独立postcheck另记PASS。完整117文件包为sphinx-final-20260907/raw-html.zip，SHA256 `f9f0a372f29c1efff2f572c7ad696ff98e53d96d1fa5d8a2044ca0b0c37a5a3e`。

**静态资源限制：服务器HTML中的logo是Git LFS指针，不能称资产完整。** 本地准备包有真PNG，但本次未成功上传到服务器；原HTML包未改写，正式发布前需恢复真实logo。文档未部署，上游PR与官网仍等待用户验收。
