# 本次提交验证

日期：2026年11月4日。

本地使用Windows与Python 3.12.10，从仓库根目录运行`python scripts/run_checks.py`：原有及审核导出测试22项、新增可移植性与实验工具测试5项，共27项通过，无跳过。

另完成`python -m compileall -q roadlab_web scripts experiments tests`、`python -m pip check`以及四类API图导出。跨目录启动测试实际启动独立网页进程、读取页面和静态资源，并验证未连接ComfyUI时健康检查返回未就绪。

核心依赖：

- `aiohttp==3.14.5`
- `numpy==2.5.4`
- `Pillow==12.3.0`
- `tqdm==4.70.1`

这些版本记录该次CPU验证，不是完整CUDA环境锁文件。API测试使用隔离任务和模拟ComfyUI响应，没有提交真实GPU生图，也未重新验证外部模型推理。GitHub Actions另配置Python 3.10/3.12的Linux检查，其状态以对应提交的Actions结果为准。

部分aiohttp字典状态键会发出迁移建议警告；测试均通过，后续可在保持接口兼容的前提下迁移AppKey。
