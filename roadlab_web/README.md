# RoadLab 网页模块

从仓库根目录运行`python roadlab_web/app.py`。完整步骤见[部署说明](../docs/deployment.md)。

`config.py`集中处理目录默认值；无需原服务器目录。任务保存在`.roadlab/data`，缓存保存在`.roadlab/cache`，可通过环境变量覆盖。网页生成图由`app.graph`和`workflows.native_graph`构建，不直接加载界面工作流JSON。

原生路线的`result-generated.png`已完成ROI合成，处于原有人车恢复之前，不能直接称为未经合成的VAE输出。保护范围来自模型预测，可能漏检；遮罩与计划矩形不是目标真值。

运行`python ../scripts/run_checks.py`执行离线CPU检查。真实GPU生成与ComfyUI版本兼容应按部署说明另行验证。
