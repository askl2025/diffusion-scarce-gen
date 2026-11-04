# 模块与数据流程

`app.py`连接上传、参数校验、异步串行队列、ComfyUI通信、结果收集及HTTP接口。`object_profiles.py`定义正式四类入口。

`mask_worker.py`调用`placement.py`组织手动区域或车道输入。`preprocess_cache.py`按图像、车道代码和权重指纹缓存折线；线宽和扩展参数改变时重新画遮罩。

`workflows.py`构建原生Fill条件、采样、解码和ROI合成。交通锥和垃圾桶的旧采样分支继续由`app.graph`构造。

`scene_service.py`保持CPU分析子进程，`scene_analysis.py`预测原有人车并进行恢复合成；仅对预测区域生效。`experiments.py`记录参数、哈希与审核来源，导出时要求完成、成员明确通过和实际目标框。

`static`提供选位、遮罩/结果预览、任务进度、取消、历史恢复和审核交互。编辑遮罩、计划位置、模型预测保护区与最终目标框属于不同信息，不能相互替代。
