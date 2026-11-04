# 早期XR透视布设工具

保留`detect_lanes.py`、`barrier_placement.py`及依赖文件。安装`requirements/lane.txt`后可运行`python scripts/placement/barrier_placement.py --help`。模型和输入路径按参数提供。

两条工作流在`workflows/legacy_xr`：车道线补全及串行逐障生成。它们与当前网页原生路线分开归档。旧XR说明提到的标牌抹除文件未在当前备份中找到，此次没有补造该实现。透视位置来自简化道路假设，需按场景校正。
