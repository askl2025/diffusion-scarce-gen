# 多物体与逐位置实验

该目录是独立实验工具，不改写正式物体入口。准备阶段可在CPU运行，不需要模型权重；生成仍依赖ComfyUI和FLUX Fill。

```bash
python experiments/multi_object/prepare.py --image path/to/input.png --output .roadlab/experiment-01 --case-id experiment-01 --strategy sequential --boxes '[[0.1,0.6,0.2,0.3],[0.6,0.6,0.2,0.3]]' --prompt 'One stone ball standing on this pavement.' --seed 42
```

框为归一化`[x,y,width,height]`。逐位置输入尺寸须是16的倍数，使用单物体提示词；联合模式使用包含数量要求的提示词。输出保存输入、mask、blend、API执行图、规划参数和哈希。`--comfy-input /path/to/ComfyUI/input`可复制文件，已有目标不会覆盖。

归档49组记录见`../evidence`，可辨别1次与多次采样、不同标签及耗时口径。两个开发背景重复使用，不是独立留出测试。
