# 复现顺序

先运行`python scripts/run_checks.py`，再配置真实模型和ComfyUI。离线检查只验证软件、图结构和像素合成，不提交GPU任务。

网页API图由`python scripts/export_workflows.py`从当前图工厂生成。每组结果需对应实际输入、编辑mask、blend、提示词、种子、步数、guidance、采样器、scheduler、LoRA及代码版本。原生分支的`result-generated`已经经过ROI合成。

多位置实验用`experiments/multi_object/prepare.py`准备独立案例。指定新的输出目录、归一化矩形、提示词与策略；逐位置模式要求图像宽高均为16的倍数，并使用单物体提示词。后续位置的种子为基础种子加位置序号。区域合成结果作为下一位置输入，与归档流程一致。

准备工具默认不提交任务。可用`--comfy-input`复制输入到共享目录，再以ComfyUI的`/prompt`接口提交`{"prompt": 执行图}`。API图不是界面编辑器工作流，见[格式说明](../workflows/README.md)。

实际目标需逐图检查数量、形体、接地与背景；正式训练前还需独立标注和留出测试，不能由计划矩形直接生成训练真值。
