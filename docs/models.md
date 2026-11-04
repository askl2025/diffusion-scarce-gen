# 外部模型清单

仓库仅发布推理接入，未发布LoRA训练代码、训练日志或权重。

| 文件/目录 | 用途 | 获取与放置 |
|---|---|---|
| `flux1-fill-dev.safetensors` | FLUX Fill基座 | [原发布方](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev)，按ComfyUI模型目录约定放置 |
| `clip_l.safetensors`、`t5xxl_fp8_e4m3fn_scaled.safetensors` | 双文本编码器 | 使用与FLUX节点兼容的编码器文件，保证`/object_info`列出对应文件名 |
| `ae.safetensors` | VAE | 使用FLUX兼容VAE，并核对发布来源 |
| `FLUX_CodaRoadBarrier_v1.0.safetensors` | 交通锥路线外部LoRA | 需自行提供；本仓库没有公开下载地址，不影响警示牌与行人原生路线 |
| `culane_r18.pth` | CLRNet R18推理 | [CLRNet](https://github.com/Turoad/CLRNet)发布的匹配权重，放入`scripts/LanDetect/_vendor/weights` |
| `culane_18.pth` | 旧UFLD推理 | [UFLD](https://github.com/cfzd/Ultra-Fast-Lane-Detection)匹配权重，旧脚本的下载方式保留 |
| `segformer-b0-cityscapes/` | 预测原有人车保护区域 | [NVIDIA模型](https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-1024-1024)，保存配置、预处理配置及模型文件至`ROADLAB_SCENE_MODEL` |

模型通过各自发布渠道获取并遵守对应使用条款。生成前核对实际文件SHA256；本次代码更新没有GPU或模型权重，不把历史权重检查写成新近重跑结果。
