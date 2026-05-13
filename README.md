# Diffusion Scarce Gen

基于扩散模型的稀缺图像生成工作流工具集，用于自动驾驶场景下的数据增强。

## 目录结构

```
├── scripts/                # Python数据处理脚本
│   ├── scarce_sample_filter.py   # CODA数据集稀缺样本筛选
│   └── coda_to_kitti.py          # CODA标注格式转KITTI格式
│
└── workflows/              # ComfyUI工作流 (JSON)
    ├── segment_background.json   # 抠图换背景 (GroundingDINO + SAM)
    ├── image_upscale.json        # 图像放大
    ├── flux_txt2img.json         # FLUX基础文生图
    ├── ipadapter_inpainting.json # IP-Adapter特征提取 + Inpainting局部重绘
    └── sdxl_multistyle.json      # SDXL多风格生成
```

## 数据集

本项目使用以下公开数据集：

| 数据集 | 说明 | 下载地址 |
|--------|------|----------|
| KITTI 2D Object | 自动驾驶目标检测基准，7481张训练图像 | [KITTI官网](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d) |
| CODA | 面向Corner Case的自动驾驶数据集 | [CODA官网](https://coda-dataset.github.io/download.html) |

## 相关项目

| 项目 | 说明 | 链接 |
|------|------|------|
| LaneDetect | 基于Ultra-Fast-Lane-Detection的车道线检测与掩膜生成 | [GitHub](https://github.com/X0H3M1/LaneDetect) |

## 快速使用

### Python脚本

```bash
pip install -r requirements.txt

# 稀缺样本筛选
python scripts/scarce_sample_filter.py

# CODA转KITTI格式
python scripts/coda_to_kitti.py
```

使用前请修改脚本中的数据路径配置。

### ComfyUI工作流

将 `workflows/` 目录下的 JSON 文件导入 ComfyUI 即可使用。

## 许可证

MIT License
