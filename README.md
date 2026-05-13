# Diffusion Scarce Gen

基于扩散模型的稀缺图像生成工作流工具集，用于自动驾驶场景下的数据增强。

## 目录结构

```
├── scripts/                # Python数据处理脚本
│   ├── scarce_sample_filter.py   # CODA数据集稀缺样本筛选
│   ├── coda_to_kitti.py          # CODA标注格式转KITTI格式
│   └── LanDetect/                # 车道线检测与掩膜生成 (Ultra-Fast-Lane-Detection)
│       ├── detect_lanes.py       # 主脚本：批量生成二值掩膜
│       └── generate_cover.py     # 覆盖图脚本：车道线红色叠加可视化
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
| Ultra-Fast-Lane-Detection | LanDetect 基于的原始车道线检测模型 (ECCV 2020) | [GitHub](https://github.com/cfzd/Ultra-Fast-Lane-Detection) |

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

### LanDetect 车道线检测

```bash
cd scripts/LanDetect
pip install torch torchvision opencv-python numpy scipy gdown

# 生成车道线二值掩膜 (output/)
python detect_lanes.py

# 生成车道线覆盖图 — 红色车道叠加到原图 (cover/)
python generate_cover.py
python generate_cover.py --alpha 0.6 --lane-width 20

# 命令行参数
#   --input-dir   ./input         输入图片目录
#   --output-dir  ./output(cover)  输出目录
#   --model-path  ./culane_18.pth  模型路径 (缺失时自动下载)
#   --lane-width  25               车道线宽度 (px)
#   --alpha       0.45             覆盖图透明度 (仅 generate_cover.py)
#   --no-cuda                      强制 CPU
```

模型基于 Ultra-Fast-Lane-Detection (ECCV 2020)，使用 CULane 预训练的 ResNet-18 backbone。

### ComfyUI工作流

将 `workflows/` 目录下的 JSON 文件导入 ComfyUI 即可使用。

## 许可证

MIT License
