# LaneDetect

基于 [Ultra-Fast-Lane-Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection) (ECCV 2020) 的车道线检测与掩膜生成工具。

将道路图片输入 `input/` 文件夹，识别车道线，生成二值掩膜（车道区域白色，背景黑色）保存至 `output/`；或将检测到的车道线以红色叠加到原图，生成覆盖图保存至 `cover/`。

## 效果示例

| 原图 | 掩膜 |
|------|------|
| sample001.jpg | sample001_mask.png |

## 环境要求

- Python 3.8+
- PyTorch 2.0+ (CUDA 可选)
- torchvision 0.15+
- OpenCV 4.x
- NumPy, SciPy

```bash
pip install torch torchvision opencv-python numpy scipy gdown
```

## 快速开始

```bash
# 处理 input/ 下所有图片，掩膜输出到 output/
python detect_lanes.py

# 生成车道线覆盖图 — 红色车道叠加到原图，输出到 cover/
python generate_cover.py
python generate_cover.py --alpha 0.6 --lane-width 20
```

首次运行会自动从 Google Drive 下载预训练模型 (~85MB，解压后 178MB)。

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input-dir` | `./input` | 输入图片目录 |
| `--output-dir` | `./output` / `./cover` | 输出目录（掩膜 / 覆盖图） |
| `--model-path` | `./culane_18.pth` | 预训练模型路径 |
| `--lane-width` | `25` | 车道线宽度 (px) |
| `--alpha` | `0.45` | 覆盖图透明度，0-1（仅 `generate_cover.py`） |
| `--no-cuda` | 否 | 强制使用 CPU |

## 项目结构

```
LaneDetect/
├── input/              # 输入图片
├── output/             # 输出掩膜 (*_mask.png)
├── cover/              # 输出覆盖图 (*_cover.jpg)
├── detect_lanes.py     # 主脚本：车道线检测与掩膜生成
├── generate_cover.py   # 覆盖图脚本：车道线红色叠加可视化
├── culane_18.pth       # 预训练模型
└── .gitignore
```

## 原理

1. 将图片缩放至 288×800，使用 ImageNet 统计量归一化
2. 通过 ResNet-18 + 行锚点分类头输出每条车道线在预定义行上的水平位置
3. 将预测坐标映射回原图尺寸，连线绘制为掩膜
