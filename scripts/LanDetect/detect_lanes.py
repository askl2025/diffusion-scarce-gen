"""
车道线检测与掩膜生成管线。
检测道路图片中的车道线并生成二值掩膜。

基于: Ultra-Fast-Lane-Detection (ECCV 2020)
https://github.com/cfzd/Ultra-Fast-Lane-Detection
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# 配置常量
# ---------------------------------------------------------------------------

CULANE_ROW_ANCHOR = np.array([
    121, 131, 141, 150, 160, 170, 180, 189, 199, 209,
    219, 228, 238, 248, 258, 267, 277, 287
], dtype=np.float32)

GRIDING_NUM = 200             # 分类网格列数
CLS_NUM_PER_LANE = 18         # 行锚点数量
NUM_LANES = 4

RESIZE_H = 288
RESIZE_W = 800

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

# ---------------------------------------------------------------------------
# 模型架构（复现自 Ultra-Fast-Lane-Detection）
# ---------------------------------------------------------------------------


class ConvBNReLU(nn.Module):
    """Conv2d → BatchNorm → ReLU 模块。"""

    def __init__(self, in_channels, out_channels, kernel_size=1,
                 stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding,
                              dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResNetBackbone(nn.Module):
    """ResNet 骨干网络，返回多尺度特征。"""

    def __init__(self, layers='18', pretrained=False):
        super().__init__()
        if layers == '18':
            model = models.resnet18(pretrained=pretrained)
            self.out_channels = 512
        elif layers == '34':
            model = models.resnet34(pretrained=pretrained)
            self.out_channels = 512
        elif layers == '50':
            model = models.resnet50(pretrained=pretrained)
            self.out_channels = 2048
        elif layers == '101':
            model = models.resnet101(pretrained=pretrained)
            self.out_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone: {layers}")

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = self.layer2(x)   # 1/8
        x3 = self.layer3(x2)  # 1/16
        fea = self.layer4(x3)  # 1/32
        return x2, x3, fea


class ParsingNet(nn.Module):
    """Ultra-Fast-Lane-Detection 解析网络。"""

    def __init__(self, size=(288, 800), pretrained=False, backbone='18',
                 cls_dim=(201, 18, 4), use_aux=False):
        super().__init__()
        self.size = size
        self.cls_dim = cls_dim
        self.use_aux = use_aux

        self.model = ResNetBackbone(layers=backbone, pretrained=pretrained)

        # 1x1 卷积将通道数降至 8
        self.pool = nn.Conv2d(self.model.out_channels, 8, kernel_size=1)

        # 经过 5 次下采样后的空间尺寸: H/32 × W/32
        h = size[0] // 32  # 288/32 = 9
        w = size[1] // 32  # 800/32 = 25
        flat_dim = 8 * h * w  # 1800

        total_dim = int(np.prod(cls_dim))

        self.cls = nn.Sequential(
            nn.Linear(flat_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, total_dim),
        )

        if use_aux:
            # 辅助分割头（CULane 推理时不使用）
            self.aux_header2 = ConvBNReLU(128, 128, kernel_size=3,
                                          stride=1, padding=1)
            self.aux_header3 = ConvBNReLU(256, 128, kernel_size=3,
                                          stride=1, padding=1)
            self.aux_header4 = ConvBNReLU(self.model.out_channels, 128,
                                          kernel_size=3, stride=1, padding=1)
            self.aux_combine = nn.Sequential(
                ConvBNReLU(384, 256, kernel_size=3, padding=1),
                nn.Conv2d(256, cls_dim[-1] + 1, kernel_size=1),
            )

    def forward(self, x):
        x2, x3, fea = self.model(x)
        fea_pool = self.pool(fea)                       # (N, 8, 9, 25)
        fea_flat = fea_pool.reshape(fea_pool.size(0), -1)  # (N, 1800)
        group_cls = self.cls(fea_flat).reshape(
            -1, *self.cls_dim)                           # (N, 201, 18, 4)
        return group_cls


def _init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


# ---------------------------------------------------------------------------
# 后处理
# ---------------------------------------------------------------------------

def _softmax_np(x, axis=0):
    """NumPy 版 softmax（无需 scipy）。"""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def compute_lane_points(output, orig_w, orig_h):
    """
    将模型输出解码为原始图像空间中的车道线点坐标。

    参数:
        output: 模型输出张量，形状 (1, 201, 18, 4)
        orig_w, orig_h: 原始图像的宽和高

    返回:
        包含 4 条车道线的列表，每条车道线为 (x, y) 点坐标列表（原始像素坐标）。
    """
    out_j = output[0].data.cpu().numpy()           # (201, 18, 4)
    out_j = out_j[:, ::-1, :]                       # 反转行锚点维度
    # 此时 axis=1 索引 0 → 底部锚点 (287)，索引 17 → 顶部锚点 (121)

    prob = _softmax_np(out_j[:-1, :, :], axis=0)    # 对 200 个网格单元做 softmax
    idx = np.arange(1, GRIDING_NUM + 1).reshape(-1, 1, 1)
    loc = np.sum(prob * idx, axis=0)                 # (18, 4) 期望网格位置

    # 抑制"无线"预测
    out_j_argmax = np.argmax(out_j, axis=0)          # (18, 4)
    loc[out_j_argmax == GRIDING_NUM] = 0              # GRIDING_NUM=200 = 无线类别

    col_sample = np.linspace(0, RESIZE_W - 1, GRIDING_NUM)
    col_sample_w = col_sample[1] - col_sample[0]

    lanes = [[], [], [], []]

    for k in range(CLS_NUM_PER_LANE):
        if loc[k, :].sum() == 0:
            continue
        for i in range(NUM_LANES):
            if loc[k, i] <= 0:
                continue
            x = int(loc[k, i] * col_sample_w * orig_w / RESIZE_W - 1)
            y = int(orig_h * (CULANE_ROW_ANCHOR[CLS_NUM_PER_LANE - 1 - k] / RESIZE_H) - 1)
            x = max(0, x)
            y = max(0, min(y, orig_h - 1))
            lanes[i].append([x, y])

    return lanes


def generate_mask(lanes_points, h, w, lane_width=25):
    """
    在二值掩膜上绘制车道折线。

    参数:
        lanes_points: 包含 4 条车道线的列表，每条车道线为 (x, y) 点坐标列表
        h, w: 掩膜高度和宽度（与原图一致）
        lane_width: 线条粗细（像素）

    返回:
        np.uint8 数组，形状 (h, w)，取值 0 或 255
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    for pts in lanes_points:
        pts = [p for p in pts if p[0] > 0]  # 过滤无效点
        if len(pts) >= 2:
            pts_array = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(mask, [pts_array], isClosed=False,
                          color=255, thickness=lane_width)
    return mask


# ---------------------------------------------------------------------------
# 模型下载与加载
# ---------------------------------------------------------------------------

GDRIVE_FILE_ID = '1zXBRTw50WOzvUp6XKsi8Zrk3MUC3uFuq'


def download_model(model_path):
    """尝试通过 gdown 下载预训练模型。"""
    print(f"未找到模型: {model_path}")
    print("尝试从 Google Drive 下载...")
    try:
        import gdown
        gdown.download(id=GDRIVE_FILE_ID, output=str(model_path), quiet=False)
        if model_path.exists():
            print("下载完成。")
            return True
        print("下载失败 — 文件未保存。")
    except ImportError:
        print("'gdown' 未安装。请使用: pip install gdown")
        print(f"或手动从以下地址下载: "
              f"https://drive.google.com/file/d/{GDRIVE_FILE_ID}/view")
        print(f"并放置到: {model_path}")
    except Exception as e:
        print(f"下载错误: {e}")
    return False


def load_model(model_path, device):
    """构建 ParsingNet 并加载预训练权重。"""
    cls_dim = (GRIDING_NUM + 1, CLS_NUM_PER_LANE, NUM_LANES)  # (201, 18, 4)
    model = ParsingNet(size=(RESIZE_H, RESIZE_W), pretrained=False,
                       backbone='18', cls_dim=cls_dim, use_aux=False)
    model.apply(_init_weights)

    ckpt = torch.load(str(model_path), map_location='cpu', weights_only=True)

    # 兼容不同检查点格式
    if 'model' in ckpt:
        state_dict = ckpt['model']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    # 去除 DataParallel 的 "module." 前缀
    cleaned = {}
    for k, v in state_dict.items():
        new_k = k[7:] if k.startswith('module.') else k
        cleaned[new_k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"提示: {len(missing)} 个缺失键（辅助头或非关键项）。")
    if unexpected:
        print(f"提示: {len(unexpected)} 个多余键（已忽略）。")

    model.to(device)
    model.eval()
    print(f"模型已加载到 {device}。")
    return model


# ---------------------------------------------------------------------------
# 图像处理
# ---------------------------------------------------------------------------

def process_images(input_dir, output_dir, model, device, lane_width):
    """处理 input_dir 中所有图片，将掩膜写入 output_dir。"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((RESIZE_H, RESIZE_W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    image_files = sorted(
        f for f in input_dir.iterdir()
        if f.suffix.lower() in IMG_EXTENSIONS
    )

    if not image_files:
        print(f"在 {input_dir} 中未找到图片文件")
        return

    print(f"找到 {len(image_files)} 张图片待处理。\n")

    for img_path in image_files:
        print(f"处理中: {img_path.name} ...", end=" ", flush=True)

        img = cv2.imread(str(img_path))
        if img is None:
            print("跳过（无法读取）")
            continue

        orig_h, orig_w = img.shape[:2]

        # BGR → RGB → PIL → 张量
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        tensor = transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)

        lanes = compute_lane_points(output, orig_w, orig_h)
        mask = generate_mask(lanes, orig_h, orig_w, lane_width=lane_width)

        # 评估输出掩膜：白色像素占比
        white_ratio = np.count_nonzero(mask) / mask.size

        out_name = f"{img_path.stem}_mask.png"
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), mask)

        if white_ratio < 0.01:
            print(f"警告：检测到的车道线像素仅占 {white_ratio:.2%}，可能未识别到车道 → {out_name}")
        else:
            print(f"完成 → {out_name}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="检测道路图片中的车道线并生成二值掩膜。")
    parser.add_argument('--input-dir', default='./input',
                        help='输入图片目录')
    parser.add_argument('--output-dir', default='./output',
                        help='输出掩膜 PNG 目录')
    parser.add_argument('--model-path', default='./culane_18.pth',
                        help='预训练模型 (.pth) 文件路径')
    parser.add_argument('--lane-width', type=int, default=25,
                        help='掩膜中车道线宽度（像素）')
    parser.add_argument('--no-cuda', action='store_true',
                        help='强制使用 CPU，即使 CUDA 可用')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available()
                          and not args.no_cuda else 'cpu')
    print(f"设备: {device}")

    model_path = Path(args.model_path)
    if not model_path.exists():
        if not download_model(model_path):
            sys.exit(1)

    model = load_model(model_path, device)
    process_images(args.input_dir, args.output_dir, model, device,
                   lane_width=args.lane_width)
    print("\n完成。")


if __name__ == '__main__':
    main()
