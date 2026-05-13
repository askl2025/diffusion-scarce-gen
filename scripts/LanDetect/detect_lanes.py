"""
Lane detection mask pipeline.
Detects lane lines in road images and generates binary masks.

Based on: Ultra-Fast-Lane-Detection (ECCV 2020)
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
# Configuration constants
# ---------------------------------------------------------------------------

CULANE_ROW_ANCHOR = np.array([
    121, 131, 141, 150, 160, 170, 180, 189, 199, 209,
    219, 228, 238, 248, 258, 267, 277, 287
], dtype=np.float32)

GRIDING_NUM = 200             # classification grid columns
CLS_NUM_PER_LANE = 18         # number of row anchors
NUM_LANES = 4

RESIZE_H = 288
RESIZE_W = 800

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

# ---------------------------------------------------------------------------
# Model architecture (replicated from Ultra-Fast-Lane-Detection)
# ---------------------------------------------------------------------------


class ConvBNReLU(nn.Module):
    """Conv2d -> BatchNorm -> ReLU block."""

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
    """ResNet backbone that returns multi-scale features."""

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
    """Ultra-Fast-Lane-Detection parsing network."""

    def __init__(self, size=(288, 800), pretrained=False, backbone='18',
                 cls_dim=(201, 18, 4), use_aux=False):
        super().__init__()
        self.size = size
        self.cls_dim = cls_dim
        self.use_aux = use_aux

        self.model = ResNetBackbone(layers=backbone, pretrained=pretrained)

        # 1x1 conv to reduce channels to 8
        self.pool = nn.Conv2d(self.model.out_channels, 8, kernel_size=1)

        # Spatial size after 5 downsamplings: H/32 x W/32
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
            # Auxiliary segmentation head (not used for CULane inference)
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
# Post-processing
# ---------------------------------------------------------------------------

def _softmax_np(x, axis=0):
    """NumPy softmax fallback (doesn't require scipy)."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def compute_lane_points(output, orig_w, orig_h):
    """
    Decode model output into lane point coordinates in original image space.

    Args:
        output: model output tensor of shape (1, 201, 18, 4)
        orig_w, orig_h: original image width and height

    Returns:
        List of 4 lanes, each a list of (x, y) points in original pixels.
    """
    out_j = output[0].data.cpu().numpy()           # (201, 18, 4)
    out_j = out_j[:, ::-1, :]                       # reverse row anchor dim
    # Now axis=1 index 0 → bottom anchor (287), index 17 → top anchor (121)

    prob = _softmax_np(out_j[:-1, :, :], axis=0)    # softmax over 200 grid cells
    idx = np.arange(1, GRIDING_NUM + 1).reshape(-1, 1, 1)
    loc = np.sum(prob * idx, axis=0)                 # (18, 4) expected grid position

    # Suppress "no-lane" predictions
    out_j_argmax = np.argmax(out_j, axis=0)          # (18, 4)
    loc[out_j_argmax == GRIDING_NUM] = 0              # GRIDING_NUM = 200 = no-lane class

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
    Draw lane polylines on a binary mask.

    Args:
        lanes_points: list of 4 lanes, each a list of (x, y) points
        h, w: mask height and width (matches original image)
        lane_width: line thickness in pixels

    Returns:
        np.uint8 array of shape (h, w), values 0 or 255
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    for pts in lanes_points:
        pts = [p for p in pts if p[0] > 0]  # filter invalid
        if len(pts) >= 2:
            pts_array = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(mask, [pts_array], isClosed=False,
                          color=255, thickness=lane_width)
    return mask


# ---------------------------------------------------------------------------
# Model download and loading
# ---------------------------------------------------------------------------

GDRIVE_FILE_ID = '1zXBRTw50WOzvUp6XKsi8Zrk3MUC3uFuq'


def download_model(model_path):
    """Attempt to download pretrained model via gdown."""
    print(f"Model not found at: {model_path}")
    print("Attempting download from Google Drive...")
    try:
        import gdown
        gdown.download(id=GDRIVE_FILE_ID, output=str(model_path), quiet=False)
        if model_path.exists():
            print("Download complete.")
            return True
        print("Download failed — file not saved.")
    except ImportError:
        print("'gdown' is not installed. Install it with:  pip install gdown")
        print(f"Or manually download from: "
              f"https://drive.google.com/file/d/{GDRIVE_FILE_ID}/view")
        print(f"And place it at: {model_path}")
    except Exception as e:
        print(f"Download error: {e}")
    return False


def load_model(model_path, device):
    """Build ParsingNet and load pretrained weights."""
    cls_dim = (GRIDING_NUM + 1, CLS_NUM_PER_LANE, NUM_LANES)  # (201, 18, 4)
    model = ParsingNet(size=(RESIZE_H, RESIZE_W), pretrained=False,
                       backbone='18', cls_dim=cls_dim, use_aux=False)
    model.apply(_init_weights)

    ckpt = torch.load(str(model_path), map_location='cpu', weights_only=True)

    # Handle different checkpoint formats
    if 'model' in ckpt:
        state_dict = ckpt['model']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    # Strip "module." prefix from DataParallel
    cleaned = {}
    for k, v in state_dict.items():
        new_k = k[7:] if k.startswith('module.') else k
        cleaned[new_k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"Info: {len(missing)} missing keys (aux head or non-critical).")
    if unexpected:
        print(f"Info: {len(unexpected)} unexpected keys (ignored).")

    model.to(device)
    model.eval()
    print(f"Model loaded on {device}.")
    return model


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------

def process_images(input_dir, output_dir, model, device, lane_width):
    """Process all images in input_dir and write masks to output_dir."""
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
        print(f"No image files found in {input_dir}")
        return

    print(f"Found {len(image_files)} image(s) to process.\n")

    for img_path in image_files:
        print(f"Processing: {img_path.name} ...", end=" ", flush=True)

        img = cv2.imread(str(img_path))
        if img is None:
            print("SKIP (unreadable)")
            continue

        orig_h, orig_w = img.shape[:2]

        # BGR → RGB → PIL → tensor
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        tensor = transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)

        lanes = compute_lane_points(output, orig_w, orig_h)
        mask = generate_mask(lanes, orig_h, orig_w, lane_width=lane_width)

        out_name = f"{img_path.stem}_mask.png"
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), mask)
        print(f"OK → {out_name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Detect lanes in road images and generate binary masks.")
    parser.add_argument('--input-dir', default='./input',
                        help='Directory containing input images')
    parser.add_argument('--output-dir', default='./output',
                        help='Directory for output mask PNGs')
    parser.add_argument('--model-path', default='./culane_18.pth',
                        help='Path to pretrained model (.pth) file')
    parser.add_argument('--lane-width', type=int, default=25,
                        help='Lane line thickness in mask (pixels)')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Force CPU even if CUDA is available')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available()
                          and not args.no_cuda else 'cpu')
    print(f"Device: {device}")

    model_path = Path(args.model_path)
    if not model_path.exists():
        if not download_model(model_path):
            sys.exit(1)

    model = load_model(model_path, device)
    process_images(args.input_dir, args.output_dir, model, device,
                   lane_width=args.lane_width)
    print("\nDone.")


if __name__ == '__main__':
    main()
