"""
Generate lane detection overlay visualizations.
Draws detected lane lines (red) on top of original road images.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

warnings.filterwarnings("ignore", category=FutureWarning)

# Import pipeline components from detect_lanes
from detect_lanes import (
    CULANE_ROW_ANCHOR, GRIDING_NUM, CLS_NUM_PER_LANE, NUM_LANES,
    RESIZE_H, RESIZE_W, IMAGENET_MEAN, IMAGENET_STD, IMG_EXTENSIONS,
    ParsingNet, _init_weights, compute_lane_points, download_model,
)


def load_model(model_path, device):
    cls_dim = (GRIDING_NUM + 1, CLS_NUM_PER_LANE, NUM_LANES)
    model = ParsingNet(size=(RESIZE_H, RESIZE_W), pretrained=False,
                       backbone='18', cls_dim=cls_dim, use_aux=False)
    model.apply(_init_weights)

    ckpt = torch.load(str(model_path), map_location='cpu', weights_only=True)
    if 'model' in ckpt:
        state_dict = ckpt['model']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    cleaned = {k[7:] if k.startswith('module.') else k: v
               for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.eval()
    return model


def generate_cover(img_path, model, device, lane_width=25, alpha=0.45):
    """
    Generate an overlay image with detected lanes highlighted.

    Returns:
        overlay image (BGR ndarray) or None on failure.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    orig_h, orig_w = img.shape[:2]

    # Preprocess
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    transform = transforms.Compose([
        transforms.Resize((RESIZE_H, RESIZE_W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tensor = transform(img_pil).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(tensor)

    lanes = compute_lane_points(output, orig_w, orig_h)

    # Draw lanes on mask
    mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    for pts in lanes:
        pts = [p for p in pts if p[0] > 0]
        if len(pts) >= 2:
            pts_array = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(mask, [pts_array], isClosed=False,
                          color=255, thickness=lane_width)

    # Colorize mask: red lanes
    overlay = img.copy()
    overlay[mask == 255] = (0, 0, 255)  # BGR red

    blended = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
    return blended


def main():
    parser = argparse.ArgumentParser(
        description="Generate lane detection cover/overlay images.")
    parser.add_argument('--input-dir', default='./input',
                        help='Directory containing input images')
    parser.add_argument('--output-dir', default='./cover',
                        help='Directory for cover images')
    parser.add_argument('--model-path', default='./culane_18.pth',
                        help='Path to pretrained model (.pth) file')
    parser.add_argument('--lane-width', type=int, default=25,
                        help='Lane line thickness (pixels)')
    parser.add_argument('--alpha', type=float, default=0.45,
                        help='Overlay opacity (0-1)')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Force CPU')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available()
                          and not args.no_cuda else 'cpu')
    print(f"Device: {device}")

    model_path = Path(args.model_path)
    if not model_path.exists():
        if not download_model(model_path):
            sys.exit(1)

    model = load_model(model_path, device)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        f for f in input_dir.iterdir()
        if f.suffix.lower() in IMG_EXTENSIONS
    )

    if not image_files:
        print(f"No image files found in {input_dir}")
        return

    print(f"Found {len(image_files)} image(s).\n")

    for img_path in image_files:
        print(f"Processing: {img_path.name} ...", end=" ", flush=True)
        result = generate_cover(img_path, model, device,
                                lane_width=args.lane_width,
                                alpha=args.alpha)
        if result is None:
            print("SKIP (unreadable)")
            continue

        out_name = f"{img_path.stem}_cover.jpg"
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"OK → {out_name}")

    print("\nDone.")


if __name__ == '__main__':
    main()
