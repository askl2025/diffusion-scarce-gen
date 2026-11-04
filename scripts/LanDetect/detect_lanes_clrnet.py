"""
CLRNet (CVPR 2022) lane detection - ResNet-18 / CULane checkpoint.

Drop-in sibling of detect_lanes_v2.py: same CLI shape, same output contract
(binary PNG mask, white lanes on black).

Official CULane F1@50 = 79.58 for this backbone, against 68.4 for the UFLD
ResNet-18 model the project started with.

Implementation notes (all verified against the vendored upstream source)
------------------------------------------------------------------------
* Input scaling is `/255` to float in [0, 1] -- see `preprocess()`. Getting
  this wrong does not crash, it silently makes the classifier reject every
  proposal (max lane score 0.016 instead of 0.93 on a native CULane frame).
* The mean-subtraction in every config (`img_norm = dict(mean=[103.939,
  116.779, 123.68], std=[1., 1., 1.])`) is DEAD CONFIG: nothing in the repo
  ever instantiates the `Normalize` process. Subtracting the mean drops the
  score back to 0.03, so do not "fix" it by adding normalisation.
* Resize is `cv2.INTER_CUBIC` to (800, 320), because the config's named
  transforms are dispatched through imgaug (`getattr(iaa, name)(...)`), whose
  Resize defaults to cubic.
* `cut_height` crops the top of the frame before resizing. Upstream uses 270
  of 590 rows for CULane; that number is camera-specific, so it is exposed as
  `--keep` (fraction of rows kept from the bottom) and swept on our data.
* Decoding reuses upstream's own `CLRHead.get_lanes`, driven by temporarily
  setting cfg.ori_img_h / cfg.cut_height so its coordinate mapping lands in
  our image's pixel frame.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / '_vendor'))

import clrnet_runtime as RT                                          # noqa: E402
from detect_lanes_v2 import IMG_EXTENSIONS                           # noqa: E402

DEFAULT_MODEL = _HERE / '_vendor' / 'weights' / 'culane_r18.pth'
DEFAULT_CFG = _HERE / '_vendor' / 'CLRNet' / 'configs' / 'clrnet' / 'clr_resnet18_culane.py'


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def draw_mask(lanes, h, w, lane_width, antialias=True):
    """
    Render dense polylines to a binary mask.

    Same anti-aliased supersampled approach as detect_lanes_v2.draw_mask, but
    operating on raw (N, 2) point arrays: CLRNet already emits ~120 points per
    lane, so no curve fitting is needed.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    if not lanes:
        return mask
    ss = 2 if antialias else 1
    big = np.zeros((h * ss, w * ss), dtype=np.uint8)
    for pts in lanes:
        if len(pts) < 2:
            continue
        poly = (np.asarray(pts, dtype=np.float64) * ss).astype(np.int32)
        cv2.polylines(big, [poly.reshape(-1, 1, 2)], False, 255,
                      thickness=max(1, lane_width * ss), lineType=cv2.LINE_AA)
    if ss > 1:
        big = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)
    mask[big > 127] = 255
    return mask


# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------

def preprocess(img_bgr, keep, img_w=800, img_h=320, cut_height=None):
    """
    Crop the top of the frame, resize to the network input size, scale to [0,1].

    keep = fraction of rows kept from the bottom (matching detect_lanes_v2).
    Upstream's CULane setting (cut 270 of 590) is keep = 0.5424.

    The `/255` mirrors `GenerateLaneLine.__call__` in
    clrnet/datasets/process/generate_lane_line.py:

        sample['img'] = img.astype(np.float32) / 255.

    It runs in BOTH the train and eval branches, and there is no mean/std
    subtraction anywhere on the inference path (see module docstring).

    Returns (tensor, cut_height_px).
    """
    h, w = img_bgr.shape[:2]
    if cut_height is None:
        cut_height = int(round(h * (1.0 - keep)))
    cut_height = max(0, min(cut_height, h - 1))

    crop = img_bgr[cut_height:, :, :]
    resized = cv2.resize(crop, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
    scaled = resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(scaled).permute(2, 0, 1).unsqueeze(0)
    return tensor, cut_height


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def detect(net, cfg, img_bgr, device, keep=0.5424, conf=None, dense=120):
    """
    Returns a list of (N, 2) float arrays of pixel coordinates, plus the
    number of raw proposals that survived the confidence threshold.
    """
    h, w = img_bgr.shape[:2]
    tensor, cut_height = preprocess(img_bgr, keep, cfg.img_w, cfg.img_h)

    preds = net(tensor.to(device))

    # Drive upstream's decode with our image geometry.
    cfg.ori_img_w = w
    cfg.ori_img_h = h
    cfg.cut_height = cut_height
    if conf is not None:
        cfg.test_parameters.conf_threshold = conf

    lanes = net.heads.get_lanes(preds)[0]

    out = []
    for lane in lanes:
        # lane(ys) expects/produces coordinates normalised to [0, 1] of the
        # ORIGINAL image; it returns -2 outside its own valid y span.
        ys = np.linspace(lane.min_y, lane.max_y, dense)
        xs = lane(ys)
        ok = (xs >= 0.0) & (xs < 1.0)
        if ok.sum() < 2:
            continue
        pts = np.stack([xs[ok] * w, ys[ok] * h], axis=1)
        out.append(pts)
    return out, len(lanes)


@torch.no_grad()
def detect_regions(net, cfg, img_bgr, device, keep=0.5424, conf=None, dense=120):
    """
    Same decode as detect(), but also returns the proposal start_y / length so
    callers can reason about which part of the image each lane came from.
    """
    lanes, n = detect(net, cfg, img_bgr, device, keep=keep, conf=conf, dense=dense)
    return lanes, n


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="CLRNet R18 lane masks (CULane checkpoint).")
    ap.add_argument('--input-dir', default='./input')
    ap.add_argument('--output-dir', default='./output')
    ap.add_argument('--model-path', default=str(DEFAULT_MODEL))
    ap.add_argument('--config', default=str(DEFAULT_CFG))
    ap.add_argument('--lane-width', type=int, default=0,
                    help='Mask line width in px; 0 = auto (h/41)')
    ap.add_argument('--keep', type=float, default=0.5424,
                    help='Bottom fraction of rows kept (upstream CULane = 0.5424)')
    ap.add_argument('--conf', type=float, default=None,
                    help='Override confidence threshold (checkpoint uses 0.4)')
    ap.add_argument('--no-cuda', action='store_true')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available()
                          and not args.no_cuda else 'cpu')
    net, cfg, info = RT.load_detector(args.config, args.model_path, device)
    if info.get('unexpected'):
        print(f"WARNING: {len(info['unexpected'])} unexpected checkpoint keys")
    print(f"Device: {device}   CLRNet R18   keep={args.keep}  "
          f"conf={args.conf if args.conf is not None else cfg.test_parameters.conf_threshold}")

    in_dir, out_dir = Path(args.input_dir), Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in in_dir.iterdir() if f.suffix.lower() in IMG_EXTENSIONS)
    if not files:
        print(f"No images in {in_dir}")
        return

    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"{f.name}: unreadable")
            continue
        h, w = img.shape[:2]
        width = args.lane_width or max(4, int(round(h / 41.0)))
        lanes, n_prop = detect(net, cfg, img, device, keep=args.keep,
                               conf=args.conf)
        mask = draw_mask(lanes, h, w, width)
        cv2.imwrite(str(out_dir / f"{f.stem}_mask.png"), mask)
        print(f"{f.name:34s} lanes={len(lanes)} (proposals={n_prop}) width={width}")
    print("\nDone.")


if __name__ == '__main__':
    main()
