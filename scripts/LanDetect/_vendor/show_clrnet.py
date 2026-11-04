"""Visual check: CLRNet vs UFLD v2 on the same images."""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '_vendor'))

import clrnet_runtime as RT
import detect_lanes_clrnet as C
import detect_lanes_v2 as V2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- CLRNet
cnet, ccfg, _ = RT.load_detector(
    ROOT / '_vendor/CLRNet/configs/clrnet/clr_resnet18_culane.py',
    ROOT / '_vendor/weights/culane_r18.pth', device)
# --- UFLD v2
unet = V2.load_model(str(ROOT / 'culane_18.pth'), device)


def panel(img, mask, label):
    out = img.copy()
    if mask is not None:
        colour = np.zeros_like(img)
        colour[mask > 0] = (0, 0, 255)
        out = cv2.addWeighted(img, 0.35, colour, 0.65, 0)
    cv2.rectangle(out, (0, 0), (560, 52), (0, 0, 0), -1)
    cv2.putText(out, label, (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.25,
                (255, 255, 255), 2, cv2.LINE_AA)
    return out


for name in ['sample001', '000038_1616179765900', 'sample003']:
    img = cv2.imread(str(ROOT / 'input' / f'{name}.jpg'))
    h, w = img.shape[:2]

    lanes_c, n_c = C.detect(cnet, ccfg, img, device, keep=0.5424, conf=0.4)
    m_c = C.draw_mask(lanes_c, h, w, 25)

    lanes_u, _ = V2.detect(unet, img, device, conf_thresh=0.3, keep=0.8)
    m_u = V2.draw_mask(lanes_u, h, w, 25)

    grid = np.vstack([
        np.hstack([panel(img, None, f'{name}  input'),
                   panel(img, m_u, f'UFLD v2  ({len(lanes_u)} lanes)')]),
        np.hstack([panel(img, m_c, f'CLRNet R18  ({len(lanes_c)} lanes, {n_c} prop)'),
                   panel(img, m_c, 'CLRNet (repeat)')]),
    ])
    grid = cv2.resize(grid, (1800, 956), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(ROOT / '_vendor' / f'vis_{name}.jpg'), grid,
                [cv2.IMWRITE_JPEG_QUALITY, 93])
    print('wrote', f'_vendor/vis_{name}.jpg', f'CLRNet={len(lanes_c)} UFLD={len(lanes_u)}')
