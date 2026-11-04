"""
Run the vendored, unmodified CLRNet under Python 3.8 + torch 2.0.1 + cu117,
with no mmcv install and no CUDA compilation.

Why this exists
---------------
CLRNet's official code needs `mmcv`, `addict`, `yapf` and six, plus a compiled
CUDA NMS op (`clrnet.ops.nms_impl`). None of those are installable here
without a compiler toolchain, and requirements.txt pins torch 1.8 / mmcv 1.2.5.

Rather than re-implement the network (and risk drifting from the checkpoint's
key names), this module leaves `_vendor/CLRNet` byte-for-byte upstream and
supplies the missing pieces:

  * `_vendor/shim/`  -- minimal mmcv.cnn.ConvModule, addict.Dict, yapf, six
  * `lane_nms()`     -- an exact Python port of csrc/nms_kernel.cu

The NMS port is faithful to the CUDA kernel, including its lane-specific IoU:
two lanes are suppressed when the MEAN absolute x-difference over their
overlapping strips falls below `overlap` pixels. This is not box IoU, so
`torchvision.ops.nms` would NOT be a drop-in replacement here.
"""

import sys
import types
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
SHIM_DIR = _HERE / 'shim'
CLRNET_DIR = _HERE / 'CLRNet'
DEFAULT_CFG = CLRNET_DIR / 'configs' / 'clrnet' / 'clr_resnet18_culane.py'

# ---- constants from clrnet/ops/csrc/nms_kernel.cu -------------------------
N_OFFSETS = 72
N_STRIPS = N_OFFSETS - 1        # 71
PROP_SIZE = 5 + N_OFFSETS       # 77
DATASET_OFFSET = 0

_ready = False


# ---------------------------------------------------------------------------
# exact port of devIoU() from nms_kernel.cu
# ---------------------------------------------------------------------------

def _dev_iou(a, b, threshold):
    """True == 'suppress b', matching the kernel's inverted return value."""
    start_a = int(a[2] * N_STRIPS - DATASET_OFFSET + 0.5)   # C cast truncates
    start_b = int(b[2] * N_STRIPS - DATASET_OFFSET + 0.5)
    start = max(start_a, start_b)

    end_a = int(start_a + a[4] - 1 + 0.5 - (1 if (a[4] - 1) < 0 else 0))
    end_b = int(start_b + b[4] - 1 + 0.5 - (1 if (b[4] - 1) < 0 else 0))
    end = min(min(end_a, end_b), N_OFFSETS - 1)

    if end < start:
        return False
    dist = np.abs(a[5 + start:5 + end + 1] - b[5 + start:5 + end + 1]).sum()
    return bool(dist < threshold * (end - start + 1))


def lane_nms(boxes, scores, overlap, top_k):
    """Greedy NMS in descending score order; returns original indices."""
    b = boxes.detach().cpu().numpy()
    s = scores.detach().cpu().numpy()
    n = len(s)
    order = np.argsort(-s, kind='stable')

    suppressed = np.zeros(n, dtype=bool)
    keep = []
    for i in range(n):
        if suppressed[i]:
            continue
        ii = order[i]
        keep.append(int(ii))
        if len(keep) == top_k:
            break
        for j in range(i + 1, n):
            if suppressed[j]:
                continue
            if _dev_iou(b[ii], b[order[j]], overlap):
                suppressed[j] = True
    return keep


def nms_forward(boxes, scores, thresh, top_k):
    """Signature-compatible with the CUDA op: (keep, num_to_keep, parent)."""
    keep = lane_nms(boxes, scores, float(thresh), int(top_k))
    n = boxes.shape[0]
    keep_arr = torch.zeros(n, dtype=torch.long)
    if keep:
        keep_arr[:len(keep)] = torch.tensor(keep, dtype=torch.long)
    parent = torch.zeros(n, dtype=torch.long)
    return keep_arr, len(keep), parent


# ---------------------------------------------------------------------------
# import plumbing
# ---------------------------------------------------------------------------

def setup():
    """Make `import clrnet` work. Idempotent."""
    global _ready
    if _ready:
        return
    for p in (str(SHIM_DIR), str(CLRNET_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)

    # clrnet/ops/nms.py does `from . import nms_impl`, which does not exist
    # because the CUDA extension was never built. Pre-register it so the
    # upstream file stays untouched.
    if 'clrnet.ops.nms_impl' not in sys.modules:
        mod = types.ModuleType('clrnet.ops.nms_impl')
        mod.nms_forward = nms_forward
        sys.modules['clrnet.ops.nms_impl'] = mod
    _ready = True


# ---------------------------------------------------------------------------
# model construction
# ---------------------------------------------------------------------------

def load_config(cfg_path):
    """
    Config.fromfile replacement.

    Upstream's Config._file2dict() does:

        temp = tempfile.NamedTemporaryFile(dir=..., suffix='.py')   # held open
        shutil.copyfile(filename, osp.join(dir, basename(temp.name)))

    On Linux opening an already-open path is fine; on Windows the first handle
    is exclusive, so the copy raises PermissionError. This loader imports the
    .py directly and never touches a temp file.
    """
    setup()
    import importlib.util

    from clrnet.utils.config import Config

    cfg_path = Path(cfg_path).resolve()
    mod_name = '_clrnet_cfg_' + cfg_path.stem
    spec = importlib.util.spec_from_file_location(mod_name, str(cfg_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
        cfg_dict = {k: v for k, v in vars(module).items()
                    if not k.startswith('__')}
    finally:
        sys.modules.pop(mod_name, None)
    with open(cfg_path, 'r', encoding='utf-8') as fh:
        cfg_text = fh.read()
    return Config(cfg_dict, cfg_text=cfg_text, filename=str(cfg_path))


def load_detector(cfg_path=None, weight_path=None, device='cpu'):
    """Build CLRNet from a config and load the official checkpoint."""
    setup()
    from clrnet.models.registry import build_net

    cfg = load_config(cfg_path or DEFAULT_CFG)
    # The config asks for ImageNet-pretrained ResNet weights, but we always
    # overwrite the backbone from the CULane checkpoint, so skip the download.
    if weight_path:
        cfg.backbone.pretrained = False
    net = build_net(cfg)

    info = {}
    if weight_path:
        ckpt = torch.load(str(weight_path), map_location='cpu')
        state = ckpt.get('net', ckpt.get('state_dict', ckpt))
        state = {k[7:] if k.startswith('module.') else k: v
                 for k, v in state.items()}
        missing, unexpected = net.load_state_dict(state, strict=False)
        info = {'missing': list(missing), 'unexpected': list(unexpected),
                'n_loaded': len(state)}
    net.to(device).eval()
    return net, cfg, info


if __name__ == '__main__':
    setup()
    import clrnet
    from clrnet.ops import nms as clrnet_nms
    print('clrnet imported OK from', clrnet.__file__)
    print('clrnet.ops.nms ->', clrnet_nms)
    net, cfg, info = load_detector(weight_path=None)
    n_param = sum(p.numel() for p in net.parameters())
    print(f'Detector built: {n_param/1e6:.2f} M params')
    print(f'  img_w={cfg.img_w} img_h={cfg.img_h} cut_height={cfg.cut_height}')
    print(f'  ori {cfg.ori_img_w}x{cfg.ori_img_h}  '
          f'conf={cfg.test_parameters.conf_threshold} '
          f'nms={cfg.test_parameters.nms_thres} topk={cfg.max_lanes}')
