"""Portable filesystem defaults; environment variables override each location."""
import os
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]


def location(name, default):
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


HOME = location('ROADLAB_HOME', REPOSITORY / '.roadlab')
DATA = location('ROADLAB_DATA', HOME / 'data')
CACHE = location('ROADLAB_CACHE', HOME / 'cache')
COMFY_ROOT = location('COMFY_ROOT', REPOSITORY / 'ComfyUI')
COMFY_DATA = location('COMFY_DATA', COMFY_ROOT)
LANE_DIR = location('LANE_DIR', REPOSITORY / 'scripts' / 'LanDetect')
SCENE_MODEL = location('ROADLAB_SCENE_MODEL', REPOSITORY / 'models' / 'segformer-b0-cityscapes')
COMFY_URL = os.environ.get('COMFY_URL', 'http://127.0.0.1:8188').rstrip('/')
