"""Isolated CPU mask worker; manual placement does not require a lane model."""
import json
from pathlib import Path
import sys
import os

import numpy as np
from PIL import Image

from object_profiles import native_type
from placement import plan_dustbin, dustbin_mask, rectangle_region, rectangle_mask, native_inputs
from preprocess_cache import lane_points,render_lanes
from scene_analysis import write_scene_files


def main():
    folder, expand, lane_dir = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    cache_dir=Path(sys.argv[4]) if len(sys.argv)>4 else folder.parent/'roadlab-cache'
    cfg = json.loads((folder/'job.json').read_text())['settings']
    object_type = cfg.get('object_type', 'traffic_cone')
    native = native_type(cfg)
    manual = native or (object_type == 'dustbin' and cfg.get('placement_mode', 'manual') == 'manual')
    image = Image.open(folder/'input.png').convert('RGB')
    w, h = image.size
    lanes = []
    lane_cache=None
    if not manual:
        lanes,lane_cache=lane_points(image,lane_dir,cache_dir,cfg.get('use_lane_cache',True))
    (folder/'lanes.json').write_text(json.dumps([np.asarray(l).tolist() for l in lanes]))
    meta = {'lane_count': len(lanes), 'backend': 'manual' if manual else 'CLRNet R18'}
    if lane_cache:meta['lane_cache']=lane_cache
    if cfg.get('protect_original',False) and os.environ.get('ROADLAB_DEFER_SCENE')!='1':
        meta['scene']=write_scene_files(image,folder,cache_dir)
    if native:
        region = rectangle_region(image.size, cfg['placement'])
        green = rectangle_mask(image.size, region)
        meta['placement'] = region
        meta['generation_size'] = native_inputs(image, green, folder)
        meta['engine'] = 'native_full_frame'
    elif object_type == 'dustbin':
        region = plan_dustbin(lanes, w, h, cfg.get('placement_mode', 'manual'), cfg.get('placement'))
        green = dustbin_mask(image.size, region)
        meta['placement'] = region
    else:
        import cv2
        mask = render_lanes(lanes, h, w, max(4, round(h/41)))
        if expand:
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*expand+1, 2*expand+1)))
        pixels = np.zeros((h,w,3),dtype=np.uint8)
        pixels[:,:,1] = mask
        green = Image.fromarray(pixels)
        meta['expand'] = expand
    green.save(folder/'mask.png')
    selected = np.asarray(green)[:,:,1] > 0
    overlay = np.asarray(image).copy()
    overlay[selected] = (overlay[selected]*.5+np.array([35,220,120])*.5).astype(np.uint8)
    Image.fromarray(overlay).save(folder/'overlay.png')
    meta['coverage'] = float(selected.mean())
    (folder/'mask_meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
