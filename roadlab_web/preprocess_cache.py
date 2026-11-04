"""Content-addressed CPU lane predictions; mask width/expansion stay uncached."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


def image_digest(image):
    h=hashlib.sha256()
    h.update(str(image.size).encode());h.update(image.convert('RGB').tobytes())
    return h.hexdigest()


def file_digest(paths):
    h=hashlib.sha256()
    for path in sorted(paths,key=str):
        h.update(path.name.encode());h.update(path.read_bytes())
    return h.hexdigest()


def _detect(image,lane_dir):
    import cv2
    import torch
    sys.path.insert(0,str(lane_dir))
    import detect_lanes_clrnet as module
    torch.set_num_threads(4)
    net,cfg,info=module.RT.load_detector(module.DEFAULT_CFG,module.DEFAULT_MODEL,'cpu')
    if info.get('unexpected'):raise RuntimeError('Unexpected lane checkpoint keys')
    lanes,_=module.detect(net,cfg,cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR),'cpu')
    return [np.asarray(lane).tolist() for lane in lanes]


def lane_points(image,lane_dir,cache_dir,enabled=True):
    started=time.perf_counter();lane_dir=Path(lane_dir)
    paths=[lane_dir/'detect_lanes_clrnet.py',lane_dir/'detect_lanes_v2.py',
           lane_dir/'_vendor/clrnet_runtime.py',lane_dir/'_vendor/weights/culane_r18.pth',
           lane_dir/'_vendor/CLRNet/configs/clrnet/clr_resnet18_culane.py']
    paths += list((lane_dir/'_vendor/CLRNet/clrnet').rglob('*.py'))
    paths += list((lane_dir/'_vendor/shim').rglob('*.py'))
    signature=file_digest(paths)
    key=hashlib.sha256(('lane-points-v1:'+image_digest(image)+':'+signature).encode()).hexdigest()
    target=Path(cache_dir)/'lanes'/(key+'.json')
    hit=enabled and target.is_file()
    if hit:
        points=json.loads(target.read_text(encoding='utf-8'))['points']
    else:
        points=_detect(image,lane_dir)
        if enabled:
            target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_suffix('.json.tmp')
            temporary.write_text(json.dumps({'points':points,'backend_sha256':signature}),encoding='utf-8')
            temporary.replace(target)
    return [np.asarray(p,dtype=float) for p in points],{
        'hit':bool(hit),'key':key,'backend_sha256':signature,
        'seconds':round(time.perf_counter()-started,4)}


def render_lanes(lanes,height,width,line_width):
    """Same binary supersampling contract as detect_lanes_clrnet.draw_mask."""
    import cv2
    mask=np.zeros((height,width),dtype='uint8');big=np.zeros((height*2,width*2),dtype='uint8')
    for points in lanes:
        if len(points)<2:continue
        poly=(np.asarray(points,dtype='float64')*2).astype('int32')
        cv2.polylines(big,[poly.reshape(-1,1,2)],False,255,thickness=max(1,line_width*2),lineType=cv2.LINE_AA)
    big=cv2.resize(big,(width,height),interpolation=cv2.INTER_AREA);mask[big>127]=255
    return mask
