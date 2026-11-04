"""Object contact points and edit regions, expressed in processed-image coordinates."""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def manual_placement(raw):
    values = {k: float(raw.get(k, default)) for k, default in
              [('x', 0.60), ('y', 0.82), ('width', 0.085)]}
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError('放置参数必须是有限数值')
    if not (0.05 <= values['x'] <= 0.95 and 0.25 <= values['y'] <= 0.96
            and 0.035 <= values['width'] <= 0.18):
        raise ValueError('放置参数超出范围')
    return values


def object_region(width, height, x, y, body_width, body_height=None, padding=None):
    body_height = body_width * 1.65 if body_height is None else body_height
    body = [round(x-body_width/2), round(y-body_height),
            round(x+body_width/2), round(y)]
    pad = max(4, round(width/128)) if padding is None else padding
    edit = [body[0]-pad, body[1]-pad, body[2]+pad, body[3]+pad]
    if edit[0] < 0 or edit[1] < 0 or edit[2] >= width or y+pad*1.5 >= height:
        raise ValueError('物体区域超出图片，请调整位置或缩小垃圾桶')
    return {'contact_point': [round(x), round(y)], 'body_bbox_xyxy': body,
            'edit_bbox_xyxy': edit, 'body_width': round(body_width), 'padding': pad}


def rectangle_placement(raw):
    if not isinstance(raw, dict) or not all(k in raw for k in ('x', 'y', 'width', 'height')):
        raise ValueError('请先选择编辑区域的位置和宽高')
    values = manual_placement(raw)
    values['height'] = float(raw['height'])
    if not math.isfinite(values['height']) or not .08 <= values['height'] <= .70:
        raise ValueError('编辑区域高度超出范围')
    return values


def rectangle_region(size, placement):
    width, height = size
    return dict(object_region(width, height, placement['x']*width, placement['y']*height,
                              placement['width']*width, placement['height']*height, padding=0),
                mode='manual', source='user_selected_edit_rectangle')


def rectangle_mask(size, region):
    mask = Image.new('RGB', size)
    ImageDraw.Draw(mask).rounded_rectangle(region['edit_bbox_xyxy'], radius=5, fill=(0, 255, 0))
    return mask


def native_inputs(image, mask, folder):
    """Pad only the generation inputs; compositing restores the exact original size."""
    w, h = image.size
    pw, ph = (w+15)//16*16, (h+15)//16*16
    pixels = np.pad(np.asarray(image), ((0, ph-h), (0, pw-w), (0, 0)), mode='edge')
    Image.fromarray(pixels).save(folder/'generation-input.png')
    padded_mask = Image.new('RGB', (pw, ph))
    padded_mask.paste(mask, (0, 0))
    padded_mask.save(folder/'generation-mask.png')
    blend = padded_mask.getchannel('G').filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(3))
    zero = Image.new('L', (pw, ph))
    Image.merge('RGB', (zero, blend, zero)).save(folder/'generation-blend.png')
    return [pw, ph]


def plan_dustbin(lanes, width, height, mode='auto', manual=None):
    if mode == 'manual':
        values = manual_placement(manual or {})
        region = object_region(width, height, values['x']*width,
                               values['y']*height, values['width']*width)
        return dict(region, mode='manual', source='user_confirmed_contact_point')
    if mode != 'auto':
        raise ValueError('未知的放置模式')
    # Select two noncrossing boundaries enclosing the camera centre. This is a
    # geometric suggestion, not road segmentation or obstacle detection.
    for y in [height*0.80, height*0.73, height*0.86]:
        xs = []
        for lane in lanes:
            pts = np.asarray(lane, dtype=float)
            if len(pts) < 2:
                continue
            pts = pts[np.argsort(pts[:, 1])]
            if pts[0, 1] <= y <= pts[-1, 1]:
                xs.append(float(np.interp(y, pts[:, 1], pts[:, 0])))
        xs.sort()
        for left, right in zip(xs, xs[1:]):
            gap = right-left
            if left < width/2 < right and width*.12 <= gap <= width*.9:
                x = left + .62*gap
                body_width = max(width*.065, min(gap*.25, width*.11))
                region = object_region(width, height, x, y, body_width)
                return dict(region, mode='auto', source='lane_geometry',
                            lane_bounds_at_contact=[round(left), round(right)],
                            warning='自动位置仅依据车道线，未检测现有障碍物，请核对结果；可改为手动选位。')
    raise ValueError('没有找到可靠的左右车道边界，请选择手动选位后重试。')


def dustbin_mask(size, region):
    width, height = size
    mask = Image.new('RGB', size)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle(region['edit_bbox_xyxy'], radius=region['padding'], fill=(0, 255, 0))
    x, y = region['contact_point']
    half = region['body_width']*.75
    pad = region['padding']
    draw.ellipse((max(0, x-half), y-pad, min(width-1, x+half),
                  min(height-1, y+pad*1.5)), fill=(0, 255, 0))
    return mask


def annotation_boxes(raw):
    if not isinstance(raw, list) or len(raw) > 100:
        raise ValueError('标注框格式不正确')
    boxes = []
    for item in raw:
        if not isinstance(item, list) or len(item) != 4:
            raise ValueError('标注框应为归一化的 x、y、宽、高')
        box = [float(v) for v in item]
        x, y, w, h = box
        if not all(math.isfinite(v) for v in box) or not (
                x >= 0 and y >= 0 and w > 0 and h > 0 and x+w <= 1.000001 and y+h <= 1.000001):
            raise ValueError('标注框超出图片范围')
        boxes.append([round(v, 7) for v in box])
    return boxes
