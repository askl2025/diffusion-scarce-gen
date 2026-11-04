"""Prepare a joint or sequential native Fill graph; never submits GPU work."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys

from PIL import Image, ImageDraw, ImageOps

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'roadlab_web'))
from placement import native_inputs
from workflows import native_graph


def prepare(image_path, output, boxes, prompt, strategy='joint', seed=42,
            steps=30, guidance=30, case_id='case', comfy_input=None):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', case_id):
        raise ValueError('case-id must contain only letters, numbers, _ or -')
    if strategy not in ('joint', 'sequential') or not 1 <= steps <= 50:
        raise ValueError('Invalid strategy or steps')
    if not prompt.strip() or not boxes or len(boxes) > 8:
        raise ValueError('Provide a prompt and 1-8 normalized rectangles')
    for box in boxes:
        if len(box) != 4 or not all(math.isfinite(v) for v in box):
            raise ValueError('Each box is finite [x, y, width, height]')
        x, y, width, height = box
        if not (0 <= x < 1 and 0 <= y < 1 and width > 0 and height > 0
                and x + width <= 1 and y + height <= 1):
            raise ValueError('Rectangle is outside the image')
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
    if strategy == 'sequential' and (image.width % 16 or image.height % 16):
        raise ValueError('Sequential inputs must have width and height divisible by 16')
    pixel_boxes = [[round(x * image.width), round(y * image.height),
                    round((x + w) * image.width), round((y + h) * image.height)]
                   for x, y, w, h in boxes]
    if any(a >= c or b >= d for a, b, c, d in pixel_boxes):
        raise ValueError('Rectangle has no area after pixel rounding')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    image.save(output / 'input.png')
    mask = Image.new('RGB', image.size)
    draw = ImageDraw.Draw(mask)
    for box in pixel_boxes:
        draw.rounded_rectangle(box, radius=3, fill=(0, 255, 0))
    mask.save(output / 'mask.png')
    padded = native_inputs(image, mask, output)
    prefix = 'roadlab-experiments/' + case_id
    base = native_graph({'id': case_id, 'settings': {
        'prompt': prompt, 'seed': seed, 'steps': steps}})
    for node, name in [('4', 'generation-input.png'), ('5', 'generation-mask.png'),
                       ('25', 'generation-blend.png'), ('28', 'input.png')]:
        base[node]['inputs']['image'] = prefix + '/' + name
    base['21']['inputs']['guidance'] = guidance
    base['12']['inputs']['filename_prefix'] = prefix + '/roi'
    graph = base
    if strategy == 'sequential':
        graph = {key: copy.deepcopy(base[key]) for key in ['1', '2', '3', '4', '21', '22', '28']}
        current_pixels, destination = ['4', 0], ['28', 0]
        for index, box in enumerate(pixel_boxes):
            folder = output / 'slots' / str(index + 1)
            folder.mkdir(parents=True)
            slot_mask = Image.new('RGB', image.size)
            ImageDraw.Draw(slot_mask).rounded_rectangle(box, radius=3, fill=(0, 255, 0))
            native_inputs(image, slot_mask, folder)
            ids = {key: f'slot{index + 1}_{key}' for key in ['5', '6', '23', '9', '24', '25', '26', '27', '12']}
            for key in ids:
                node = copy.deepcopy(base[key])
                for field, value in list(node['inputs'].items()):
                    if isinstance(value, list) and len(value) == 2 and value[0] in ids:
                        node['inputs'][field] = [ids[value[0]], value[1]]
                graph[ids[key]] = node
            slot_prefix = prefix + '/slots/' + str(index + 1)
            graph[ids['5']]['inputs']['image'] = slot_prefix + '/generation-mask.png'
            graph[ids['25']]['inputs']['image'] = slot_prefix + '/generation-blend.png'
            graph[ids['23']]['inputs']['pixels'] = current_pixels
            graph[ids['9']]['inputs']['seed'] = seed + index
            graph[ids['27']]['inputs']['destination'] = destination
            graph[ids['12']]['inputs']['filename_prefix'] = prefix + '/slot' + str(index + 1)
            current_pixels = [ids['27'], 0]
            destination = current_pixels
        graph['12'] = {'class_type': 'SaveImage', 'inputs': {
            'images': current_pixels, 'filename_prefix': prefix + '/roi'}}
    metadata = {
        'strategy': strategy, 'normalized_boxes': boxes, 'pixel_boxes': pixel_boxes,
        'image_size': list(image.size), 'padded_size': padded,
        'seed': seed, 'steps': steps, 'guidance': guidance, 'prompt': prompt,
        'sampler_passes': len(boxes) if strategy == 'sequential' else 1,
        'stage_seeds': [seed + n for n in range(len(boxes))] if strategy == 'sequential' else [seed],
        'source_sha256': hashlib.sha256(Path(image_path).read_bytes()).hexdigest(),
        'training_ground_truth': False, 'review_status': 'unreviewed',
        'comfy_prefix': prefix, 'submitted': False}
    for name, value in [('prompt.json', graph), ('case.json', metadata)]:
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if comfy_input:
        target = Path(comfy_input) / prefix
        shutil.copytree(output, target)  # Existing targets fail rather than overwrite.
    return graph, metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--boxes', type=json.loads, required=True,
                        help='JSON list of normalized [x,y,width,height] rectangles')
    parser.add_argument('--prompt', required=True, help='Use a single-object prompt for sequential mode')
    parser.add_argument('--strategy', choices=['joint', 'sequential'], default='joint')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--guidance', type=float, default=30)
    parser.add_argument('--case-id', default='case')
    parser.add_argument('--comfy-input', type=Path)
    args = parser.parse_args()
    graph, metadata = prepare(args.image, args.output, args.boxes, args.prompt,
                              args.strategy, args.seed, args.steps, args.guidance,
                              args.case_id, args.comfy_input)
    print(json.dumps({'output': str(args.output), 'sampler_passes': metadata['sampler_passes'], 'submitted': False}))
