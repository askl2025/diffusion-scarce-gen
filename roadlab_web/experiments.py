"""Reproducible run manifests and explicitly reviewed COCO bounding boxes."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from object_profiles import PROFILES, profile


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest(folder, job, graph):
    folder = Path(folder)
    p = profile(job['settings'])
    record = {'schema_version': 1, 'job_id': job['id'], 'synthetic': True,
              'object_type': job['settings'].get('object_type', 'traffic_cone'),
              'category_id': p['category_id'], 'category_scheme': 'project_CODA',
              'settings': job['settings'], 'original_size': job['original_size'],
              'processed_size': job['size'],
              'placement': json.loads((folder/'mask_meta.json').read_text()),
              'models': {k: v['inputs'] for k, v in graph.items()
                         if v['class_type'] in ('UNETLoader', 'DualCLIPLoader', 'VAELoader', 'LoraLoader')},
              'sampler': graph['9']['inputs'],
              'engine': p.get('engine', 'crop_and_stitch'),
              'conditioning': {k: v['inputs'] for k, v in graph.items() if v['class_type'] == 'CLIPTextEncodeFlux'},
              'preprocessing':job.get('preprocessing',{}),
              'scene_analysis':json.loads((folder/'scene.json').read_text()) if (folder/'scene.json').is_file() else None,
              'code_sha256': {name: digest(Path(__file__).parent/name)
                              for name in ('app.py', 'mask_worker.py', 'placement.py', 'object_profiles.py', 'experiments.py', 'workflows.py','preprocess_cache.py','scene_analysis.py','scene_service.py')},
              'input_sha256': digest(folder/'input.png'),
              'original_sha256': digest(folder/'original.png'),
              'mask_sha256': digest(folder/'mask.png'),
              'annotation_note': 'Edit regions and planned boxes are NOT object ground truth. Export requires reviewed boxes.'}
    record['generation_input_sha256'] = {name: digest(folder/name)
                                         for name in ('generation-input.png', 'generation-mask.png', 'generation-blend.png')
                                         if (folder/name).is_file()}
    (folder/'experiment.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')


def finish_manifest(folder, job):
    folder = Path(folder)
    record = json.loads((folder/'experiment.json').read_text())
    record.update(result_sha256=digest(folder/'result.png'), elapsed=job['elapsed'])
    record['timings']=job.get('timings',{})
    record['protection_checks']=job.get('protection_checks')
    if (folder/'result-generated.png').is_file():record['raw_generated_sha256']=digest(folder/'result-generated.png')
    with Image.open(folder/'input.png') as original, Image.open(folder/'result.png') as result, Image.open(folder/'mask.png') as mask:
        original, result = original.convert('RGB'), result.convert('RGB')
        # StarNodes grows/feathers its mask; inspect outside a 32px allowance.
        allowed = mask.getchannel('G').filter(ImageFilter.MaxFilter(65))
        if original.size != result.size:
            raise ValueError('生成结果尺寸与处理图不一致')
        outside = np.asarray(allowed) == 0
        delta = np.abs(np.asarray(original).astype('int16')-np.asarray(result).astype('int16')).max(axis=2)
        record['background_check'] = {
            'allowance_pixels': 32, 'outside_pixel_count': int(outside.sum()),
            'changed_fraction_over_2_levels': float((delta[outside] > 2).mean()) if outside.any() else None,
            'note': 'A pixel consistency check, not an assessment of realism or object correctness.'}
    (folder/'experiment.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')


def coco(jobs, data):
    images, annotations = [], []
    for job in sorted(jobs, key=lambda j: j['id']):
        review = job.get('review', {})
        if job['status'] != 'done' or review.get('verdict') != 'accepted' or not review.get('boxes') or review.get('reviewer') != 'project_user':
            continue
        width, height = job['size']
        image_id = len(images)+1
        images.append({'id': image_id, 'file_name': job['id']+'/result.png',
                       'width': width, 'height': height, 'synthetic': True,
                       'source_job_id': job['id'],
                       'source_sha256': digest(Path(data)/job['id']/'original.png')})
        for box in review['boxes']:
            x, y, w, h = box
            pixels = [round(x*width, 3), round(y*height, 3), round(w*width, 3), round(h*height, 3)]
            annotations.append({'id': len(annotations)+1, 'image_id': image_id,
                                'category_id': profile(job['settings'])['category_id'],
                                'bbox': pixels, 'area': round(pixels[2]*pixels[3], 3), 'iscrowd': 0,
                                'reviewer': review.get('reviewer', 'project_user')})
    return {'info': {'description': 'RoadLab reviewed synthetic targets only; original scene objects are not annotated here.',
                     'split_policy': 'Group by source_sha256 before any train/test split.'},
            'images': images, 'annotations': annotations,
            'categories': [{'id': p['category_id'], 'name': name} for name, p in PROFILES.items()]}
