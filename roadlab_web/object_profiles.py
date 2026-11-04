"""Supported object types; category IDs follow this project's CODA table."""
PROFILES = {
    'traffic_cone': {
        'label': '交通锥 / 路障', 'category_id': 23,
        'lora': 'FLUX_CodaRoadBarrier_v1.0.safetensors',
        'prompt': ('A row of bright orange traffic cones with white reflective stripes, '
                   'placed close together along the road lane marking, spanning the full length '
                   'of the lane, standing upright on the asphalt, KITTI dataset perspective, '
                   'photorealistic, sharp focus, high detail'),
    },
    'dustbin': {
        'label': '垃圾桶', 'category_id': 29, 'lora': None,
        'prompt': ('A single green wheeled plastic trash bin with a closed lid, standing upright '
                   'on the asphalt road. Full bin visible, two small black wheels at the bottom, '
                   'realistic perspective from a front-facing car camera, natural contact shadow '
                   'on the road, lighting and sharpness consistent with the surrounding street photograph.'),
    },
    'warning_sign': {
        'label': '简单警示牌（！）', 'category_id': 20, 'lora': None,
        'engine': 'native_full_frame', 'single_object': True,
        'placement_defaults': {'x': 0.86, 'y': 0.85, 'width': 0.115, 'height': 0.31},
        'placement_note': '请选施工区或锥桶旁的位置；替换旧牌时，编辑框要覆盖原牌面和支架。默认生成简单感叹号，不依赖文字。',
        'prompt': ('A single portable yellow triangular caution sign on a slim dark metal folding stand, '
                   'standing at this road work zone. The face is a clean yellow triangle with a bold dark border '
                   'and just one large simple black exclamation mark in the centre: one straight vertical stroke '
                   'with one black dot below it. Entire sign and stand visible, realistic thin metal frame, '
                   'stand feet contacting the existing asphalt, matching the scene perspective and ambient lighting. '
                   'A plain warning symbol without lettering.'),
    },
    'pedestrian': {
        'label': '背向行人', 'category_id': 1, 'lora': None,
        'engine': 'native_full_frame', 'single_object': True,
        'placement_defaults': {'x': 0.90, 'y': 0.86, 'width': 0.078, 'height': 0.335},
        'placement_note': '请在人行道上选择落脚点，避开现有行人与车辆；本模式生成单个背向自然行走的人物。',
        'prompt': ('One adult pedestrian viewed from behind, walking away along this sidewalk beside the road. '
                   'Full body visible from head to both shoes, wearing an ordinary grey jacket, dark blue jeans '
                   'and plain sneakers, natural relaxed walking pose with two arms and two legs. Both feet meet '
                   'the paved sidewalk, realistic human proportions and scale, matching the shaded street photograph '
                   'and perspective.'),
    },
}


def profile(settings):
    return PROFILES[settings.get('object_type', 'traffic_cone')]


def native_type(settings):
    return profile(settings).get('engine') == 'native_full_frame'
