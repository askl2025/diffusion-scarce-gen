"""Native full-frame FLUX Fill for the experimentally validated new categories."""


def native_graph(job):
    cfg = job['settings']
    prefix = 'roadlab/' + job['id']
    return {
        '1': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'flux1-fill-dev.safetensors', 'weight_dtype': 'default'}},
        '2': {'class_type': 'DualCLIPLoader', 'inputs': {'clip_name1': 'clip_l.safetensors', 'clip_name2': 't5xxl_fp8_e4m3fn_scaled.safetensors', 'type': 'flux', 'device': 'default'}},
        '3': {'class_type': 'VAELoader', 'inputs': {'vae_name': 'ae.safetensors'}},
        '4': {'class_type': 'LoadImage', 'inputs': {'image': prefix + '/generation-input.png'}},
        '5': {'class_type': 'LoadImage', 'inputs': {'image': prefix + '/generation-mask.png'}},
        '6': {'class_type': 'ImageToMask', 'inputs': {'image': ['5', 0], 'channel': 'green'}},
        '21': {'class_type': 'CLIPTextEncodeFlux', 'inputs': {'clip': ['2', 0], 'clip_l': cfg['prompt'], 't5xxl': cfg['prompt'], 'guidance': 30.0}},
        '22': {'class_type': 'ConditioningZeroOut', 'inputs': {'conditioning': ['21', 0]}},
        '23': {'class_type': 'InpaintModelConditioning', 'inputs': {'positive': ['21', 0], 'negative': ['22', 0], 'vae': ['3', 0], 'pixels': ['4', 0], 'mask': ['6', 0], 'noise_mask': True}},
        '9': {'class_type': 'KSampler', 'inputs': {'model': ['1', 0], 'seed': cfg['seed'], 'steps': cfg['steps'], 'cfg': 1.0, 'sampler_name': 'euler', 'scheduler': 'beta', 'positive': ['23', 0], 'negative': ['23', 1], 'latent_image': ['23', 2], 'denoise': 1.0}},
        '24': {'class_type': 'VAEDecode', 'inputs': {'samples': ['9', 0], 'vae': ['3', 0]}},
        '25': {'class_type': 'LoadImage', 'inputs': {'image': prefix + '/generation-blend.png'}},
        '26': {'class_type': 'ImageToMask', 'inputs': {'image': ['25', 0], 'channel': 'green'}},
        '28': {'class_type': 'LoadImage', 'inputs': {'image': prefix + '/input.png'}},
        '27': {'class_type': 'ImageCompositeMasked', 'inputs': {'destination': ['28', 0], 'source': ['24', 0], 'x': 0, 'y': 0, 'resize_source': False, 'mask': ['26', 0]}},
        '12': {'class_type': 'SaveImage', 'inputs': {'images': ['27', 0], 'filename_prefix': prefix}},
    }
