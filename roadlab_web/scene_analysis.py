"""CPU semantic protection of original people/vehicles, with reproducible caching."""
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image,ImageFilter

from preprocess_cache import image_digest,file_digest

from config import SCENE_MODEL

MODEL=SCENE_MODEL
PROTECTED_LABELS=[11,12,13,14,15,16,17,18]


def model_available():
    return (MODEL/'config.json').is_file() and (MODEL/'preprocessor_config.json').is_file() and any((MODEL/n).is_file() for n in ['model.safetensors','pytorch_model.bin'])


def _segment(image,model_dir):
    # Runs only in the isolated preprocessing child, never in ComfyUI's process.
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    from scipy.ndimage import binary_fill_holes,label
    from transformers import SegformerImageProcessor,SegformerForSemanticSegmentation
    torch.set_num_threads(4)
    processor=SegformerImageProcessor.from_pretrained(model_dir,local_files_only=True)
    model=SegformerForSemanticSegmentation.from_pretrained(model_dir,local_files_only=True).eval()
    with torch.inference_mode():
        outputs=model(**processor(images=image,return_tensors='pt'))
        logits=torch.nn.functional.interpolate(outputs.logits,size=(image.height,image.width),mode='bilinear',align_corners=False)
        confidence,classes=logits.softmax(1).max(1)
    classes=classes[0].numpy().astype('uint8');confidence=confidence[0].numpy()
    raw=np.isin(classes,PROTECTED_LABELS)&(confidence>=.60)
    components,n=label(raw,structure=np.ones((3,3)))
    protected=np.zeros(raw.shape,dtype=bool)
    for index in range(1,n+1):
        component=components==index
        if component.sum()<12:continue
        protected |= binary_fill_holes(component)
    return classes,protected.astype('uint8')*255


def analyze(image,cache_dir,model_dir=None):
    started=time.perf_counter();model_dir=Path(model_dir or MODEL)
    files=[model_dir/'config.json',model_dir/'preprocessor_config.json']
    files += [p for p in [model_dir/'model.safetensors',model_dir/'pytorch_model.bin'] if p.is_file()]
    signature=file_digest(files)
    key=hashlib.sha256(('scene-protection-v1:'+image_digest(image)+':'+signature).encode()).hexdigest()
    folder=Path(cache_dir)/'scene'/key
    hit=(folder/'ready.json').is_file() and (folder/'protection.png').is_file() and (folder/'labels.png').is_file()
    if not hit:
        classes,protection=_segment(image,model_dir)
        folder.mkdir(parents=True,exist_ok=True)
        Image.fromarray(classes).save(folder/'labels.png');Image.fromarray(protection).save(folder/'protection.png')
        temporary=folder/'ready.json.tmp';temporary.write_text(json.dumps({'model_sha256':signature,'input_sha256':image_digest(image)}));temporary.replace(folder/'ready.json')
    mask=Image.open(folder/'protection.png').convert('L');labels=Image.open(folder/'labels.png').convert('L')
    meta={'cache_hit':hit,'cache_key':key,'seconds':round(time.perf_counter()-started,4),
          'model':'nvidia/segformer-b0-finetuned-cityscapes-1024-1024','model_sha256':signature,
          'protected_labels':PROTECTED_LABELS,'confidence_threshold':.60,
          'protected_pixels':int((np.asarray(mask)>0).sum()),
          'scope':'Predicted original people and vehicles only; segmentation may miss objects.'}
    return mask,labels,meta


def composite_protected(original,generated,mask):
    if original.size!=generated.size or original.size!=mask.size:raise ValueError('Protection size mismatch')
    core=mask.convert('L').filter(ImageFilter.MaxFilter(9))
    soft=mask.convert('L').filter(ImageFilter.MaxFilter(13)).filter(ImageFilter.GaussianBlur(3))
    weight=np.asarray(soft,dtype='float32')/255
    weight[np.asarray(core)>0]=1
    source=np.asarray(original.convert('RGB'));raw=np.asarray(generated.convert('RGB'))
    output=np.rint(raw*(1-weight[:,:,None])+source*weight[:,:,None]).clip(0,255).astype('uint8')
    selected=np.asarray(mask)>0
    output[selected]=source[selected]
    before=np.abs(raw.astype('int16')-source.astype('int16'))
    after=np.abs(output.astype('int16')-source.astype('int16'))
    metrics={'predicted_protected_pixels':int(selected.sum()),
             'before_mae':float(before[selected].mean()) if selected.any() else 0.0,
             'after_mae':float(after[selected].mean()) if selected.any() else 0.0,
             'protected_pixels_exact':bool(np.array_equal(output[selected],source[selected])),
             'note':'Protection of predicted regions; not proof that every real object was detected.'}
    return Image.fromarray(output),metrics


def write_scene_files(image,folder,cache_dir):
    folder=Path(folder)
    protection,labels,meta=analyze(image,cache_dir)
    protection.save(folder/'protection.png');labels.save(folder/'scene-labels.png')
    selected=np.asarray(protection)>0;overlay=np.asarray(image).copy()
    overlay[selected]=(overlay[selected]*.5+np.array([30,220,245])*.5).astype('uint8')
    Image.fromarray(overlay).save(folder/'protection-overlay.png')
    (folder/'scene.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    return meta
