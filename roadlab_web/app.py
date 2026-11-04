"""RoadLab: upload -> CLRNet mask -> ComfyUI FLUX Fill. Runs in ComfyUI's venv."""
import asyncio
import contextlib
import io
import json
import logging
import os
from pathlib import Path
import secrets
import shutil
import sys
import time
import zipfile

from aiohttp import web, ClientSession, ClientTimeout, WSMsgType
from PIL import Image, ImageOps, UnidentifiedImageError

from object_profiles import PROFILES, profile, native_type
from placement import manual_placement, object_region, annotation_boxes, rectangle_placement, rectangle_region
from experiments import manifest, finish_manifest, coco
from workflows import native_graph
from scene_analysis import model_available,composite_protected

ROOT = Path(__file__).resolve().parent
from config import DATA, CACHE, COMFY_DATA, COMFY_URL
from config import LANE_DIR as CONFIGURED_LANE_DIR

COMFY = COMFY_URL
LANE_DIR = str(CONFIGURED_LANE_DIR)
DEFAULT_PROMPT = PROFILES['traffic_cone']['prompt']
TERMINAL = {'done', 'error', 'cancelled'}
Image.MAX_IMAGE_PIXELS = 24_000_000
log = logging.getLogger('roadlab')


def save(job):
    job['updated_at'] = time.time()
    path = DATA / job['id'] / 'job.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def update(job, **values):
    job.update(values)
    save(job)


def public(job):
    result = dict(job)
    result['object_label'] = profile(job['settings'])['label']
    result['files'] = {name: f"api/jobs/{job['id']}/files/{name}.png"
                       for name in ('original', 'input', 'mask', 'overlay', 'result','protection-overlay')
                       if (DATA / job['id'] / f'{name}.png').is_file()}
    result['bundle_url'] = f"api/jobs/{job['id']}/bundle"
    result['experiment_url'] = f"api/jobs/{job['id']}/experiment"
    return result


def graph(job):
    cfg = job['settings']
    if native_type(cfg):
        return native_graph(job)
    prefix = f"roadlab/{job['id']}"
    graph = {
        '1': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'flux1-fill-dev.safetensors', 'weight_dtype': 'default'}},
        '2': {'class_type': 'DualCLIPLoader', 'inputs': {'clip_name1': 'clip_l.safetensors', 'clip_name2': 't5xxl_fp8_e4m3fn_scaled.safetensors', 'type': 'flux', 'device': 'default'}},
        '3': {'class_type': 'VAELoader', 'inputs': {'vae_name': 'ae.safetensors'}},
        '4': {'class_type': 'LoadImage', 'inputs': {'image': prefix + '/input.png'}},
        '5': {'class_type': 'LoadImage', 'inputs': {'image': prefix + '/mask.png'}},
        '6': {'class_type': 'ImageToMask', 'inputs': {'image': ['5', 0], 'channel': 'green'}},
        '13': {'class_type': 'LoraLoader', 'inputs': {'model': ['1', 0], 'clip': ['2', 0], 'lora_name': 'FLUX_CodaRoadBarrier_v1.0.safetensors', 'strength_model': 1.0, 'strength_clip': 1.0}},
        '9': {'class_type': 'FluxFillSampler', 'inputs': {
            'model': ['13', 0], 'clip': ['2', 0], 'vae': ['3', 0], 'image': ['4', 0],
            'mask': ['6', 0], 'text': cfg['prompt'], 'seed': cfg['seed'], 'steps': cfg['steps'],
            'cfg': 1.0, 'sampler_name': 'euler', 'scheduler': 'beta', 'denoise': 1.0,
            'noise_mask': True, 'batch_size': 1, 'differential_attention': True, 'use_teacache': False}},
        '12': {'class_type': 'SaveImage', 'inputs': {'images': ['9', 0], 'filename_prefix': f'roadlab/{job["id"]}'}},
    }


    if not profile(cfg)['lora']:
        del graph['13']
        graph['9']['inputs']['model'] = ['1', 0]
    return graph


async def api_get(session, path):
    async with session.get(COMFY + path, timeout=ClientTimeout(total=15)) as response:
        response.raise_for_status()
        return await response.json()


async def requirements(session, cfg=None):
    cfg = cfg or {'object_type': 'traffic_cone'}
    info = await api_get(session, '/object_info')
    needed = {'UNETLoader': ('unet_name', ['flux1-fill-dev.safetensors']),
              'DualCLIPLoader': ('clip_name1', ['clip_l.safetensors', 't5xxl_fp8_e4m3fn_scaled.safetensors']),
              'VAELoader': ('vae_name', ['ae.safetensors'])}
    if profile(cfg)['lora']:
        needed['LoraLoader'] = ('lora_name', [profile(cfg)['lora']])
    nodes = ['SaveImage', 'LoadImage', 'ImageToMask']
    if native_type(cfg):
        nodes += ['CLIPTextEncodeFlux', 'ConditioningZeroOut', 'InpaintModelConditioning',
                  'KSampler', 'VAEDecode', 'ImageCompositeMasked']
    else:
        nodes += ['FluxFillSampler']
    missing = [n for n in nodes if n not in info]
    for node, (field, names) in needed.items():
        options = info.get(node, {}).get('input', {}).get('required', {}).get(field, [[]])[0]
        missing.extend(name for name in names if name not in options)
    needs_lanes = cfg.get('object_type', 'traffic_cone') == 'traffic_cone' or cfg.get('placement_mode') == 'auto'
    if needs_lanes and not (Path(LANE_DIR) / '_vendor/weights/culane_r18.pth').is_file():
        missing.append('CLRNet 车道检测权重')
    if cfg.get('protect_original',True) and not model_available():
        missing.append('原有目标保护模型')
    return missing


async def health(request):
    object_type = request.query.get('object_type', 'traffic_cone')
    if object_type not in PROFILES:
        raise web.HTTPBadRequest(text='未知的生成物体')
    cfg = {'object_type': object_type, 'placement_mode': request.query.get('placement_mode', 'manual')}
    cfg['protect_original']=request.query.get('protect_original','1')!='0'
    if native_type(cfg) and cfg['placement_mode'] != 'manual':
        raise web.HTTPBadRequest(text='本类别需要手动选择编辑区域')
    profiles = {name: {'label': p['label'], 'default_prompt': p['prompt'], 'category_id': p['category_id'],
                      'manual_rectangle': p.get('engine') == 'native_full_frame',
                      'placement_defaults': p.get('placement_defaults', {'x': .60, 'y': .82, 'width': .085}),
                      'placement_note': p.get('placement_note', '')}
                for name, p in PROFILES.items()}
    try:
        missing = await requirements(request.app['session'], cfg)
        return web.json_response({'ready': not missing, 'missing': missing,
                                 'default_prompt': profile(cfg)['prompt'], 'profiles': profiles})
    except Exception:
        return web.json_response({'ready': False, 'missing': ['ComfyUI 服务暂时未连接'],
                                 'default_prompt': profile(cfg)['prompt'], 'profiles': profiles})


def settings(raw):
    try:
        object_type = raw.get('object_type', 'traffic_cone')
        if object_type not in PROFILES:
            raise ValueError()
        result = {'object_type': object_type,
                  'steps': int(raw.get('steps', 30)), 'max_side': int(raw.get('max_side', 1024)),
                  'expand': int(raw.get('expand', 50)),
                  'seed': int(raw['seed']) if str(raw.get('seed', '')).strip() else secrets.randbelow(2**53),
                  'prompt': str(raw.get('prompt', PROFILES[object_type]['prompt'])).strip()}
        result['protect_original']=raw.get('protect_original',True)
        if not isinstance(result['protect_original'],bool):raise ValueError('保护开关需要布尔值')
        if native_type(result):
            if raw.get('placement_mode', 'manual') != 'manual':
                raise ValueError('本类别需要手动选择编辑区域')
            result['placement_mode'] = 'manual'
            result['placement'] = rectangle_placement(raw.get('placement'))
        elif object_type == 'dustbin':
            result['placement_mode'] = raw.get('placement_mode', 'manual')
            if result['placement_mode'] not in ('manual', 'auto'):
                raise ValueError()
            if result['placement_mode'] == 'manual':
                result['placement'] = manual_placement(raw.get('placement', {}))
        if object_type=='traffic_cone' or result.get('placement_mode')=='auto':
            result['use_lane_cache']=raw.get('use_lane_cache',True)
            if not isinstance(result['use_lane_cache'],bool):raise ValueError('缓存开关需要布尔值')
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
        raise web.HTTPBadRequest(text='参数格式不正确')
    if not (1 <= result['steps'] <= 50 and result['max_side'] in (768, 1024, 1536, 1920)
            and 0 <= result['expand'] <= 180 and 0 <= result['seed'] <= 2**53 - 1
            and 1 <= len(result['prompt']) <= 2000):
        raise web.HTTPBadRequest(text='参数超出允许范围')
    return result


async def create_job(request):
    # Reject cross-origin submissions without adding credentials to the browser.
    if request.headers.get('Sec-Fetch-Site') == 'cross-site':
        raise web.HTTPForbidden(text='请从本网站上传图片')
    if request.content_type != 'multipart/form-data':
        raise web.HTTPBadRequest(text='请通过上传表单提交图片')
    if len([j for j in request.app['jobs'].values() if j['status'] not in TERMINAL]) >= 8:
        raise web.HTTPTooManyRequests(text='等待任务较多，请稍后重试')
    if request.content_length and request.content_length > 14 * 1024 * 1024:
        raise web.HTTPRequestEntityTooLarge(max_size=14*1024*1024, actual_size=request.content_length)
    reader = await request.multipart()
    image = None
    raw = {}
    total = 0
    while part := await reader.next():
        chunks = bytearray()
        while chunk := await part.read_chunk():
            total += len(chunk)
            if total > 12 * 1024 * 1024:
                raise web.HTTPBadRequest(text='上传总大小不能超过 12 MB')
            chunks.extend(chunk)
        if part.name == 'image':
            image = bytes(chunks)
        elif part.name == 'settings':
            try:
                raw = json.loads(chunks)
                if not isinstance(raw, dict):
                    raise ValueError()
            except (ValueError, UnicodeError):
                raise web.HTTPBadRequest(text='参数格式不正确')
    if not image:
        raise web.HTTPBadRequest(text='请选择一张道路图片')
    cfg = settings(raw)
    if len([j for j in request.app['jobs'].values() if j['status'] not in TERMINAL]) >= 8:
        raise web.HTTPTooManyRequests(text='等待任务较多，请稍后重试')
    try:
        with Image.open(io.BytesIO(image)) as source:
            if source.format not in ('JPEG', 'PNG', 'WEBP') or source.width * source.height > 24_000_000:
                raise ValueError()
            im = ImageOps.exif_transpose(source).convert('RGB')
            im.load()
            if min(im.size) < 64:
                raise ValueError()
    except (UnidentifiedImageError, ValueError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise web.HTTPBadRequest(text='图片无法读取。请使用 JPG、PNG 或 WebP，尺寸至少 64 像素且不超过 2400 万像素。')
    job_id = secrets.token_hex(12)
    original_size = list(im.size)
    original = im
    im = im.copy()
    im.thumbnail((cfg['max_side'], cfg['max_side']), Image.Resampling.LANCZOS)
    if native_type(cfg):
        try:
            rectangle_region(im.size, cfg['placement'])
        except ValueError as exc:
            raise web.HTTPBadRequest(text=str(exc))
    elif cfg.get('placement_mode') == 'manual':
        p = cfg['placement']
        try:
            object_region(im.width, im.height, p['x']*im.width, p['y']*im.height, p['width']*im.width)
        except ValueError as exc:
            raise web.HTTPBadRequest(text=str(exc))
    folder = DATA / job_id
    folder.mkdir(parents=True)
    original.save(folder / 'original.png')
    im.save(folder / 'input.png')
    job = {'id': job_id, 'status': 'queued', 'message': '已加入处理队列', 'progress': 0,
           'created_at': time.time(), 'settings': cfg, 'original_size': original_size,
           'size': list(im.size), 'lane_count': None}
    save(job)
    request.app['jobs'][job_id] = job
    await request.app['queue'].put(job_id)
    return web.json_response(public(job), status=202)


async def list_jobs(request):
    jobs = sorted(request.app['jobs'].values(), key=lambda j: j['created_at'], reverse=True)[:24]
    return web.json_response([public(j) for j in jobs])


def get_job(request):
    job = request.app['jobs'].get(request.match_info['job_id'])
    if job is None:
        raise web.HTTPNotFound(text='任务不存在')
    return job


async def read_job(request):
    return web.json_response(public(get_job(request)))


async def experiment_response(request):
    job = get_job(request)
    path = DATA / job['id'] / 'experiment.json'
    if not path.is_file():
        raise web.HTTPNotFound(text='该任务尚无实验记录')
    return web.FileResponse(path, headers={'Content-Disposition': f'attachment; filename="experiment_{job["id"]}.json"'})


async def review_job(request):
    if request.headers.get('Sec-Fetch-Site') == 'cross-site':
        raise web.HTTPForbidden(text='请从本网站审核')
    job = get_job(request)
    if job['status'] != 'done':
        raise web.HTTPBadRequest(text='请等待生成成功后再审核')
    try:
        raw = await request.json()
        verdict = raw['verdict']
        reviewer = raw.get('reviewer', 'project_user')
        if reviewer not in ('project_user', 'codex_visual_review'):
            raise ValueError('未知的审核来源')
        boxes = annotation_boxes(raw.get('boxes', []))
        if verdict not in ('accepted', 'rejected'):
            raise ValueError('未知的审核结论')
        single = job['settings'].get('object_type') == 'dustbin' or profile(job['settings']).get('single_object', False)
        if verdict == 'accepted' and (not boxes or (single and len(boxes) != 1)):
            raise ValueError('通过审核需绘制实际物体框；本次单物体生成应恰好标注一个')
        review = {'verdict': verdict, 'boxes': boxes if verdict == 'accepted' else [],
                  'reviewer': reviewer,
                  'notes': str(raw.get('notes', ''))[:2000], 'reviewed_at': time.time(),
                  'coordinate_space': 'processed_result_normalized_xywh'}
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise web.HTTPBadRequest(text=str(exc) or '审核格式不正确')
    (DATA / job['id'] / 'annotation_review.json').write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding='utf-8')
    update(job, review=review)
    return web.json_response(public(job))


async def export_annotations(request):
    return web.json_response(coco(request.app['jobs'].values(), DATA), headers={
        'Content-Disposition': 'attachment; filename="roadlab_reviewed_targets.coco.json"'})


async def bundle_response(request):
    job = get_job(request)
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in ('original.png', 'input.png', 'mask.png', 'overlay.png', 'result.png',
                     'job.json', 'prompt.json', 'mask_meta.json', 'lanes.json', 'comfy_history.json',
                     'experiment.json', 'annotation_review.json', 'generation-input.png',
                     'generation-mask.png', 'generation-blend.png'):
            path = DATA / job['id'] / name
            if path.is_file():
                archive.write(path, job['id']+'/'+name)
        for name in ('result-generated.png','protection.png','protection-overlay.png','scene-labels.png','scene.json'):
            path=DATA/job['id']/name
            if path.is_file():archive.write(path,job['id']+'/'+name)
        archive.writestr('annotations.coco.json', json.dumps(coco([job], DATA), ensure_ascii=False, indent=2))
    return web.Response(body=payload.getvalue(), content_type='application/zip', headers={
        'Content-Disposition': f'attachment; filename="roadlab_{job["id"]}.zip"'})


async def file_response(request):
    job = get_job(request)
    name = request.match_info['name']
    if name not in ('original', 'input', 'mask', 'overlay', 'result','protection-overlay'):
        raise web.HTTPNotFound()
    path = DATA / job['id'] / f'{name}.png'
    if not path.is_file():
        raise web.HTTPNotFound()
    response = web.FileResponse(path)
    if request.query.get('download'):
        response.headers['Content-Disposition'] = f'attachment; filename="roadlab_{job["id"]}_{name}.png"'
    return response


async def cancel_job(request):
    job = get_job(request)
    if job['status'] in TERMINAL:
        return web.json_response(public(job))
    session = request.app['session']
    prompt_id = job.get('prompt_id')
    if prompt_id:
        await cancel_prompt(session, prompt_id)
    update(job, status='cancelled', message='任务已取消')
    return web.json_response(public(job))


async def cancel_prompt(session, prompt_id):
    # This ComfyUI version atomically targets one ID, avoiding a global interrupt.
    async with session.post(COMFY + '/api/jobs/' + prompt_id + '/cancel', json={}) as response:
        response.raise_for_status()


async def prepare_scene(app,job,folder):
    process=app.get('scene_process')
    if process is None or process.returncode is not None:
        CACHE.mkdir(parents=True,exist_ok=True)
        with (CACHE/'scene-worker.log').open('ab') as log:
            process=await asyncio.create_subprocess_exec(sys.executable,str(ROOT/'scene_service.py'),
                '--cache',str(CACHE),stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,
                stderr=log,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
        app['scene_process']=process
    process.stdin.write((json.dumps({'id':job['id'],'folder':str(folder)})+'\n').encode())
    await process.stdin.drain()
    try:
        line=await asyncio.wait_for(process.stdout.readline(),timeout=60)
        response=json.loads(line)
        if response.get('id')!=job['id'] or response.get('error'):raise RuntimeError(response.get('error','Scene response mismatch'))
        return response['scene']
    except (ValueError,RuntimeError,asyncio.TimeoutError):
        if process.returncode is None:
            process.terminate();await process.wait()
        app['scene_process']=None
        raise


async def run_job(app, job):
    folder = DATA / job['id']
    session = app['session']
    phase_start=time.perf_counter();timings={'app_queue_seconds':round(job.get('started_at',time.time())-job['created_at'],4)}
    missing = await requirements(session, job['settings'])
    if job['status'] == 'cancelled':
        return
    if missing:
        raise RuntimeError('缺少运行组件：' + '、'.join(missing))
    update(job, status='masking', message='正在生成物体遮罩', progress=8)
    proc = await asyncio.create_subprocess_exec(
        sys.executable, str(ROOT / 'mask_worker.py'), str(folder),
        str(job['settings']['expand']), LANE_DIR, str(CACHE),
        env=dict(os.environ,ROADLAB_DEFER_SCENE='1'),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        output, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
    except BaseException:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await proc.wait()
        raise
    (folder / 'mask.log').write_bytes(output)
    if job['status'] == 'cancelled':
        return
    if proc.returncode:
        log.error('Mask failed %s: %s', job['id'], output.decode(errors='replace'))
        raise RuntimeError('自动遮罩生成失败，请检查车道模型运行日志。')
    meta = json.loads((folder / 'mask_meta.json').read_text())
    if job['settings'].get('protect_original',False):
        meta['scene']=await prepare_scene(app,job,folder)
        (folder/'mask_meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    timings['preparation_seconds']=round(time.perf_counter()-phase_start,4)
    if job['status']=='cancelled':return
    update(job, lane_count=meta['lane_count'], placement=meta.get('placement'),
           preprocessing={'lane_cache':meta.get('lane_cache'),'scene':meta.get('scene')})
    if meta['coverage'] < 0.0001 or (job['settings'].get('object_type', 'traffic_cone') == 'traffic_cone' and not meta['lane_count']):
        raise RuntimeError('未检测到可靠车道线。请换一张车道标线更清晰的道路图片。')
    dest = COMFY_DATA / 'input/roadlab' / job['id']
    dest.mkdir(parents=True, exist_ok=True)
    inputs = ['input.png', 'mask.png']
    if native_type(job['settings']):
        inputs += ['generation-input.png', 'generation-mask.png', 'generation-blend.png']
    for name in inputs:
        shutil.copyfile(folder / name, dest / name)
    prompt = graph(job)
    (folder / 'prompt.json').write_text(json.dumps(prompt, indent=2), encoding='utf-8')
    manifest(folder, job, prompt)
    update(job, status='generating', message='遮罩已生成，正在提交生图任务', progress=20)
    async with session.ws_connect(COMFY + '/ws?clientId=roadlab-' + job['id'], heartbeat=30) as ws:
        if job['status'] == 'cancelled':
            return
        async with session.post(COMFY + '/prompt', json={'prompt': prompt, 'client_id': 'roadlab-' + job['id']}) as response:
            result = await response.json()
            if response.status != 200 or 'prompt_id' not in result:
                log.error('Prompt rejected: %s', result)
                raise RuntimeError('ComfyUI 工作流校验失败，请查看服务日志。')
        prompt_id = result['prompt_id']
        update(job, prompt_id=prompt_id, message='正在加载模型或等待 GPU')
        if job['status'] == 'cancelled':
            await cancel_prompt(session, prompt_id)
            update(job, message='任务已取消')
            return
        started, last_check = time.monotonic(), 0
        fast_history_until=0
        while time.monotonic() - started < 1800:
            if job['status'] == 'cancelled':
                return
            if time.monotonic() - last_check > 2 or time.monotonic()<fast_history_until:
                history = await api_get(session, '/history/' + prompt_id)
                last_check = time.monotonic()
                if prompt_id in history:
                    item = history[prompt_id]
                    (folder / 'comfy_history.json').write_text(json.dumps(item, ensure_ascii=False), encoding='utf-8')
                    if item.get('status', {}).get('status_str') != 'success':
                        log.error('Generation failed %s: %s', job['id'], item.get('status'))
                        messages = item.get('status', {}).get('messages', [])
                        reason = next((m[1].get('exception_message', '') for m in messages if m[0] == 'execution_error'), '')
                        raise RuntimeError('ComfyUI 生图失败：' + (reason[:350] or '请查看执行日志'))
                    events=item.get('status',{}).get('messages',[])
                    event_times={name:v.get('timestamp',0)/1000 for name,v in events if name in ('execution_start','execution_success')}
                    if event_times.get('execution_success'):
                        timings['completion_observation_delay_seconds']=round(time.time()-event_times['execution_success'],4)
                        timings['comfy_execution_seconds']=round(event_times['execution_success']-event_times.get('execution_start',event_times['execution_success']),4)
                    images = item.get('outputs', {}).get('12', {}).get('images', [])
                    if not images:
                        raise RuntimeError('任务结束，但未找到输出图片')
                    postprocess_start=time.perf_counter()
                    async with session.get(COMFY + '/view', params=images[0], timeout=ClientTimeout(total=30)) as image_response:
                        image_response.raise_for_status()
                        generated=await image_response.read()
                        (folder / 'result-generated.png').write_bytes(generated)
                    if job['settings'].get('protect_original',False):
                        with Image.open(folder/'input.png') as original,Image.open(folder/'result-generated.png') as raw,Image.open(folder/'protection.png') as protection:
                            protected,metrics=composite_protected(original,raw,protection)
                        protected.save(folder/'result.png')
                        job['protection_checks']=metrics
                    else:
                        (folder/'result.png').write_bytes(generated)
                    timings['postprocess_seconds']=round(time.perf_counter()-postprocess_start,4)
                    if job['status'] == 'cancelled':
                        return
                    elapsed = round(time.time() - job['started_at'], 1)
                    finish_manifest(folder, dict(job, elapsed=elapsed,timings=timings))
                    update(job, status='done', progress=100, message='生成完成', elapsed=elapsed,timings=timings)
                    return
            if time.monotonic()<fast_history_until:
                await asyncio.sleep(.1)
                continue
            try:
                message = await ws.receive(timeout=2)
            except asyncio.TimeoutError:
                continue
            if message.type == WSMsgType.TEXT:
                event = json.loads(message.data)
                values = event.get('data', {})
                if values.get('prompt_id', prompt_id) != prompt_id:
                    continue
                if event.get('type') == 'progress':
                    step, total = values.get('value', 0), values.get('max', job['settings']['steps'])
                    update(job, progress=25 + round(65 * step / max(total, 1)), message=f'正在生图 · {step}/{total} 步')
                elif event.get('type') == 'executing' and values.get('node') == '9':
                    update(job, message='正在编码图片、加载模型并采样')
                elif event.get('type') in ('execution_success','execution_error','execution_interrupted'):
                    last_check=0
                    fast_history_until=time.monotonic()+1.5
            elif message.type in (WSMsgType.CLOSED, WSMsgType.ERROR):
                raise RuntimeError('ComfyUI 连接中断，请检查服务后重新生成')
        await cancel_prompt(session, prompt_id)
        raise RuntimeError('生图超过 30 分钟，任务已停止，请检查 ComfyUI 任务状态')


async def worker(app):
    while True:
        job_id = await app['queue'].get()
        job = app['jobs'][job_id]
        try:
            if job['status'] == 'cancelled':
                continue
            update(job, started_at=time.time())
            await run_job(app, job)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception('Job failed %s', job_id)
            if job.get('prompt_id'):
                with contextlib.suppress(Exception):
                    await cancel_prompt(app['session'], job['prompt_id'])
            if job['status'] != 'cancelled':
                update(job, status='error', message=str(exc))
        finally:
            app['queue'].task_done()


async def lifecycle(app):
    DATA.mkdir(parents=True, exist_ok=True)
    app['jobs'], app['queue'] = {}, asyncio.Queue()
    app['scene_process']=None
    for path in DATA.glob('*/job.json'):
        try:
            job = json.loads(path.read_text(encoding='utf-8'))
            if job['status'] not in TERMINAL:
                update(job, status='error', message='网页服务曾重启，任务未能确认完成。请查看 ComfyUI 后再重试。')
            app['jobs'][job['id']] = job
        except (ValueError, KeyError):
            log.exception('Cannot load %s', path)
    async with ClientSession(timeout=ClientTimeout(total=30)) as session:
        app['session'] = session
        task = asyncio.create_task(worker(app))
        yield
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        process=app.get('scene_process')
        if process is not None and process.returncode is None:
            process.terminate()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(),timeout=5)
            if process.returncode is None:process.kill();await process.wait()


@web.middleware
async def errors(request, handler):
    try:
        response = await handler(request)
    except web.HTTPException as exc:
        if request.path.startswith('/api/'):
            response = web.json_response({'error': exc.text}, status=exc.status)
        else:
            raise
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response


def create_app():
    app = web.Application(client_max_size=14 * 1024 * 1024, middlewares=[errors])
    app.cleanup_ctx.append(lifecycle)
    app.router.add_get('/', lambda _: web.FileResponse(ROOT / 'static/index.html'))
    app.router.add_get('/api/health', health)
    app.router.add_get('/api/jobs', list_jobs)
    app.router.add_post('/api/jobs', create_job)
    app.router.add_get('/api/jobs/{job_id}', read_job)
    app.router.add_get('/api/jobs/{job_id}/experiment', experiment_response)
    app.router.add_get('/api/jobs/{job_id}/bundle', bundle_response)
    app.router.add_post('/api/jobs/{job_id}/review', review_job)
    app.router.add_get('/api/export', export_annotations)
    app.router.add_post('/api/jobs/{job_id}/cancel', cancel_job)
    app.router.add_get('/api/jobs/{job_id}/files/{name}.png', file_response)
    app.router.add_static('/static/', ROOT / 'static')
    return app


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    web.run_app(create_app(), host=os.environ.get('HOST', '127.0.0.1'), port=int(os.environ.get('PORT', '6008')), access_log=None)
