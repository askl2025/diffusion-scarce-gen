import asyncio
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

import app
from experiments import coco
from scene_analysis import composite_protected,analyze
from preprocess_cache import lane_points
from placement import annotation_boxes, object_region, plan_dustbin, dustbin_mask, rectangle_region


class GeometryTests(unittest.TestCase):
    def test_native_worker_pads_odd_images_without_moving_the_region(self):
        for object_type, size in [('warning_sign', (1001, 667)), ('pedestrian', (667, 1001))]:
            with self.subTest(object_type=object_type), tempfile.TemporaryDirectory() as temp:
                p = Path(temp)
                pixels = np.zeros((size[1], size[0], 3), dtype=np.uint8)
                pixels[:, :, 0] = np.arange(size[0], dtype=np.uint16) % 255
                Image.fromarray(pixels).save(p/'input.png')
                cfg = app.settings({'object_type': object_type, 'protect_original':False,'placement': {'x': .6, 'y': .85, 'width': .10, 'height': .3}})
                (p/'job.json').write_text(json.dumps({'settings': cfg}))
                result = subprocess.run([sys.executable, str(Path(__file__).with_name('mask_worker.py')),
                                         temp, '50', '/does-not-exist'], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                meta = json.loads((p/'mask_meta.json').read_text())
                self.assertEqual(meta['engine'], 'native_full_frame')
                self.assertEqual(meta['generation_size'], [(size[0]+15)//16*16, (size[1]+15)//16*16])
                padded = np.asarray(Image.open(p/'generation-input.png'))
                self.assertTrue(np.array_equal(padded[:size[1], :size[0]], pixels))
                raw_mask = np.asarray(Image.open(p/'mask.png'))
                padded_mask = np.asarray(Image.open(p/'generation-mask.png'))
                self.assertTrue(np.array_equal(padded_mask[:size[1], :size[0]], raw_mask))
                self.assertFalse(padded_mask[size[1]:].any())
                self.assertFalse(padded_mask[:, size[0]:].any())
                self.assertEqual(meta['lane_count'], 0)

    def test_manual_without_lane_predictions(self):
        region = plan_dustbin([], 1024, 544, 'manual', {'x': .6, 'y': .82, 'width': .085})
        mask = np.asarray(dustbin_mask((1024, 544), region))
        self.assertTrue(mask[:, :, 1].any())
        self.assertFalse(mask[:, :, 0].any())
        self.assertGreater(region['body_bbox_xyxy'][1], 0)

    def test_auto_scale_tracks_local_lane_width(self):
        far = [[[430, 200], [400, 500]], [[570, 200], [600, 500]]]
        near = [[[380, 200], [200, 500]], [[620, 200], [800, 500]]]
        self.assertLess(plan_dustbin(far, 1000, 500)['body_width'],
                        plan_dustbin(near, 1000, 500)['body_width'])

    def test_auto_requires_both_boundaries(self):
        with self.assertRaisesRegex(ValueError, '手动'):
            plan_dustbin([[[100, 200], [200, 500]]], 1000, 500)

    def test_out_of_frame_and_nan_rejected(self):
        with self.assertRaises(ValueError):
            object_region(1000, 500, 50, 400, 180)
        with self.assertRaises(web.HTTPBadRequest):
            app.settings({'object_type': 'dustbin', 'placement': {'x': 'nan'}})
        with self.assertRaises(ValueError):
            annotation_boxes([[.9, .1, .2, .2]])

    def test_manual_worker_has_no_lane_dependency(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            Image.new('RGB', (1024, 544)).save(p/'input.png')
            (p/'job.json').write_text(json.dumps({'settings': app.settings({'object_type': 'dustbin','protect_original':False})}))
            result = subprocess.run([sys.executable, str(Path(__file__).with_name('mask_worker.py')),
                                     temp, '50', '/does-not-exist'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            meta = json.loads((p/'mask_meta.json').read_text())
            self.assertEqual(meta['backend'], 'manual')
            self.assertEqual(meta['lane_count'], 0)
            self.assertGreater(meta['coverage'], 0)


class OptimizationTests(unittest.TestCase):
    def test_protected_pixels_exact_and_distant_generated_pixels_unchanged(self):
        original=Image.new('RGB',(80,60),(20,80,120));generated=Image.new('RGB',(80,60),(200,30,20))
        mask=Image.new('L',(80,60));mask.paste(255,(30,20,45,40))
        final,metrics=composite_protected(original,generated,mask)
        array=np.asarray(final)
        self.assertTrue(np.array_equal(array[20:40,30:45],np.asarray(original)[20:40,30:45]))
        self.assertTrue(np.array_equal(array[0,0],np.asarray(generated)[0,0]))
        self.assertEqual(metrics['after_mae'],0)
        self.assertGreater(metrics['before_mae'],0)

    def test_scene_cache_invalidates_for_input_and_model_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);model=folder/'model';model.mkdir()
            for name in ['config.json','preprocessor_config.json','pytorch_model.bin']:(model/name).write_bytes(b'v1')
            image=Image.new('RGB',(40,30));labels=np.zeros((30,40),dtype='uint8');mask=labels.copy();mask[5:10,5:10]=255
            with patch('scene_analysis._segment',return_value=(labels,mask)) as predict:
                a=analyze(image,folder/'cache',model)[2];b=analyze(image,folder/'cache',model)[2]
                self.assertFalse(a['cache_hit']);self.assertTrue(b['cache_hit']);self.assertEqual(predict.call_count,1)
                image.putpixel((1,1),(1,2,3));analyze(image,folder/'cache',model)
                (model/'pytorch_model.bin').write_bytes(b'v2');analyze(image,folder/'cache',model)
                self.assertEqual(predict.call_count,3)

    def test_lane_cache_reuses_predictions_and_invalidates_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);model=folder/'lane'
            names=['detect_lanes_clrnet.py','detect_lanes_v2.py','_vendor/clrnet_runtime.py','_vendor/weights/culane_r18.pth','_vendor/CLRNet/configs/clrnet/clr_resnet18_culane.py']
            for name in names:
                p=model/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'v1')
            image=Image.new('RGB',(80,60));points=[[[10,20],[30,50]]]
            with patch('preprocess_cache._detect',return_value=points) as detect:
                a=lane_points(image,model,folder/'cache')[1];b=lane_points(image,model,folder/'cache')[1]
                self.assertFalse(a['hit']);self.assertTrue(b['hit']);self.assertEqual(detect.call_count,1)
                (model/'detect_lanes_clrnet.py').write_bytes(b'v2');lane_points(image,model,folder/'cache')
                self.assertEqual(detect.call_count,2)


class GraphTests(unittest.TestCase):
    def test_new_categories_use_native_model_and_original_size_destination(self):
        for name in ['warning_sign', 'pedestrian']:
            cfg = app.settings({'object_type': name, 'placement': {'x': .8, 'y': .85, 'width': .1, 'height': .3}})
            graph = app.graph({'id': 'native', 'settings': cfg})
            self.assertEqual(graph['9']['class_type'], 'KSampler')
            self.assertFalse(any(n['class_type'] in ('LoraLoader', 'FluxFillSampler') for n in graph.values()))
            self.assertEqual(graph['21']['inputs']['guidance'], 30)
            self.assertEqual(graph['27']['inputs']['destination'], ['28', 0])
            self.assertEqual(graph['28']['inputs']['image'], 'roadlab/native/input.png')
            self.assertEqual(graph['4']['inputs']['image'], 'roadlab/native/generation-input.png')

    def test_new_categories_require_an_explicit_manual_rectangle(self):
        for raw in [{'object_type': 'warning_sign'},
                    {'object_type': 'pedestrian', 'placement_mode': 'auto'},
                    {'object_type': 'pedestrian', 'placement': {'x': .8, 'y': .8, 'width': .1, 'height': 'nan'}}]:
            with self.subTest(raw=raw), self.assertRaises(web.HTTPBadRequest):
                app.settings(raw)

    def test_legacy_default_and_dustbin_branches(self):
        cone = app.graph({'id': 'test', 'settings': app.settings({'seed': 1})})
        dustbin = app.graph({'id': 'test', 'settings': app.settings({'seed': 1, 'object_type': 'dustbin'})})
        self.assertEqual(cone['9']['inputs']['model'], ['13', 0])
        self.assertEqual(cone['13']['inputs']['strength_model'], 1)
        self.assertNotIn('13', dustbin)
        self.assertEqual(dustbin['9']['inputs']['model'], ['1', 0])

    def test_legacy_graph_matches_frozen_baseline(self):
        fixture = Path(__file__).resolve().parents[1] / 'tests/fixtures/legacy_graphs.json'
        expected = json.loads(fixture.read_text(encoding='utf-8'))
        for object_type in ['traffic_cone', 'dustbin']:
            cfg = {'prompt': 'test', 'steps': 30, 'seed': 42, 'object_type': object_type}
            self.assertEqual(app.graph({'id': 'baseline', 'settings': cfg}), expected[object_type])


async def idle_worker(application):
    await asyncio.Event().wait()


class APITests(unittest.IsolatedAsyncioTestCase):
    async def test_new_categories_upload_review_and_coda_ids(self):
        for name, category in [('warning_sign', 20), ('pedestrian', 1)]:
            response = await self.upload({'object_type': name, 'placement': {'x': .8, 'y': .85, 'width': .1, 'height': .3}})
            self.assertEqual(response.status, 202, await response.text())
            job = self.client.server.app['jobs'][(await response.json())['id']]
            job['status'] = 'done'
            folder = app.DATA/job['id']
            Image.new('RGB', (1024, 544)).save(folder/'result.png')
            url = '/api/jobs/'+job['id']+'/review'
            response = await self.client.post(url, json={'verdict': 'accepted', 'boxes': [[.1, .2, .1, .3], [.3, .2, .1, .3]]})
            self.assertEqual(response.status, 400)
            response = await self.client.post(url, json={'verdict': 'accepted', 'boxes': [[.1, .2, .1, .3]]})
            self.assertEqual(response.status, 200, await response.text())
            self.assertEqual(coco([job], app.DATA)['annotations'][0]['category_id'], category)

    async def test_native_dependency_check_does_not_require_starnodes_or_lane_weights(self):
        info = {n: {} for n in ['SaveImage', 'LoadImage', 'ImageToMask', 'CLIPTextEncodeFlux', 'ConditioningZeroOut',
                                'InpaintModelConditioning', 'KSampler', 'VAEDecode', 'ImageCompositeMasked']}
        for n, field, names in [('UNETLoader', 'unet_name', ['flux1-fill-dev.safetensors']),
                                ('DualCLIPLoader', 'clip_name1', ['clip_l.safetensors', 't5xxl_fp8_e4m3fn_scaled.safetensors']),
                                ('VAELoader', 'vae_name', ['ae.safetensors'])]:
            info[n] = {'input': {'required': {field: [names]}}}
        async def fake_info(session, path):
            return info
        with patch.object(app, 'api_get', fake_info), patch.object(app, 'LANE_DIR', '/not-present'):
            response = await self.client.get('/api/health?object_type=warning_sign')
            self.assertTrue((await response.json())['ready'])
            response = await self.client.get('/api/health?object_type=pedestrian&placement_mode=auto')
            self.assertEqual(response.status, 400)
            del info['KSampler']
            response = await self.client.get('/api/health?object_type=pedestrian')
            self.assertIn('KSampler', (await response.json())['missing'])

    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.data_patch = patch.object(app, 'DATA', Path(self.temp.name))
        self.worker_patch = patch.object(app, 'worker', idle_worker)
        self.model_patch = patch.object(app,'model_available',return_value=True)
        self.data_patch.start(); self.worker_patch.start();self.model_patch.start()
        self.client = TestClient(TestServer(app.create_app()))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        self.worker_patch.stop(); self.data_patch.stop(); self.model_patch.stop();self.temp.cleanup()

    async def upload(self, settings):
        image = io.BytesIO(); Image.new('RGB', (1024, 544)).save(image, format='PNG')
        form = FormData(); form.add_field('image', image.getvalue(), filename='test.png', content_type='image/png')
        form.add_field('settings', json.dumps(settings))
        return await self.client.post('/api/jobs', data=form)

    async def test_upload_cancel_and_legacy_default(self):
        for settings, expected in [({}, 'traffic_cone'), ({'object_type': 'dustbin'}, 'dustbin')]:
            r = await self.upload(settings)
            self.assertEqual(r.status, 202, await r.text())
            job = await r.json()
            self.assertEqual(job['settings']['object_type'], expected)
            r = await self.client.post('/api/jobs/'+job['id']+'/cancel')
            self.assertEqual((await r.json())['status'], 'cancelled')

    async def test_invalid_upload_settings_leave_no_job(self):
        r = await self.upload({'object_type': 'dustbin', 'placement': {'x': .05, 'width': .18}})
        self.assertEqual(r.status, 400)
        self.assertEqual(list(app.DATA.iterdir()), [])

    async def test_review_export_and_download_use_actual_boxes(self):
        job = await (await self.upload({'object_type': 'dustbin'})).json()
        job = self.client.server.app['jobs'][job['id']]
        job['status'] = 'done'
        folder = app.DATA/job['id']
        Image.new('RGB', (1024, 544)).save(folder/'result.png')
        r = await self.client.post('/api/jobs/'+job['id']+'/review', json={'verdict': 'accepted', 'boxes': []})
        self.assertEqual(r.status, 400)
        self.assertEqual(coco([job], app.DATA)['annotations'], [])
        r = await self.client.post('/api/jobs/'+job['id']+'/review', json={'verdict': 'accepted', 'boxes': [[.1, .2, .1, .3]]})
        self.assertEqual(r.status, 200, await r.text())
        exported = await (await self.client.get('/api/export')).json()
        self.assertEqual(exported['annotations'][0]['category_id'], 29)
        self.assertEqual(exported['annotations'][0]['bbox'], [102.4, 108.8, 102.4, 163.2])
        r = await self.client.get('/api/jobs/'+job['id']+'/bundle')
        with zipfile.ZipFile(io.BytesIO(await r.read())) as archive:
            self.assertIn(job['id']+'/result.png', archive.namelist())
            self.assertEqual(len(json.loads(archive.read('annotations.coco.json'))['annotations']), 1)
        r = await self.client.post('/api/jobs/'+job['id']+'/review', json={'verdict': 'rejected'})
        self.assertEqual(r.status, 200)
        self.assertEqual((await (await self.client.get('/api/export')).json())['images'], [])

    async def test_dependency_checks_are_per_object(self):
        info = {n: {} for n in ('FluxFillSampler', 'SaveImage', 'LoadImage', 'ImageToMask')}
        for n, field, names in [('UNETLoader', 'unet_name', ['flux1-fill-dev.safetensors']),
                                ('DualCLIPLoader', 'clip_name1', ['clip_l.safetensors', 't5xxl_fp8_e4m3fn_scaled.safetensors']),
                                ('VAELoader', 'vae_name', ['ae.safetensors'])]:
            info[n] = {'input': {'required': {field: [names]}}}
        async def fake_info(session, path):
            return info
        with patch.object(app, 'api_get', fake_info), patch.object(app, 'LANE_DIR', '/not-present'):
            r = await self.client.get('/api/health?object_type=dustbin&placement_mode=manual')
            self.assertTrue((await r.json())['ready'])
            r = await self.client.get('/api/health?object_type=traffic_cone')
            self.assertFalse((await r.json())['ready'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
