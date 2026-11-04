import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import socket
import time
import unittest
import urllib.error
import urllib.request
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


class PortableToolTests(unittest.TestCase):
    def test_bundled_clrnet_snapshot_is_complete(self):
        snapshot=json.loads((ROOT/'docs/clrnet-snapshot.json').read_text('utf-8'))
        for name, expected in snapshot['files'].items():
            with self.subTest(path=name):
                self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),expected)

    def test_web_starts_outside_the_repo_and_serves_static_files_without_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            with socket.socket() as listener:
                listener.bind(('127.0.0.1', 0))
                port = listener.getsockname()[1]
            env = dict(os.environ, HOST='127.0.0.1', PORT=str(port),
                       ROADLAB_HOME=str(Path(tmp)/'runtime'), COMFY_URL='http://127.0.0.1:1')
            with (Path(tmp)/'server.log').open('w',encoding='utf-8') as log:
                process = subprocess.Popen([sys.executable,str(ROOT/'roadlab_web/app.py')],
                                           cwd=tmp,env=env,stdout=log,stderr=log)
                try:
                    base=f'http://127.0.0.1:{port}'
                    for _ in range(80):
                        if process.poll() is not None:
                            self.fail('Server exited: '+(Path(tmp)/'server.log').read_text('utf-8'))
                        try:
                            with urllib.request.urlopen(base+'/',timeout=1) as response:
                                self.assertIn('RoadLab',response.read().decode('utf-8'))
                            break
                        except (urllib.error.URLError,TimeoutError):
                            time.sleep(.1)
                    else:self.fail('Server startup timed out')
                    with urllib.request.urlopen(base+'/static/app.js',timeout=2) as response:
                        self.assertTrue(response.read())
                    with urllib.request.urlopen(base+'/api/health?object_type=warning_sign&protect_original=0',timeout=20) as response:
                        self.assertFalse(json.load(response)['ready'])
                finally:
                    process.terminate()
                    process.wait(timeout=5)

    def test_configuration_is_independent_of_working_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = ('import sys,json;sys.path.insert(0,' + repr(str(ROOT/'roadlab_web')) + ');'
                      'import config;print(json.dumps([str(config.DATA),str(config.LANE_DIR),config.COMFY_URL]))')
            env = dict(os.environ, ROADLAB_HOME=str(Path(tmp)/'runtime'), COMFY_URL='http://localhost:8188/')
            values = [json.loads(subprocess.check_output([sys.executable, '-c', script], cwd=cwd, env=env))
                      for cwd in [ROOT, Path(tmp)]]
            self.assertEqual(values[0], values[1])
            self.assertEqual(values[0][0], str(Path(tmp)/'runtime/data'))
            self.assertEqual(values[0][2], 'http://localhost:8188')

    def test_sequential_graph_feeds_each_composite_into_the_next_slot(self):
        spec = importlib.util.spec_from_file_location('multi_object_prepare', ROOT/'experiments/multi_object/prepare.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp)/'source.png';Image.new('RGB',(128,96)).save(image)
            graph, metadata = module.prepare(image,Path(tmp)/'case',[[.1,.5,.2,.3],[.6,.5,.2,.3]],'One stone ball.', 'sequential',100)
            self.assertEqual(metadata['sampler_passes'],2)
            self.assertEqual(graph['slot2_23']['inputs']['pixels'],['slot1_27',0])
            self.assertEqual(graph['slot2_27']['inputs']['destination'],['slot1_27',0])
            self.assertEqual(graph['slot2_9']['inputs']['seed'],101)
            self.assertEqual(graph['12']['inputs']['images'],['slot2_27',0])
            for node in graph.values():
                for value in node['inputs'].values():
                    if isinstance(value,list) and len(value)==2:
                        self.assertIn(value[0],graph)
            self.assertFalse(metadata['submitted'])

    def test_invalid_slot_size_does_not_create_a_partial_experiment(self):
        spec = importlib.util.spec_from_file_location('multi_object_prepare_invalid', ROOT/'experiments/multi_object/prepare.py')
        module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            image=Path(tmp)/'source.png';Image.new('RGB',(127,95)).save(image)
            output=Path(tmp)/'case'
            with self.assertRaisesRegex(ValueError,'divisible by 16'):
                module.prepare(image,output,[[.1,.5,.2,.3]],'One ball.','sequential')
            self.assertFalse(output.exists())


if __name__=='__main__':unittest.main()
