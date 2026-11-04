"""Export API prompt graphs from the exact web graph factory, without a GPU."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'roadlab_web'))
import app


def export(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in app.PROFILES:
        raw = {'object_type': name, 'seed': 42, 'protect_original': False}
        if name in ('warning_sign', 'pedestrian'):
            raw['placement'] = {'x': .8, 'y': .85, 'width': .1, 'height': .3}
        graph = app.graph({'id': 'example-' + name, 'settings': app.settings(raw)})
        (output / (name + '.json')).write_text(json.dumps(graph, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'workflows' / 'api')
    export(p.parse_args().output)
