"""Run the offline CPU suite from any working directory."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in ['roadlab_web', 'tests']:
    result = subprocess.run([sys.executable, '-m', 'unittest', 'discover',
                             '-s', str(ROOT / directory), '-p', 'test_*.py', '-v'], cwd=ROOT)
    if result.returncode:
        raise SystemExit(result.returncode)
