"""Run local behavior tests without editable/global installs."""
from pathlib import Path
import os
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
env = os.environ.copy()
env['PYTHONPATH'] = str(root / 'src')
result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], cwd=root, env=env)
raise SystemExit(result.returncode)
