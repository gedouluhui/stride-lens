import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from test_report import sample

ROOT = Path(__file__).resolve().parents[1]


def run_cli(args, cwd, code=None):
    env = os.environ.copy(); env['PYTHONPATH'] = str(ROOT / 'src')
    command = [sys.executable, '-m', 'stridelens', *args] if code is None else [sys.executable, '-c', code, *args]
    return subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30)


class CliTests(unittest.TestCase):
    def test_help_and_render_work_when_torch_import_is_forbidden(self):
        blocker = "import sys,runpy,importlib.abc;\nclass Block(importlib.abc.MetaPathFinder):\n def find_spec(self,name,*args):\n  if name=='torch' or name.startswith('torch.'): raise RuntimeError('torch import forbidden')\nsys.meta_path.insert(0,Block());sys.argv=['stridelens']+sys.argv[1:];runpy.run_module('stridelens',run_name='__main__')"
        with tempfile.TemporaryDirectory() as d:
            result = run_cli(['--help'], d, blocker)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('render', result.stdout)
            Path(d, 'in.json').write_text(json.dumps(sample()), encoding='utf-8')
            result = run_cli(['render', 'in.json', '--out', 'out.html'], d, blocker)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(Path(d, 'out.html').is_file())

    def test_demo_makes_real_snapshot_and_preserves_existing_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            args = ['demo', '--out', 'demo.html', '--json', 'demo.json']
            result = run_cli(args, d)
            self.assertEqual(result.returncode, 0, result.stderr)
            before = Path(d, 'demo.html').read_bytes()
            s = json.loads(Path(d, 'demo.json').read_text(encoding='utf-8'))
            self.assertEqual(len(s['tensors']), 7)
            self.assertEqual([v['status'] for v in s['views']], ['rejected', 'accepted'])
            result = run_cli(args, d)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertEqual(Path(d, 'demo.html').read_bytes(), before)
            result = run_cli(args + ['--force'], d)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_and_missing_input_fail_without_output(self):
        with tempfile.TemporaryDirectory() as d:
            for content in [b'{}', b'not JSON', b'\xff', b'[' * 2000, b' ' * (4 * 1024 * 1024 + 1)]:
                Path(d, 'in.json').write_bytes(content)
                r = run_cli(['render', 'in.json', '--out', 'out.html'], d)
                self.assertEqual(r.returncode, 2, r.stderr)
                self.assertFalse(Path(d, 'out.html').exists())
            r = run_cli(['render', 'absent.json', '--out', 'out.html'], d)
            self.assertEqual(r.returncode, 2)

    def test_demo_output_paths_must_differ(self):
        with tempfile.TemporaryDirectory() as d:
            r = run_cli(['demo', '--out', 'same', '--json', 'same'], d)
            self.assertEqual(r.returncode, 2)
            self.assertFalse(Path(d, 'same').exists())

    def test_invalid_schema_never_prints_traceback(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, 'in.json').write_text('null', encoding='utf-8')
            r = run_cli(['render', 'in.json', '--out', 'out.html'], d)
            self.assertEqual(r.returncode, 2)
            self.assertNotIn('Traceback', r.stderr)


if __name__ == '__main__':
    unittest.main()
