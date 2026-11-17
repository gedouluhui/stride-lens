"""Offline command line interface."""
import argparse
import json
from pathlib import Path
import sys

from .report import atomic_write, write_report
from .schema import MAX_JSON_BYTES


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Read-only tensor layout evidence. No server or model needed.')
    commands = parser.add_subparsers(dest='command', required=True)
    demo = commands.add_parser('demo', help='generate an offline report from real synthetic CPU tensors')
    demo.add_argument('--out', default='stridelens-demo.html')
    demo.add_argument('--json', dest='json_out')
    render = commands.add_parser('render', help='render a saved snapshot without importing PyTorch')
    render.add_argument('input')
    render.add_argument('--out', required=True)
    for sub in (demo, render):
        sub.add_argument('--force', action='store_true', help='replace existing output explicitly')
    args = parser.parse_args(argv)
    try:
        outputs = [Path(args.out)]
        if args.command == 'demo' and args.json_out:
            outputs.append(Path(args.json_out))
        if len({p.resolve() for p in outputs}) != len(outputs):
            raise ValueError('HTML and JSON output paths must differ')
        if not args.force and any(p.exists() for p in outputs):
            raise FileExistsError('output exists; use --force to replace it')
        if args.command == 'demo':
            from .demo import demo_snapshot
            snapshot = demo_snapshot()
        else:
            if Path(args.input).resolve() in {p.resolve() for p in outputs}:
                raise ValueError('input and output paths must differ')
            with open(args.input, 'rb') as stream:
                data = stream.read(MAX_JSON_BYTES + 1)
            if len(data) > MAX_JSON_BYTES:
                raise ValueError('snapshot exceeds 4 MiB')
            snapshot = json.loads(data.decode('utf-8'))
        write_report(snapshot, args.out, overwrite=args.force)
        if args.command == 'demo' and args.json_out:
            atomic_write(args.json_out, json.dumps(snapshot, indent=2, ensure_ascii=True, allow_nan=False) + '\n', overwrite=args.force)
        # Keep status ASCII-only for Windows consoles; supplied paths need not be echoed.
        print('Report written. Open the HTML file locally; it makes no network requests.')
        return 0
    except (OSError, ValueError, TypeError, ImportError, RuntimeError, RecursionError) as exc:
        detail = str(exc).encode('ascii', 'backslashreplace').decode('ascii')
        print(f'stridelens: {detail}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
