"""Check actual source-archive contents without extracting or executing them."""
import argparse
from pathlib import Path, PurePosixPath
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ('LICENSE', 'README.md', 'pyproject.toml', 'MANIFEST.in',
            'scripts/verify.py', 'scripts/check_sdist.py',
            'docs/demo.html', 'docs/demo.json')


def check(archive):
    expected = {name: (ROOT / name).read_bytes() for name in REQUIRED if (ROOT / name).is_file()}
    # Missing source files must not silently shrink the archive contract.
    errors = [f'source missing: {name}' for name in REQUIRED if name not in expected]
    for folder in ('src/stridelens', 'tests'):
        for path in (ROOT / folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.py', '.html'):
                expected[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    found, prefix = {}, None
    with tarfile.open(archive, 'r:gz') as package:
        for member in package.getmembers():
            parts = PurePosixPath(member.name).parts
            if not parts or '..' in parts or member.name.startswith('/'):
                errors.append('unsafe archive path')
                continue
            if prefix is None:
                prefix = parts[0]
            if parts[0] != prefix:
                errors.append('multiple archive roots')
            if len(parts) == 1 and member.isdir():
                continue
            name = '/'.join(parts[1:])
            if name.startswith(('artifacts/', '.git/', '.venv/', 'docs/superpowers/')) or 'HANDOVER' in name:
                errors.append(f'private/internal path: {name}')
            if not member.isdir() and not member.isfile():
                errors.append(f'unsupported archive member: {name}')
            if member.isfile():
                if name in found:
                    errors.append(f'duplicate archive file: {name}')
                found[name] = member
        for name, content in expected.items():
            member = found.get(name)
            if member is None:
                errors.append(f'archive missing: {name}')
            elif member.size != len(content) or package.extractfile(member).read() != content:
                errors.append(f'archive/source bytes differ: {name}')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    try:
        errors = check(args.archive)
    except (OSError, tarfile.TarError, ValueError) as exc:
        print(f'sdist check could not run: {type(exc).__name__}', file=sys.stderr)
        return 2
    for error in errors:
        print(error, file=sys.stderr)
    print('sdist content/byte contract: ' + ('FAIL' if errors else 'PASS'))
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
