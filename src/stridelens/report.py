"""Self-contained report rendering and atomic, non-overwriting output."""
from importlib.resources import files
import json
import os
from pathlib import Path
import tempfile

from .schema import validate_snapshot


def atomic_write(path, text: str, *, overwrite: bool = False) -> None:
    destination = Path(path)
    if not overwrite and destination.exists():
        raise FileExistsError('output already exists; use --force to replace it')
    fd, temporary = tempfile.mkstemp(prefix='.stridelens-', dir=destination.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            # Atomic no-clobber publication; no check-then-replace race.
            os.link(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def render_html(snapshot: dict) -> str:
    validate_snapshot(snapshot)
    data = json.dumps(snapshot, ensure_ascii=True, allow_nan=False, separators=(',', ':'))
    data = data.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    template = files('stridelens').joinpath('assets/report.html').read_text(encoding='utf-8')
    if template.count('@@SNAPSHOT@@') != 1:
        raise RuntimeError('invalid report template')
    return template.replace('@@SNAPSHOT@@', data)


def write_report(snapshot: dict, path, *, overwrite: bool = False) -> None:
    atomic_write(path, render_html(snapshot), overwrite=overwrite)
