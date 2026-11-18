"""Read-only tensor layout evidence, with no torch import until capture()."""
from .capture import capture
from .report import render_html, write_report
from .schema import validate_snapshot

__version__ = '0.1.1'
__all__ = ['capture', 'render_html', 'write_report', 'validate_snapshot']
