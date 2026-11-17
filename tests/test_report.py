import copy
from html.parser import HTMLParser
import json
from pathlib import Path
import tempfile
import unittest

try:
    from stridelens.schema import validate_snapshot
    from stridelens.report import render_html, write_report
except ModuleNotFoundError:
    validate_snapshot = render_html = write_report = None


def sample():
    return {'schema': 'stridelens.snapshot.v1',
            'producer': {'name': 'stridelens', 'version': '0.1.0', 'torch': '2.12.0+cpu'},
            'limits': {'max_elements': 4096, 'max_tensors': 32},
            'tensors': [{'name': 'x', 'shape': [2], 'strides_elements': [1],
                         'storage_offset_elements': 0, 'element_size': 4, 'storage_id': 'S1',
                         'storage_nbytes': 8, 'numel': 2, 'contiguous': True,
                         'enumeration': 'complete', 'offsets_bytes': [0, 4], 'unique_bytes': 8,
                         'internal_overlap': False, 'merge_barriers': []}], 'pairs': [], 'views': []}


class DataScript(HTMLParser):
    def __init__(self):
        super().__init__()
        self.inside = False
        self.parts = []
    def handle_starttag(self, tag, attrs):
        if tag == 'script' and dict(attrs).get('id') == 'snapshot-data':
            self.inside = True
    def handle_endtag(self, tag):
        if tag == 'script':
            self.inside = False
    def handle_data(self, data):
        if self.inside:
            self.parts.append(data)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(validate_snapshot, 'schema and report are not implemented')

    def test_accepts_hand_specified_snapshot(self):
        self.assertIsNone(validate_snapshot(sample()))

    def test_unknown_schema_and_unexpected_keys_rejected(self):
        for field, value in [('schema', 'future.v2'), ('raw_pointer', 123456)]:
            s = sample(); s[field] = value
            with self.assertRaises(ValueError):
                validate_snapshot(s)

    def test_bool_and_unsafe_integer_rejected(self):
        for field, value in [('numel', True), ('storage_nbytes', 2 ** 53), ('contiguous', 1)]:
            s = sample(); s['tensors'][0][field] = value
            with self.assertRaises(ValueError):
                validate_snapshot(s)

    def test_wrong_offset_count_and_arithmetic_rejected(self):
        for offsets in [[0], [0, 0], [0, 8], [False, 4]]:
            s = sample(); s['tensors'][0]['offsets_bytes'] = offsets
            with self.assertRaises(ValueError):
                validate_snapshot(s)

    def test_derived_metrics_cannot_lie(self):
        for field, value in [('unique_bytes', 4), ('internal_overlap', True), ('numel', 3)]:
            s = sample(); s['tensors'][0][field] = value
            with self.assertRaises(ValueError):
                validate_snapshot(s)

    def test_duplicate_name_and_missing_pairs_rejected(self):
        s = sample(); s['tensors'].append(copy.deepcopy(s['tensors'][0]))
        with self.assertRaises(ValueError):
            validate_snapshot(s)
        s['tensors'][1]['name'] = 'y'
        with self.assertRaises(ValueError):
            validate_snapshot(s)

    def test_orphan_pair_and_false_unknown_zero_rejected(self):
        s = sample()
        s['pairs'] = [{'a': 'x', 'b': 'not here', 'same_storage_region': True,
                       'overlap': 'unknown', 'overlap_bytes': 0, 'basis': 'budget_exceeded'}]
        with self.assertRaises(ValueError):
            validate_snapshot(s)
        s['tensors'].append(copy.deepcopy(s['tensors'][0])); s['tensors'][1]['name'] = 'y'
        s['pairs'][0]['b'] = 'y'
        with self.assertRaises(ValueError):
            validate_snapshot(s)

    def test_oversized_string_and_nonfinite_rejected(self):
        for value in ['x' * 129, float('nan')]:
            s = sample(); s['tensors'][0]['name'] = value
            with self.assertRaises(ValueError):
                validate_snapshot(s)

    def test_script_escaping_preserves_label_without_executing_it(self):
        s = sample(); s['tensors'][0]['name'] = '</script><script>alert(1)</script>&雪'
        page = render_html(s)
        self.assertNotIn('<script>alert(1)</script>', page)
        parser = DataScript(); parser.feed(page)
        self.assertEqual(json.loads(''.join(parser.parts)), s)

    def test_atomic_non_overwrite_preserves_original(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'report.html'
            p.write_bytes(b'existing')
            with self.assertRaises(FileExistsError):
                write_report(sample(), p)
            self.assertEqual(p.read_bytes(), b'existing')
            self.assertEqual(list(Path(d).iterdir()), [p])
            write_report(sample(), p, overwrite=True)
            self.assertTrue(p.read_text(encoding='utf-8').startswith('<!doctype html>'))

    def test_render_contains_usable_index_and_tensor_controls(self):
        page = render_html(sample())
        parser = DataScript(); parser.feed(page)
        self.assertEqual(json.loads(''.join(parser.parts))['tensors'][0]['name'], 'x')
        self.assertIn('id="tensor-select"', page)
        self.assertIn('id="index-form"', page)
        self.assertIn('id="logical-grid"', page)


if __name__ == '__main__':
    unittest.main()
