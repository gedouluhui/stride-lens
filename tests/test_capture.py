"""Behavior tests: each expected address is literal or read from real scalar tensors."""
import itertools
import json
import unittest

import torch

try:
    from stridelens.capture import capture
except ModuleNotFoundError:
    capture = None


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(capture, 'capture API has not been implemented')
        self.x = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    def test_disjoint_slices_do_not_alias_despite_same_storage(self):
        s = capture({'even': self.x[:, ::2], 'odd': self.x[:, 1::2]})
        self.assertEqual(s['tensors'][0]['offsets_bytes'], [0, 8, 16, 24, 32, 40])
        self.assertEqual(s['tensors'][1]['offsets_bytes'], [4, 12, 20, 28, 36, 44])
        self.assertTrue(s['pairs'][0]['same_storage_region'])
        self.assertEqual(s['pairs'][0]['overlap'], 'no')
        self.assertEqual(s['pairs'][0]['overlap_bytes'], 0)

    def test_offsets_match_actual_scalar_addresses(self):
        for t in [self.x, self.x.t(), self.x[1:, 1::2], self.x[:1].expand(3, 4)]:
            with self.subTest(shape=tuple(t.shape), stride=t.stride()):
                actual = capture({'x': t})['tensors'][0]['offsets_bytes']
                base = t.untyped_storage().data_ptr()
                expected = [t[index].data_ptr() - base for index in
                            itertools.product(*(range(n) for n in t.shape))]
                self.assertEqual(actual, expected)

    def test_transpose_and_copy_have_distinct_relationships(self):
        s = capture({'base': self.x, 'transpose': self.x.t(), 'copy': self.x.t().reshape(-1)})
        p = {(p['a'], p['b']): p for p in s['pairs']}
        self.assertEqual(p['base', 'transpose']['overlap_bytes'], 48)
        self.assertEqual(p['base', 'copy']['overlap'], 'no')
        self.assertFalse(p['base', 'copy']['same_storage_region'])

    def test_broadcast_internal_overlap_is_counted_in_bytes(self):
        t = capture({'x': self.x[:1].expand(3, 4)})['tensors'][0]
        self.assertTrue(t['internal_overlap'])
        self.assertEqual(t['unique_bytes'], 16)
        self.assertEqual(t['offsets_bytes'], [0, 4, 8, 12] * 3)

    def test_empty_and_scalar_do_not_collapse_to_same_thing(self):
        s = capture({'a': torch.empty(0), 'b': torch.empty(0), 'scalar': torch.tensor(9.)})
        a, b, scalar = s['tensors']
        self.assertNotEqual(a['storage_id'], b['storage_id'])
        self.assertEqual(a['offsets_bytes'], [])
        self.assertFalse(a['internal_overlap'])
        self.assertEqual(scalar['shape'], [])
        self.assertEqual(scalar['offsets_bytes'], [0])
        self.assertTrue(all(p['overlap'] == 'no' for p in s['pairs']))

    def test_budget_is_unknown_not_no(self):
        s = capture({'base': self.x, 'transpose': self.x.t()}, max_elements=2)
        self.assertIsNone(s['tensors'][0]['offsets_bytes'])
        self.assertIsNone(s['tensors'][0]['internal_overlap'])
        self.assertEqual(s['pairs'][0]['overlap'], 'unknown')
        self.assertIsNone(s['pairs'][0]['overlap_bytes'])
        s = capture({'base': self.x, 'copy': self.x.clone()}, max_elements=2)
        self.assertEqual(s['pairs'][0]['overlap'], 'no')

    def test_separate_unaligned_storage_wrappers_still_overlap(self):
        backing = bytearray(17)
        a = torch.frombuffer(backing, dtype=torch.float32, count=3, offset=0)
        b = torch.frombuffer(backing, dtype=torch.float32, count=3, offset=1)
        p = capture({'a': a, 'b': b})['pairs'][0]
        self.assertFalse(p['same_storage_region'])
        self.assertEqual(p['overlap'], 'yes')
        self.assertEqual(p['overlap_bytes'], 11)

    def test_actual_view_accepts_noncontiguous_compatible_slice(self):
        s = capture({'x': self.x[:, ::2]}, view_requests={'x': [-1]})
        v = s['views'][0]
        self.assertFalse(s['tensors'][0]['contiguous'])
        self.assertEqual(v['status'], 'accepted')
        self.assertEqual(v['result_shape'], [6])
        self.assertEqual(v['result_strides'], [2])

    def test_actual_view_rejects_flattened_transpose(self):
        s = capture({'x': self.x.t()}, view_requests={'x': [-1]})
        self.assertEqual(s['views'][0]['status'], 'rejected')
        self.assertIsNone(s['views'][0]['result_shape'])
        self.assertEqual(s['tensors'][0]['merge_barriers'],
                         [{'left_axis': 0, 'right_axis': 1, 'left_stride': 1, 'required_stride': 12}])

    def test_capture_does_not_modify_user_tensor(self):
        before = self.x.clone()
        version = self.x._version
        capture({'x': self.x}, view_requests={'x': [2, 6]})
        self.assertTrue(torch.equal(self.x, before))
        self.assertEqual(self.x._version, version)

    def test_snapshot_does_not_disclose_values_or_process_address(self):
        x = torch.full((2, 2), 8675309., dtype=torch.float32)
        text = json.dumps(capture({'private': x}))
        self.assertNotIn(str(x.untyped_storage().data_ptr()), text)
        self.assertNotIn('8675309', text)
        self.assertNotIn('data_ptr', text)
        self.assertNotIn('C:/Users', text)

    def test_unsupported_tensor_kinds_rejected(self):
        cases = [self.x.to(torch.float64), self.x.to(torch.int32),
                 torch.nn.Parameter(self.x), self.x.to_sparse(),
                 torch.empty((2, 2), device='meta'), [1., 2.]]
        for t in cases:
            with self.subTest(kind=type(t)):
                with self.assertRaises((TypeError, ValueError)):
                    capture({'x': t})

    def test_invalid_limits_names_and_requests_rejected(self):
        for limit in [True, 0, -1, 4097, 1.5]:
            with self.subTest(limit=limit), self.assertRaises((TypeError, ValueError)):
                capture({'x': self.x}, max_elements=limit)
        for inputs in [{}, {'': self.x}, {'x' * 129: self.x}, {1: self.x},
                       {str(i): self.x for i in range(33)}]:
            with self.assertRaises((TypeError, ValueError)):
                capture(inputs)
        for request in [{'absent': [2]}, {'x': [True, 12]}, {'x': [-1, -1]},
                        {'x': [-2]}, {'x': '12'}]:
            with self.assertRaises((TypeError, ValueError)):
                capture({'x': self.x}, view_requests=request)

    def test_zero_stride_huge_shape_is_bounded(self):
        x = torch.ones(1).expand(1000000)
        s = capture({'x': x}, max_elements=4)
        self.assertEqual(s['tensors'][0]['numel'], 1000000)
        self.assertEqual(s['tensors'][0]['enumeration'], 'budget_exceeded')

    def test_deterministic_report_ignores_process_addresses(self):
        a = capture({'x': self.x, 'v': self.x.t()})
        b = capture({'x': self.x.clone(), 'v': self.x.clone().t()})
        self.assertNotEqual(a['pairs'][0]['overlap'], b['pairs'][0]['overlap'])
        c = self.x.clone()
        self.assertEqual(a, capture({'x': c, 'v': c.t()}))


if __name__ == '__main__':
    unittest.main()
