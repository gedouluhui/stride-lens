"""Counterexamples confirmed by the first independent core review."""
import tracemalloc
import unittest
import warnings

import torch
from stridelens import capture, validate_snapshot


class ReviewRegressions(unittest.TestCase):
    def test_empty_huge_axes_do_not_materialize_product_pools(self):
        x = torch.empty(0, 100000)
        tracemalloc.start()
        try:
            s = capture({'x': x}, max_elements=1)
            _, capture_peak = tracemalloc.get_traced_memory()
            tracemalloc.reset_peak()
            validate_snapshot(s)
            _, validate_peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLess(capture_peak, 256000, 'empty capture allocated dimension-sized pools')
        self.assertLess(validate_peak, 256000, 'empty validation allocated dimension-sized pools')

    def test_same_region_intersecting_bounds_cannot_claim_disjoint(self):
        x = torch.ones(2)
        s = capture({'a': x, 'b': x}, max_elements=1)
        s['pairs'][0].update(overlap='no', overlap_bytes=0, basis='disjoint_bounds')
        with self.assertRaises(ValueError):
            validate_snapshot(s)

    def test_cross_region_overlap_cannot_exceed_unique_bytes(self):
        x, y = torch.ones(1).expand(4), torch.ones(4)
        s = capture({'a': x, 'b': y})
        s['pairs'][0].update(overlap='yes', overlap_bytes=8, basis='enumerated_bytes')
        with self.assertRaises(ValueError):
            validate_snapshot(s)

    def test_view_shape_must_match_explicit_request(self):
        s = capture({'x': torch.ones(2)}, view_requests={'x': [2]})
        s['views'][0].update(result_shape=[1, 2], result_strides=[2, 1])
        with self.assertRaises(ValueError):
            validate_snapshot(s)

    def test_view_strides_must_remain_inside_storage(self):
        s = capture({'x': torch.ones(2)}, view_requests={'x': [2]})
        s['views'][0]['result_strides'] = [100]
        with self.assertRaises(ValueError):
            validate_snapshot(s)

    def test_overflowing_view_product_is_rejected_before_torch(self):
        with self.assertRaises(ValueError):
            capture({'x': torch.empty(0)}, view_requests={'x': [2**32, 2**32]})

    def test_false_contiguity_rejected_but_singletons_and_empty_accepted(self):
        s = capture({'x': torch.ones(2, 3).t()})
        s['tensors'][0]['contiguous'] = True
        with self.assertRaises(ValueError):
            validate_snapshot(s)
        for x in [torch.empty(0, 5), torch.ones(1, 1, 3).transpose(0, 1), torch.tensor(1.)]:
            validate_snapshot(capture({'x': x}))

    def test_in_bounds_view_must_preserve_logical_address_sequence(self):
        x = torch.arange(4, dtype=torch.float32)[::2]
        s = capture({'x': x}, view_requests={'x': [2]})
        s['views'][0]['result_strides'] = [1]
        with self.assertRaises(ValueError):
            validate_snapshot(s)

    def test_shared_invalid_containers_rejected_before_serialization(self):
        nested = []
        for _ in range(20):
            nested = [nested, nested]
        s = capture({'x': torch.ones(1)})
        s['views'] = nested
        tracemalloc.start()
        try:
            with self.assertRaises(ValueError):
                validate_snapshot(s)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertLess(peak, 256000, 'invalid shared containers amplified during serialization')

    def test_named_view_error_does_not_invent_stride_cause(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            x = torch.ones(2).refine_names('N')
            s = capture({'x': x}, view_requests={'x': [2]})
        self.assertEqual(s['views'][0]['status'], 'rejected')
        self.assertNotIn('incompatible shape or strides', s['views'][0]['reason'])


if __name__ == '__main__':
    unittest.main()
