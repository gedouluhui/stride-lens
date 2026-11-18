"""Budget-independent mapping checks, with a separate index-enumeration oracle."""
from itertools import product
import unittest

import torch
from stridelens import capture, validate_snapshot
try:
    from stridelens.layout import offset_signature
except ModuleNotFoundError:
    offset_signature = None


class QualityR3(unittest.TestCase):
    def test_budget_does_not_hide_wrong_view_addresses(self):
        for budget in (1, 2):
            s = capture({'x': torch.arange(4, dtype=torch.float32)[::2]},
                        max_elements=budget, view_requests={'x': [2]})
            s['views'][0]['result_strides'] = [1]
            with self.assertRaises(ValueError):
                validate_snapshot(s)

    def test_duplicate_address_sets_cannot_disagree_about_third_tensor(self):
        backing = bytearray(16)
        x = torch.frombuffer(backing, dtype=torch.float32, count=4)
        c = torch.frombuffer(backing, dtype=torch.float32, count=3, offset=1)
        for y in [x, x.reshape(2, 2).t()]:
            s = capture({'a': x, 'b': y, 'c': c})
            for pair in s['pairs']:
                if pair['a'] == 'b' and pair['b'] == 'c':
                    pair.update(overlap='no', overlap_bytes=0, basis='disjoint_bounds')
            with self.assertRaises(ValueError):
                validate_snapshot(s)

    def test_ambiguous_empty_inference_cannot_be_accepted_record(self):
        s = capture({'x': torch.empty(0)}, view_requests={'x': [0, 1]})
        s['views'][0]['requested_shape'] = [0, -1]
        with self.assertRaises(ValueError):
            validate_snapshot(s)

    def test_valid_view_records_across_low_budgets(self):
        examples = [(torch.arange(24, dtype=torch.float32).reshape(2, 3, 4), [6, 4]),
                    (torch.ones(1, 4).expand(3, 4), [3, 2, 2]),
                    (torch.arange(12, dtype=torch.float32)[::2], [2, 3]),
                    (torch.empty(0, 4), [0, 2, 2]), (torch.tensor(1.), [])]
        for x, shape in examples:
            for budget in (1, 2, 4096):
                with self.subTest(shape=shape, budget=budget):
                    s = capture({'x': x}, max_elements=budget, view_requests={'x': shape})
                    self.assertEqual(s['views'][0]['status'], 'accepted')
                    validate_snapshot(s)

    def test_signature_matches_independent_small_sequence_equivalence(self):
        self.assertIsNotNone(offset_signature, 'symbolic mapping is not implemented')
        sequence_to_signature, signature_to_sequence = {}, {}
        for rank in range(4):
            for shape in product((0, 1, 2, 3), repeat=rank):
                for strides in product((0, 1, 2, 3), repeat=rank):
                    for offset in (0, 2):
                        seq = tuple(offset + sum(i*s for i,s in zip(index, strides))
                                    for index in product(*(range(n) for n in shape)))
                        sig = offset_signature(shape, strides, offset)
                        self.assertEqual(sequence_to_signature.setdefault(seq, sig), sig,
                                         (shape, strides, offset, 'same sequence, unequal signature'))
                        self.assertEqual(signature_to_sequence.setdefault(sig, seq), seq,
                                         (shape, strides, offset, 'unequal sequences, same signature'))

    def test_sequence_signature_does_not_pretend_to_identify_sets(self):
        self.assertIsNotNone(offset_signature, 'symbolic mapping is not implemented')
        self.assertNotEqual(offset_signature((2, 2), (2, 1), 0),
                            offset_signature((2, 2), (1, 2), 0))


if __name__ == '__main__':
    unittest.main()
