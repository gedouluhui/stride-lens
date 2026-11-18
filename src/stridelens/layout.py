"""Exact flattened-address signatures for validated nonnegative strided layouts.

This identifies ordered address SEQUENCES, not unordered addressed sets. It is
independent of the number of elements and does not allocate an element list.
"""
from collections import defaultdict
import math


def offset_signature(shape, strides, offset):
    """Canonical floor-basis form of offsets for 0 <= flat_index < numel.

    Axis i with suffix product q contributes
    s_i * floor(k/q) - n_i*s_i * floor(k/(q*n_i)).
    Terms with denominator >= N vanish on the domain. Equal denominators combine.
    Remaining floor(k/q) functions are independent: at the smallest differing q,
    evaluating k=q distinguishes the two expressions. This also removes singleton
    axes and handles zero strides. Empty domains all have the empty signature.
    Inputs have already passed schema validation; this helper is not a parser.
    """
    n = math.prod(shape)
    if n == 0:
        return (0,)
    terms = defaultdict(int)
    q = 1
    for size, stride in reversed(list(zip(shape, strides))):
        if q < n:
            terms[q] += stride
        q *= size
        if q < n:
            terms[q] -= size * stride
    return (n, offset, tuple(sorted((q, c) for q, c in terms.items() if c)))


def address_set_groups(tensors):
    """Proven equivalence only: exact enumerated sets or equal address sequences."""
    parent = {t['name']: t['name'] for t in tensors}

    def find(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    seen = {}
    for tensor in tensors:
        name, region = tensor['name'], tensor['storage_id']
        keys = [('sequence', region, offset_signature(tensor['shape'], tensor['strides_elements'],
                                                      tensor['storage_offset_elements']))]
        if tensor['offsets_bytes'] is not None:
            keys.append(('set', region, frozenset(tensor['offsets_bytes'])))
        for key in keys:
            if key in seen:
                parent[find(name)] = find(seen[key])
            else:
                seen[key] = name
    return {name: find(name) for name in parent}
