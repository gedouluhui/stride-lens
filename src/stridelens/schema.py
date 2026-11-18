"""Bounded structural and arithmetic validation; does not import a tensor runtime."""
from itertools import combinations, product
import json
import math
import re
from .layout import address_set_groups, offset_signature

MAX_JSON_BYTES = 4 * 1024 * 1024
SAFE_INT = (1 << 53) - 1


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _keys(value, names, label):
    _require(type(value) is dict and set(value) == set(names.split()), f'invalid {label} fields')


def _integer(value, label, low=0, high=SAFE_INT):
    _require(type(value) is int and low <= value <= high, f'invalid {label}')


def _string(value, label, maximum=128):
    _require(type(value) is str and 0 < len(value) <= maximum, f'invalid {label}')


def _vector(value, label, low=0):
    _require(type(value) is list and len(value) <= 16, f'invalid {label}')
    for x in value:
        _integer(x, label, low)


def _bounds(t):
    start = t['storage_offset_elements'] * 4
    end = start if not t['numel'] else start + sum((n-1)*s for n,s in zip(t['shape'], t['strides_elements'])) * 4 + 4
    return start, end


def _contiguous(shape, strides):
    if 0 in shape:
        return True
    expected = 1
    for n, stride in reversed(list(zip(shape, strides))):
        if n != 1:
            if stride != expected:
                return False
            expected *= n
    return True


def _tensor(t, limit):
    _keys(t, 'name shape strides_elements storage_offset_elements element_size storage_id '
          'storage_nbytes numel contiguous enumeration offsets_bytes unique_bytes internal_overlap merge_barriers', 'tensor')
    _string(t['name'], 'tensor name')
    _vector(t['shape'], 'shape'); _vector(t['strides_elements'], 'strides')
    _require(len(t['shape']) == len(t['strides_elements']), 'shape/stride rank mismatch')
    for key in ('storage_offset_elements', 'storage_nbytes', 'numel', 'element_size'):
        _integer(t[key], key)
    _require(t['element_size'] == 4, 'only float32 byte layouts are supported')
    _require(t['numel'] == math.prod(t['shape']), 'element count mismatch')
    _require(type(t['contiguous']) is bool, 'contiguous must be a boolean')
    _require(t['contiguous'] == _contiguous(t['shape'], t['strides_elements']), 'contiguity contradicts shape/strides')
    _require(type(t['storage_id']) is str and re.fullmatch(r'S(?:[1-9]|[12][0-9]|3[0-2])', t['storage_id']), 'invalid storage id')
    _integer(t['storage_offset_elements'] * 4, 'byte offset')
    if t['numel']:
        end = (t['storage_offset_elements'] + sum((n-1)*s for n,s in zip(t['shape'], t['strides_elements']))) * 4 + 4
        _integer(end, 'byte extent')
        _require(end <= t['storage_nbytes'], 'layout outside storage')
    _require(type(t['enumeration']) is str, 'enumeration must be a string')
    if t['numel'] <= limit:
        _require(t['enumeration'] == 'complete', 'small layout must be enumerated')
        offsets = t['offsets_bytes']
        _require(type(offsets) is list and len(offsets) == t['numel'], 'offset count mismatch')
        for x in offsets:
            _integer(x, 'offset')
        expected = [(t['storage_offset_elements'] + sum(i*s for i,s in zip(index, t['strides_elements']))) * 4
                    for index in (product(*(range(n) for n in t['shape'])) if t['numel'] else ())]
        _require(offsets == expected, 'offset arithmetic mismatch')
        unique = len(set(offsets)) * 4
        _integer(t['unique_bytes'], 'unique bytes')
        _require(t['unique_bytes'] == unique, 'unique byte count mismatch')
        _require(type(t['internal_overlap']) is bool and t['internal_overlap'] == (unique < t['numel'] * 4), 'internal overlap mismatch')
    else:
        _require(t['enumeration'] == 'budget_exceeded', 'large layout must report its budget')
        _require(all(t[k] is None for k in ('offsets_bytes', 'unique_bytes', 'internal_overlap')), 'unknown layout must not report exact metrics')
    axes = [i for i,n in enumerate(t['shape']) if n > 1]
    expected_barriers = [{'left_axis': a, 'right_axis': b, 'left_stride': t['strides_elements'][a],
                          'required_stride': t['strides_elements'][b] * t['shape'][b]}
                         for a,b in zip(axes, axes[1:])
                         if t['strides_elements'][a] != t['strides_elements'][b] * t['shape'][b]]
    _require(type(t['merge_barriers']) is list and len(t['merge_barriers']) <= 15, 'invalid merge barriers')
    for b in t['merge_barriers']:
        _keys(b, 'left_axis right_axis left_stride required_stride', 'barrier')
        for key in b:
            _integer(b[key], 'barrier integer')
    _require(t['merge_barriers'] == expected_barriers, 'merge barrier mismatch')


def validate_snapshot(snapshot: dict) -> None:
    """Reject malformed reports. Validation is not provenance authentication."""
    _keys(snapshot, 'schema producer limits tensors pairs views', 'snapshot')
    # Reject invalid shapes/types before serialization: a tiny shared-container
    # graph can otherwise expand exponentially into a JSON string.
    _require(type(snapshot['schema']) is str and snapshot['schema'] == 'stridelens.snapshot.v1', 'unsupported snapshot schema')
    _keys(snapshot['producer'], 'name version torch', 'producer')
    for k,v in snapshot['producer'].items():
        _string(v, f'producer {k}')
    _require(snapshot['producer']['name'] == 'stridelens', 'unexpected producer')
    _keys(snapshot['limits'], 'max_elements max_tensors', 'limits')
    limit = snapshot['limits']['max_elements']
    _integer(limit, 'max_elements', 1, 4096)
    _require(type(snapshot['limits']['max_tensors']) is int and snapshot['limits']['max_tensors'] == 32, 'invalid tensor cap')
    tensors = snapshot['tensors']
    _require(type(tensors) is list and 1 <= len(tensors) <= 32, 'invalid tensor list')
    by_name, regions = {}, {}
    for t in tensors:
        _tensor(t, limit)
        _require(t['name'] not in by_name, 'duplicate tensor name')
        by_name[t['name']] = t
        sid = t['storage_id']
        _require(sid not in regions or regions[sid] == t['storage_nbytes'], 'inconsistent storage region size')
        regions[sid] = t['storage_nbytes']
    pairs = snapshot['pairs']
    expected_pairs = {frozenset((a,b)) for a,b in combinations(by_name, 2)}
    _require(type(pairs) is list and len(pairs) == len(expected_pairs), 'pair list incomplete')
    seen = set()
    for p in pairs:
        _keys(p, 'a b same_storage_region overlap overlap_bytes basis', 'pair')
        _require(type(p['a']) is str and type(p['b']) is str, 'pair names must be strings')
        key = frozenset((p['a'], p['b']))
        _require(key in expected_pairs and key not in seen, 'duplicate or orphan pair')
        seen.add(key)
        a, b = by_name[p['a']], by_name[p['b']]
        _require(type(p['same_storage_region']) is bool and p['same_storage_region'] == (a['storage_id'] == b['storage_id']), 'storage relation mismatch')
        _require(type(p['overlap']) is str and p['overlap'] in ('yes', 'no', 'unknown'), 'invalid overlap status')
        _require(type(p['basis']) is str and p['basis'] in ('empty', 'enumerated_bytes', 'disjoint_bounds', 'budget_exceeded'), 'invalid overlap basis')
        count = p['overlap_bytes']
        if p['overlap'] == 'unknown':
            _require(count is None and p['basis'] == 'budget_exceeded', 'unknown must not contain exact overlap')
            _require(a['numel'] and b['numel'] and (a['offsets_bytes'] is None or b['offsets_bytes'] is None), 'unknown without an exceeded budget')
        else:
            _integer(count, 'overlap bytes')
            _require((count > 0) == (p['overlap'] == 'yes'), 'overlap count/status mismatch')
            caps = [t['unique_bytes'] if t['unique_bytes'] is not None else t['numel'] * 4 for t in (a, b)]
            _require(count <= min(caps), 'overlap count exceeds addressed bytes')
            _require(p['basis'] != 'budget_exceeded', 'exact result cannot use unknown basis')
            if p['basis'] in ('empty', 'disjoint_bounds'):
                _require(count == 0, 'empty/disjoint pair must have zero overlap')
            if p['basis'] == 'empty':
                _require(not a['numel'] or not b['numel'], 'empty basis for nonempty tensors')
            if p['basis'] == 'disjoint_bounds' and p['same_storage_region'] and a['numel'] and b['numel']:
                alo, ahi = _bounds(a)
                blo, bhi = _bounds(b)
                _require(ahi <= blo or bhi <= alo, 'claimed disjoint bounds intersect')
            if p['basis'] == 'enumerated_bytes':
                _require(a['offsets_bytes'] is not None and b['offsets_bytes'] is not None, 'enumeration missing')
            if p['same_storage_region'] and a['offsets_bytes'] is not None and b['offsets_bytes'] is not None:
                exact = len(set(a['offsets_bytes']) & set(b['offsets_bytes'])) * 4
                _require(count == exact, 'same-region overlap arithmetic mismatch')
    groups = address_set_groups(tensors)
    exact_relations = {}
    for pair in pairs:
        if pair['overlap'] == 'unknown':
            continue
        key = tuple(sorted((groups[pair['a']], groups[pair['b']])))
        evidence = (pair['overlap'], pair['overlap_bytes'])
        _require(exact_relations.setdefault(key, evidence) == evidence,
                 'equivalent address sets have contradictory pair evidence')
    views = snapshot['views']
    _require(type(views) is list and len(views) <= len(tensors), 'invalid views')
    requested = set()
    for v in views:
        _keys(v, 'tensor requested_shape status result_shape result_strides reason', 'view')
        _require(type(v['tensor']) is str and v['tensor'] in by_name and v['tensor'] not in requested, 'invalid view tensor')
        requested.add(v['tensor'])
        _vector(v['requested_shape'], 'view request', -1)
        _require(v['requested_shape'].count(-1) <= 1, 'multiple inferred dimensions')
        _integer(math.prod(n for n in v['requested_shape'] if n != -1), 'requested shape product')
        _string(v['reason'], 'view reason', 256)
        _require(type(v['status']) is str and v['status'] in ('accepted', 'rejected'), 'invalid view status')
        if v['status'] == 'accepted':
            _require(not (-1 in v['requested_shape'] and
                          math.prod(n for n in v['requested_shape'] if n != -1) == 0),
                     'accepted view has ambiguous empty-shape inference')
            _vector(v['result_shape'], 'view result'); _vector(v['result_strides'], 'view strides')
            _require(len(v['result_shape']) == len(v['result_strides']), 'view rank mismatch')
            source = by_name[v['tensor']]
            _require(math.prod(v['result_shape']) == source['numel'], 'view element mismatch')
            _require(len(v['requested_shape']) == len(v['result_shape']), 'view/request rank mismatch')
            _require(all(want == -1 or want == got for want,got in zip(v['requested_shape'], v['result_shape'])), 'view result differs from request')
            if source['numel']:
                end = (source['storage_offset_elements'] + sum((n-1)*s for n,s in zip(v['result_shape'], v['result_strides']))) * 4 + 4
                _integer(end, 'view byte extent')
                _require(end <= source['storage_nbytes'], 'view result outside storage')
            _require(offset_signature(source['shape'], source['strides_elements'], source['storage_offset_elements']) ==
                     offset_signature(v['result_shape'], v['result_strides'], source['storage_offset_elements']),
                     'view changes logical byte addresses')
            if source['offsets_bytes'] is not None:
                result_offsets = [(source['storage_offset_elements'] + sum(i*s for i,s in zip(index, v['result_strides']))) * 4
                                  for index in (product(*(range(n) for n in v['result_shape'])) if source['numel'] else ())]
                _require(result_offsets == source['offsets_bytes'], 'view changes logical byte addresses')
        else:
            _require(v['result_shape'] is None and v['result_strides'] is None, 'rejected view cannot contain results')
    # The entire structure is now bounded and made of exact JSON base types.
    size = len(json.dumps(snapshot, ensure_ascii=True, allow_nan=False).encode('utf-8'))
    _require(size <= MAX_JSON_BYTES, 'snapshot exceeds 4 MiB')
