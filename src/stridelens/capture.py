"""Read-only CPU tensor layout capture. Process addresses never enter the snapshot."""
from collections.abc import Mapping, Sequence
from itertools import combinations, product
import math

SAFE_INT = (1 << 53) - 1


def _natural(value, label):
    if type(value) is not int or not 0 <= value <= SAFE_INT:
        raise ValueError(f'{label} must be a nonnegative safe integer')
    return value


def _request_shape(value):
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) > 16:
        raise ValueError('view shape must be a sequence with at most 16 axes')
    shape = list(value)
    if any(type(x) is not int or not -1 <= x <= SAFE_INT for x in shape) or shape.count(-1) > 1:
        raise ValueError('view shape needs nonnegative integers and at most one -1')
    _natural(math.prod(x for x in shape if x != -1), 'requested shape product')
    return shape


def _barriers(shape, strides):
    axes = [i for i, size in enumerate(shape) if size > 1]
    return [{'left_axis': a, 'right_axis': b, 'left_stride': strides[a],
             'required_stride': strides[b] * shape[b]}
            for a, b in zip(axes, axes[1:]) if strides[a] != strides[b] * shape[b]]


def capture(tensors: Mapping[str, object], *, max_elements: int = 4096,
            view_requests: Mapping[str, Sequence[int]] | None = None) -> dict:
    """Capture one live set of tensors; all pairwise overlap decisions are byte-based.

    Unsupported inputs fail before returning any snapshot. Large overlapping bounds
    are UNKNOWN unless every addressed byte was enumerated. No tensor data is read.
    """
    if type(max_elements) is not int or not 1 <= max_elements <= 4096:
        raise ValueError('max_elements must be an integer in 1..4096')
    if not isinstance(tensors, Mapping) or not 1 <= len(tensors) <= 32:
        raise ValueError('supply a mapping of 1..32 named tensors')
    items = list(tensors.items())  # Retain tensor references until pair analysis ends.
    names = [name for name, _ in items]
    if any(type(name) is not str or not name or len(name) > 128 for name in names):
        raise ValueError('tensor names must be nonempty strings of at most 128 characters')
    if len(set(names)) != len(names):
        raise ValueError('tensor names must be unique')
    requests = {} if view_requests is None else view_requests
    if not isinstance(requests, Mapping) or any(name not in names for name in requests):
        raise ValueError('view requests must reference captured tensors')
    requests = {name: _request_shape(shape) for name, shape in requests.items()}
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('capture requires PyTorch; rendering a saved snapshot does not') from exc
    from . import __version__

    regions, rows, internals, views = {}, [], [], []
    for name, tensor in items:
        if type(tensor) is not torch.Tensor:
            raise TypeError('only exact torch.Tensor is supported; explicitly detach Parameters')
        if (tensor.device.type != 'cpu' or tensor.layout != torch.strided
                or tensor.dtype != torch.float32 or tensor.is_conj() or tensor.is_neg()):
            raise ValueError('only ordinary CPU strided float32 tensors are supported')
        shape, strides = list(tensor.shape), list(tensor.stride())
        if len(shape) > 16:
            raise ValueError('at most 16 axes are supported')
        for value in shape + strides:
            _natural(value, 'shape/stride')
        offset = _natural(tensor.storage_offset(), 'storage offset')
        numel = _natural(math.prod(shape), 'element count')
        storage = tensor.untyped_storage()
        nbytes = _natural(storage.nbytes(), 'storage bytes')
        pointer = storage.data_ptr()
        region = (pointer, nbytes) if nbytes else ('empty', name)
        sid = regions.setdefault(region, f'S{len(regions) + 1}')
        lo = _natural(offset * 4, 'byte offset')
        hi = lo if numel == 0 else _natural(
            (offset + sum((n - 1) * s for n, s in zip(shape, strides))) * 4 + 4,
            'byte extent')
        if numel and hi > nbytes:
            raise ValueError('tensor extent lies outside its reported storage')
        if numel <= max_elements:
            offsets = [(offset + sum(i * s for i, s in zip(index, strides))) * 4
                       for index in (product(*(range(n) for n in shape)) if numel else ())]
            byte_set = {pointer + start + byte for start in offsets for byte in range(4)}
            unique = len(byte_set)
        else:
            offsets, byte_set, unique = None, None, None
        barriers = _barriers(shape, strides)
        for barrier in barriers:
            _natural(barrier['required_stride'], 'merge stride')
        rows.append({
            'name': name, 'shape': shape, 'strides_elements': strides,
            'storage_offset_elements': offset, 'element_size': 4,
            'storage_id': sid, 'storage_nbytes': nbytes, 'numel': numel,
            'contiguous': bool(tensor.is_contiguous()),
            'enumeration': 'complete' if offsets is not None else 'budget_exceeded',
            'offsets_bytes': offsets, 'unique_bytes': unique,
            'internal_overlap': unique < numel * 4 if unique is not None else None,
            'merge_barriers': barriers,
        })
        internals.append((pointer + lo, pointer + hi, byte_set, numel))
        if name in requests:
            req = requests[name]
            try:
                result = tensor.view(tuple(req))
            except RuntimeError:
                views.append({'tensor': name, 'requested_shape': req, 'status': 'rejected',
                              'result_shape': None, 'result_strides': None,
                              'reason': 'PyTorch rejected this view. No output tensor was created.'})
            else:
                rs, rt = list(result.shape), list(result.stride())
                for value in rs + rt:
                    _natural(value, 'view shape/stride')
                if math.prod(rs) != numel:
                    raise ValueError('view result has inconsistent element-count arithmetic')
                views.append({'tensor': name, 'requested_shape': req, 'status': 'accepted',
                              'result_shape': rs, 'result_strides': rt,
                              'reason': 'PyTorch accepted this view without copying storage.'})
    pairs = []
    for a, b in combinations(range(len(rows)), 2):
        alo, ahi, aset, an = internals[a]
        blo, bhi, bset, bn = internals[b]
        if not an or not bn:
            count, basis = 0, 'empty'
        elif ahi <= blo or bhi <= alo:
            count, basis = 0, 'disjoint_bounds'
        elif aset is not None and bset is not None:
            count, basis = len(aset & bset), 'enumerated_bytes'
        else:
            count, basis = None, 'budget_exceeded'
        pairs.append({'a': rows[a]['name'], 'b': rows[b]['name'],
                      'same_storage_region': rows[a]['storage_id'] == rows[b]['storage_id'],
                      'overlap': 'unknown' if count is None else ('yes' if count else 'no'),
                      'overlap_bytes': count, 'basis': basis})
    return {'schema': 'stridelens.snapshot.v1',
            'producer': {'name': 'stridelens', 'version': __version__, 'torch': str(torch.__version__)},
            'limits': {'max_elements': max_elements, 'max_tensors': 32},
            'tensors': rows, 'pairs': pairs, 'views': views}
