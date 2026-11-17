"""Synthetic tensors only; no model, network or GPU required."""
from .capture import capture


def demo_snapshot() -> dict:
    import torch
    base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    tensors = {'base': base, 'transpose': base.t(), 'even columns': base[:, ::2],
               'odd columns': base[:, 1::2], 'reshape copy': base.t().reshape(-1),
               'broadcast': base[:1, :].expand(3, 4), 'empty': base[:0]}
    return capture(tensors, view_requests={'transpose': [-1], 'even columns': [-1]})
