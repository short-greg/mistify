
import torch
from .._base import CompositionBase, maxmin, ComplementBase,get_comp_weight_size
from torch import nn


def rand(*size: int, dtype=torch.float32, device='cpu'):

    return (torch.rand(*size, device=device, dtype=dtype)).round()


def negatives(*size: int, dtype=torch.float32, device='cpu'):

    return torch.zeros(*size, dtype=dtype, device=device)


def positives(*size: int, dtype=torch.float32, device='cpu'):

    return torch.ones(*size, dtype=dtype, device=device)


def differ(m1: torch.Tensor, m2: torch.Tensor) -> 'torch.Tensor':
    return (m1 - m2).clamp(0.0, 1.0)


def unify(m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
    return torch.max(m1, m2)


def intersect(m1: torch.Tensor, m2: torch.Tensor) -> 'torch.Tensor':
    return torch.min(m1, m2)


def inclusion(m1: torch.Tensor, m2: torch.Tensor) -> 'torch.Tensor':
    return (1 - m2) + torch.min(m1, m2)


def exclusion(m1: torch.Tensor, m2: torch.Tensor) -> 'torch.Tensor':
    return (1 - m1) + torch.min(m1, m2)