import torch


def differ(m1: torch.Tensor, m2: torch.Tensor) -> 'torch.Tensor':
    return (m1 - m2).clamp(0.0, 1.0)


def unify(m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
    return torch.max(m1, m2)


def intersect(m1: torch.Tensor, m2: torch.Tensor) -> 'torch.Tensor':
    return torch.min(m1, m2)


def unify_on(m1: torch.Tensor, dim: int=-1, keepdim: bool=False) -> 'torch.Tensor':
    return torch.max(m1, dim=dim, keepdim=keepdim)


def intersect_on(m1: torch.Tensor, dim: int=-1, keepdim: bool=False) -> 'torch.Tensor':
    return torch.min(m1, dim=dim, keepdim=keepdim)


def inclusion(m1: torch.Tensor, m2: torch.Tensor, dim: int=None) -> 'torch.Tensor':
    base = (m1 <= m2).type_as(m1)
    if dim is None:
        return base
    return base.min(dim=dim)[0]


def exclusion(m1: torch.Tensor, m2: torch.Tensor, dim: int=None) -> 'torch.Tensor':
    base = (m1 >= m2).type_as(m1)    
    if dim is None:
        return base
    return base.min(dim=dim)[0]
