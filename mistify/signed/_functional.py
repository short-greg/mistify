# 1st party

# 3rd party
import torch

# local
from ..functional import to_signed, to_binary


def differ(m1: torch.Tensor, m2: 'torch.Tensor') -> 'torch.Tensor':
    """Calculate difference between Ternary tensors

    Args:
        m1 (torch.Tensor): 
        m2 (torch.Tensor): 

    Returns:
        torch.Tensor: 
    """
    m1 = to_binary(m1)
    m2 = to_binary(m2)
    return to_signed(m1 - m2).clamp(-1, 1)


def unify(m1: torch.Tensor, m2: 'torch.Tensor') -> 'torch.Tensor':
    """Calcluate the union of two ternary sets

    Args:
        m1 (torch.Tensor): Ternary set
        m2 (torch.Tensor): Ternary set

    Returns:
        torch.Tensor: Unified ternary set
    """
    return torch.max(m1, m2)


def intersect(m1: torch.Tensor, m2: 'torch.Tensor') -> 'torch.Tensor':
    """Calcluate the intersection of two ternary sets

    Args:
        m1 (torch.Tensor): Ternary set
        m2 (torch.Tensor): Ternary set

    Returns:
        torch.Tensor: Intersected ternary set
    """
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


def complement(m: torch.Tensor) -> torch.Tensor:
    return -m


def forget(m: torch.Tensor, p: float) -> torch.Tensor:
    """Randomly forget values (this will make them unknown)

    Args:
        m (torch.Tensor): the membership matrix
        p (float): the probability of forgetting

    Returns:
        torch.Tensor: the tensor with randomly forgotten values
    """
    return m * (torch.rand_like(m) < p).type_as(m)
