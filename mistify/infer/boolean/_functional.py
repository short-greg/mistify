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

def complement(m: torch.Tensor) -> torch.Tensor:
    return 1 - m


def forget(m: torch.Tensor, p: float) -> torch.Tensor:
    """Randomly forget values (this will make them unknown)

    Args:
        m (torch.Tensor): the membership matrix
        p (float): the probability of forgetting

    Returns:
        torch.Tensor: the tensor with randomly forgotten values
    """
    return m * (torch.rand_like(m) < p).type_as(m) + 0.5


def else_(m: torch.Tensor, dim: int=-1, keepdim: bool=False) -> torch.Tensor:

        y = m.max(dim=dim, keepdim=keepdim)[0]
        return (1 - y)
