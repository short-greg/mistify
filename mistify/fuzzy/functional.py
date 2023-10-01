import typing
import torch


def differ(m: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
    """
    Take the difference between two fuzzy sets
    
    Args:
        m (torch.Tensor): Fuzzy set to subtract from 
        m2 (torch.Tensor): Fuzzy set to subtract

    Returns:
        torch.Tensor: 
    """
    return (m - m2).clamp(0.0, 1.0)


def intersect(m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
    """intersect two fuzzy sets

    Args:
        m1 (torch.Tensor): Fuzzy set to intersect
        m2 (torch.Tensor): Fuzzy set to intersect with

    Returns:
        torch.Tensor: Intersection of two fuzzy sets
    """
    return torch.min(m1, m2)


def intersect_on(m: torch.Tensor, dim: int=-1) -> torch.Tensor:
    """Intersect elements of a fuzzy set on specfiied dimension

    Args:
        m (torch.Tensor): Fuzzy set to intersect

    Returns:
        torch.Tensor: Intersection of two fuzzy sets
    """
    return torch.min(m, dim=dim)[0]


def unify(m: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
    """union on two fuzzy sets

    Args:
        m (torch.Tensor):  Fuzzy set to take union of
        m2 (torch.Tensor): Fuzzy set to take union with

    Returns:
        torch.Tensor: Union of two fuzzy sets
    """
    return torch.max(m, m2)


def unify_on(m: torch.Tensor, dim: int=-1) -> torch.Tensor:
    """Unify elements of a fuzzy set on specfiied dimension

    Args:
        m (torch.Tensor): Fuzzy set to take the union of

    Returns:
        torch.Tensor: Union of two fuzzy sets
    """
    return torch.max(m, dim=dim)[0]


def inclusion(m1: torch.Tensor, m2: torch.Tensor, dim: int=None) -> 'torch.Tensor':
    base = (1 - m2) + torch.min(m1, m2)
    if dim is None:
        return base.type_as(m1)
    return base.min(dim=dim)[0].type_as(m1)


def exclusion(m1: torch.Tensor, m2: torch.Tensor, dim: int=None) -> 'torch.Tensor':
    base = (1 - m1) + torch.min(m1, m2)
    if dim is None:
        return base.type_as(m1)
    return base.min(dim=dim)[0].type_as(m1)


def weight_func(wf: typing.Union[str, typing.Callable]) -> typing.Callable:
    if isinstance(wf, str):
        return torch.sigmoid if wf == "sigmoid" else torch.clamp
    return wf
