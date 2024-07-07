import torch

# local
from ._m import (
    to_signed, to_boolean, 
)
from ._grad import (
    G,
    ClampG, MinOnG, MaxOnG
)
from ._join import inter_on


def differ(m: torch.Tensor, m2: torch.Tensor, g: G=None) -> torch.Tensor:
    """
    Take the difference between two fuzzy sets
    
    Args:
        m (torch.Tensor): Fuzzy set to subtract from 
        m2 (torch.Tensor): Fuzzy set to subtract

    Returns:
        torch.Tensor: 
    """
    difference = (m - m2)
    if g is None:
        return difference.clamp(0.0, 1.0)
    return ClampG.apply(difference, 0.0, 1.0, g)


def signed_differ(m1: torch.Tensor, m2: torch.Tensor, g: G=None) -> torch.Tensor:
    """Calculate difference between Ternary tensors

    Args:
        m1 (torch.Tensor): 
        m2 (torch.Tensor): 

    Returns:
        torch.Tensor: 
    """
    # 1, -1 = 1
    # -1, 1 = 1
    # 1, 1 = -1
    # 1, 0 = 0
    # -torch.sign(m1)torch.min(m1, m2)

    m1 = to_boolean(m1)
    m2 = to_boolean(m2)
    difference = to_signed(m1 - m2)
    if g is None:
        return difference.clamp(-1, 1)
    return ClampG.apply(difference, -1.0, 1.0, g)


def inclusion(m1: torch.Tensor, m2: torch.Tensor, dim: int=None, g: G=None) -> 'torch.Tensor':
    """Calculate whether m1 is included in m2. If dim is None then it will calculate per
    element otherwise it will aggregate over that dimension

    Args:
        m1 (torch.Tensor): The membership to calculate the inclusion of
        m2 (torch.Tensor): The membership to check if m1 is included
        dim (int, optional): The dimension to aggregate over. Defaults to None.

    Returns:
        torch.Tensor: the tensor describing inclusion
    """
    base = (1.0 - m1) 
    if g is None:
        included = base + torch.min(m2, m1)
    else:
        
        included = base + inter_on(m2, m1, g)
    if dim is None:
        return included.type_as(m1)
    if g is None:
        return included.min(dim=dim)[0].type_as(m1)
    return MinOnG.apply(included, dim, False, g)


def exclusion(m1: torch.Tensor, m2: torch.Tensor, dim: int=None, g: G=None) -> 'torch.Tensor':
    """Calculate whether m1 is excluded from m2. If dim is None then it will calculate per
    element otherwise it will aggregate over that dimension

    Args:
        m1 (torch.Tensor): The membership to calculate the exclusion of
        m2 (torch.Tensor): The membership to check if m1 is excluded
        dim (int, optional): The dimension to aggregate over. Defaults to None.

    Returns:
        torch.Tensor: the tensor describing inclusion
    """

    base = (1.0 - m2) 
    if g is None:
        included = base + torch.min(m2, m1)
    else:
        included = base + inter_on(m2, m1, g)
    if dim is None:
        return included.type_as(m1)
    if g is None:
        return included.min(dim=dim)[0].type_as(m1)
    return MinOnG.apply(included, dim, False, g)


def complement(m: torch.Tensor) -> torch.Tensor:
    """Calculate the complement

    Args:
        m (torch.Tensor): The membership

    Returns:
        torch.Tensor: The fuzzy complement
    """
    return 1 - m


def signed_complement(m: torch.Tensor) -> torch.Tensor:
    """Calculate the complement of a signed membership

    Args:
        m (torch.Tensor): The membership

    Returns:
        torch.Tensor: The fuzzy complement
    """
    return - m


def else_(m: torch.Tensor, dim: int=-1, keepdim: bool=False, g: G=None) -> torch.Tensor:
    """Take the 'else' on a set

    Args:
        m (torch.Tensor): The fuzzy set
        dim (int, optional): The dimension to calculate on. Defaults to -1.
        keepdim (bool, optional): Whether to keep the dimension of m. Defaults to False.

    Returns:
        torch.Tensor: the else value of m along the dimension
    """
    if g is None:
        return 1 - m.max(dim=dim, keepdim=keepdim)[0]
    return 1 - MaxOnG.apply(m, dim, keepdim, g)[0]


def signed_else_(m: torch.Tensor, dim: int=-1, keepdim: bool=False, g: G=None) -> torch.Tensor:
    """Take the 'else' on a set that uses -1 for negatives

    Args:
        m (torch.Tensor): the fuzzy set
        dim (int, optional): the dimension to take the else on. Defaults to -1.
        keepdim (bool, optional): whether to keep the dimension. Defaults to False.

    Returns:
        torch.Tensor: the else
    """
    if g is None:
        return -m.max(dim=dim, keepdim=keepdim)[0]
    return -MaxOnG.apply(m, dim, keepdim, g)[0]


# def signed_inclusion(m1: torch.Tensor, m2: torch.Tensor, dim: int=None) -> 'torch.Tensor':
#     """Calculate whether m1 is included in m2. If dim is None then it will calculate per
#     element otherwise it will aggregate over that dimension

#     Args:
#         m1 (torch.Tensor): The membership to calculate the inclusion of
#         m2 (torch.Tensor): The membership to check if m1 is included
#         dim (int, optional): The dimension to aggregate over. Defaults to None.

#     Returns:
#         torch.Tensor: the tensor describing inclusion
#     """


#     base = (m1 <= m2).type_as(m1)
#     if dim is None:
#         return base
#     return base.min(dim=dim)[0]


# Figure out how to do it for signed

# def binary_exclusion(m1: torch.Tensor, m2: torch.Tensor, dim: int=None) -> 'torch.Tensor':
#     """Calculate whether m1 is excluded from m2. If dim is None then it will calculate per
#     element otherwise it will aggregate over that dimension

#     Args:
#         m1 (torch.Tensor): The membership to calculate the exclusion of
#         m2 (torch.Tensor): The membership to check if m1 is excluded
#         dim (int, optional): The dimension to aggregate over. Defaults to None.

#     Returns:
#         torch.Tensor: the tensor describing inclusion
#     """
#     base = (m1 >= m2).type_as(m1)    
#     if dim is None:
#         return base
#     return base.min(dim=dim)[0]

