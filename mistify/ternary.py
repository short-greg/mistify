# 1st party
import typing
import typing


# 3rd party
import torch
import torch.nn as nn

# local
from ._core import CompositionBase, maxmin, ComplementBase, get_comp_weight_size


def differ(m1: torch.Tensor, m2: 'torch.Tensor') -> 'torch.Tensor':
    """Calculate difference between Ternary tensors

    Args:
        m1 (torch.Tensor): 
        m2 (torch.Tensor): 

    Returns:
        torch.Tensor: 
    """
    return (m1 - m2).clamp(-1.0, 1.0)

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

def inclusion(m1: torch.Tensor, m2: 'torch.Tensor') -> 'torch.Tensor':
    """Calculate whether first set includes the second

    Args:
        m1 (torch.Tensor): Ternary set
        m2 (torch.Tensor): Ternary set

    Returns:
        torch.Tensor: Level of inclusion of second set by first
    """
    return (1 - m2) + torch.min(m1, m2)

def exclusion(m1: torch.Tensor, m2: 'torch.Tensor') -> 'torch.Tensor':
    """Calculate whether first set excludes the second

    Args:
        m1 (torch.Tensor): Ternary set
        m2 (torch.Tensor): Ternary set

    Returns:
        torch.Tensor: Level of exclusion of second set by the first
    """
    return (1 - m1) + torch.min(m1, m2)


def negatives(*size: int, dtype=torch.float32, device='cpu') -> torch.Tensor:
    """
    Args:
        dtype (, optional): The type to set the ternary set to. Defaults to torch.float32.
        device (str, optional): _description_. Defaults to 'cpu'.

    Returns:
        torch.Tensor: Ternary set with all negative values
    """
    return -torch.ones(*size, dtype=dtype, device=device)

def positives(*size: int, dtype=torch.float32, device='cpu'):
    """
    Args:
        dtype (, optional): The type to set the ternary set to. Defaults to torch.float32.
        device (str, optional): _description_. Defaults to 'cpu'.

    Returns:
        torch.Tensor: Ternary set with all positive values
    """

    return torch.ones(*size, dtype=dtype, device=device)

def unknowns(*size: int, dtype=torch.float32, device='cpu') -> torch.Tensor:
    """
    Args:
        dtype (, optional): The type to set the ternary set to. Defaults to torch.float32.
        device (str, optional): _description_. Defaults to 'cpu'.

    Returns:
        torch.Tensor: Ternary set with all unknown values
    """

    return torch.zeros(*size, dtype=dtype, device=device)


def rand(*size: int, dtype=torch.float32, device='cpu') -> torch.Tensor:
    """
    Args:
        dtype (_type_, optional): . Defaults to torch.float32.
        device (str, optional): . Defaults to 'cpu'.

    Returns:
        torch.Tensor: Random value
    """

    return ((torch.rand(*size, device=device, dtype=dtype)) * 3).floor() - 1


class TernaryComposition(CompositionBase):
    """Calculate the relation between two ternary sets
    """

    def init_weight(self, in_features: int, out_features: int, in_variables: int = None) -> torch.Tensor:
        return positives(get_comp_weight_size(in_features, out_features, in_variables))

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """ 
        Args:
            m (torch.Tensor): 

        Returns:
            torch.Tensor: Relationship between ternary set and the weights
        """
        return maxmin(m, self.weight.data[None]).round()


class TernaryComplement(ComplementBase):

    def complement(self, m: torch.Tensor) -> torch.Tensor:
        """
        Args:
            m (torch.Tensor): The membership tensor

        Returns:
            torch.Tensor: The complement of the ternary set
        """

        return -m
