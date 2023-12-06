# 1st party
import typing
from abc import abstractmethod

# 3rd party
import torch
import torch.nn as nn

# local
from .. import functional
from ..utils import EnumFactory
from . import signed
from . import fuzzy
from . import boolean


class JunctionOn(nn.Module):
    """Intersect sets that comprise a fuzzy set on a dimension
    """
    F = EnumFactory()

    def __init__(self, f: str='min', dim: int=-1, keepdim: bool=False):
        """Intersect sets that comprise a fuzzy set on a specified dimension

        Args:
            f (str, optional): The function to use for intersection. Defaults to 'min'.
            dim (int, optional): Dimension to intersect on. Defaults to -1.
            keepdim (bool, optional): Whether to keep the dim or not. Defaults to False.

        Raises:
            ValueError: If the intersection function is invalid
        """
        super().__init__()
        self._f = self.F.factory(f)
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return self._f(m, dim=self.dim, keepdim=self.keepdim)


class IntersectionOn(nn.Module):
    """Intersect sets that comprise a fuzzy set on a dimension
    """
    F = EnumFactory(
        min=functional.min_on,
        min_ada=functional.smooth_min_on,
        prod=functional.prod_on
    )


class UnionOn(nn.Module):
    """Union on a specific dimension
    """
    F = EnumFactory(
        max = functional.max_on,
        max_ada = functional.smooth_max_on
    )


class Else(nn.Module):

    F = EnumFactory(
        boolean=boolean.else_,
        fuzzy=fuzzy.else_,
        signed= signed.else_
    )
    def __init__(self, f: typing.Callable='fuzzy', dim=-1, keepdim: bool = False):
        """Calculate else along a certain dimension It calculates the sum of all the membership values along the dimension

        Args:
            dim (int, optional): _description_. Defaults to -1.
            keepdim (bool, optional): _description_. Defaults to False.
        """
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        self._f = self.F.factory(f)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """Calculate the else for the fuzzy set

        Args:
            m (torch.Tensor): the membership value

        Returns:
            torch.Tensor: the else of the fuzzy set
        """
        return self._f(m, dim=self.dim, keepdim=self.keepdim)


class Complement(nn.Module):

    F = EnumFactory(
        signed = signed.complement,
        boolean = boolean.complement,
        fuzzy= fuzzy.complement
    )

    def __init__(self, f: typing.Callable='boolean'):
        """Calculate else along a certain dimension It calculates the sum of all the membership values along the dimension

        Args:
            dim (int, optional): _description_. Defaults to -1.
            keepdim (bool, optional): _description_. Defaults to False.
        """
        super().__init__()
        self._f = self.F.factory(f)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """Calculate the complement for the set

        Args:
            m (torch.Tensor): the membership value

        Returns:
            torch.Tensor: the complement of the set
        """
        return self._f(m)


class Exclusion(nn.Module):

    @abstractmethod
    def forward(self, m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Inclusion(nn.Module):

    @abstractmethod
    def forward(self, m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
