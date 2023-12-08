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
            f (typing.Callable, optional): The else function. Defaults to 'fuzzy'.
            dim (int, optional): The dimension to take the else on. Defaults to -1.
            keepdim (bool, optional): Whether to keep the dimension after taking the else. Defaults to False.
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


class CatElse(nn.Module):

    def __init__(self, f: typing.Callable='fuzzy', dim=-1):
        """Take the "Else" of the fuzzy set and then complement

        Args:
            f (typing.Callable, optional): The else function. Defaults to 'fuzzy'.
            dim (int, optional): The dimension to take the else on. Defaults to -1.
        """
        super().__init__()
        self._else = Else(f, dim, True)

    def forward(self, m: torch.Tensor) -> torch.Tensor:

        else_ = self._else(m)
        return torch.cat(
            [m, else_], dim=self._else.dim
        )


class Complement(nn.Module):

    F = EnumFactory(
        signed = signed.complement,
        boolean = boolean.complement,
        fuzzy= fuzzy.complement
    )

    def __init__(self, f: typing.Callable='boolean'):
        """Take the complement of the set

        Args:
            f (typing.Callable, optional): The complement function. Defaults to 'boolean'.
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


class CatComplement(nn.Module):

    def __init__(self, f: typing.Callable='boolean', dim=-1):
        """Take the complement and then concatenate it

        Args:
            f (typing.Callable, optional): The complement function. Defaults to 'boolean'.
            dim (int, optional): The dim to cat on. Defaults to -1.
        """
        super().__init__()
        self._complement = Complement(f)
        self.dim = dim

    def forward(self, m: torch.Tensor) -> torch.Tensor:

        complement = self._complement(m)
        return torch.cat(
            [m, complement], dim=self.dim
        )


class Exclusion(nn.Module):

    @abstractmethod
    def forward(self, m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Inclusion(nn.Module):

    @abstractmethod
    def forward(self, m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError