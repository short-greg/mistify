# 1st party
import typing

# 3rd party
import torch
import torch.nn as nn

# local
from .. import _functional
from ..utils import EnumFactory
from .._functional import _set_ops as set_ops


class JunctionOn(nn.Module):
    """Intersect sets that comprise a fuzzy set on a dimension
    """

    F = EnumFactory()

    def __init__(self, f: typing.Union[typing.Callable, str]='inter_on', dim: int=-1, keepdim: bool=False):
        """Join sets that comprise a fuzzy set on a specified dimension

        Args:
            f (str, optional): The function to use for junction. Defaults to 'min'.
            dim (int, optional): Dimension to junction on. Defaults to -1.
            keepdim (bool, optional): Whether to keep the dim or not. Defaults to False.

        Raises:
            ValueError: If the junction function is invalid
        """
        super().__init__()
        self._f = self.F.f(f)
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return self._f(m, dim=self.dim, keepdim=self.keepdim)


class IntersectionOn(JunctionOn):
    """Intersect sets that comprise a fuzzy set on a dimension
    """

    F = EnumFactory(
        inter_on=_functional.inter_on,
        smooth_inter_on=_functional.smooth_inter_on,
        bounded_inter_on=_functional.bounded_inter_on,
        prob_inter_on=_functional.prob_inter_on,
        ada_inter_on=_functional.ada_inter_on
    )


class UnionOn(JunctionOn):
    """Union on a specific dimension
    """
    F = EnumFactory(
        union_on = _functional.union_on,
        smooth_union_on = _functional.smooth_union_on,
        bounded_inter_on=_functional.bounded_union_on,
        ada_union_on=_functional.ada_union_on,
        prob_union_on=_functional.prob_union_on
    )

    def __init__(self, f: typing.Union[typing.Callable, str]='union_on', dim: int=-1, keepdim: bool=False):
        """Union sets that comprise a fuzzy set on a specified dimension

        Args:
            f (str, optional): The function to use for union. Defaults to 'min'.
            dim (int, optional): Dimension to union on. Defaults to -1.
            keepdim (bool, optional): Whether to keep the dim or not. Defaults to False.

        Raises:
            ValueError: If the union function is invalid
        """
        super().__init__(f, dim, keepdim)


class Else(nn.Module):

    F = EnumFactory(
        else_=set_ops.else_,
        signed_else=set_ops.signed_else_
    )

    def __init__(self, f: typing.Callable='else_', dim=-1, keepdim: bool = False):
        """Calculate else along a certain dimension It calculates the sum of all the membership values along the dimension

        Args:
            f (typing.Callable, optional): The else function. Defaults to 'fuzzy'.
            dim (int, optional): The dimension to take the else on. Defaults to -1.
            keepdim (bool, optional): Whether to keep the dimension after taking the else. Defaults to False.
        """
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        self._f = self.F.f(f)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """Calculate the else for the fuzzy set

        Args:
            m (torch.Tensor): the membership value

        Returns:
            torch.Tensor: the else of the fuzzy set
        """
        return self._f(m, dim=self.dim, keepdim=self.keepdim)


class CatElse(nn.Module):

    def __init__(self, f: typing.Callable='else_', dim=-1):
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
        complement = set_ops.complement,
        signed_complement = set_ops.signed_complement,
    )

    def __init__(self, f: typing.Callable='complement'):
        """Take the complement of the set

        Args:
            f (typing.Callable, optional): The complement function. Defaults to 'boolean'.
        """
        super().__init__()
        self._f = self.F.f(f)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """Calculate the complement for the set

        Args:
            m (torch.Tensor): the membership value

        Returns:
            torch.Tensor: the complement of the set
        """
        return self._f(m)


class CatComplement(nn.Module):

    def __init__(self, f: typing.Callable='complement', dim=-1):
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
