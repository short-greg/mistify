"""
For Type 1 Fuzzy Sets where 0 is False, 1 is True, and a 
number in between is partial truth
"""

"""
For Type 1 Fuzzy Sets where 0 is False, 1 is True, and a 
number in between is partial truth
"""

# 1st party
import typing

# 3rd party
import torch
import torch.nn as nn

# local
from .._base import (
    UnionOn, IntersectionOn, Complement
)
from .. import _base as base
from ... import functional
from . import _generate
from ...utils import weight_func, EnumFactory


class IntersectionOnEnum(EnumFactory):

    min = functional.min_on
    min_ada = functional.smooth_min_on
    prod = functional.prod_on


class UnionOnEnum(EnumFactory):

    max = functional.min_on
    max_ada = functional.smooth_max_on


class AndEnum(EnumFactory):

    min_max = functional.minmax
    min_max_ada = functional.ada_minmax


class OrEnum(EnumFactory):

    max_min = functional.maxmin
    maxmin_ada = functional.ada_minmax
    max_prod = functional.maxprod


class FuzzyComplement(Complement):

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return 1 - m


class FuzzyIntersectionOn(IntersectionOn):
    """Intersect sets that comprise a fuzzy set on a dimension

    Args:
        IntersectionOn (_type_): _description_
    """

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
        self._f = IntersectionOnEnum.factory(f)
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return self._f(m, dim=self.dim, keepdim=self.keepdim)


class FuzzyUnionOn(UnionOn):

    def __init__(self, f: str='max', dim: int=-1, keepdim: bool=False):
        super().__init__()
        self._f = UnionOnEnum.factory(f)
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return self._f(m, dim=self.dim, keepdim=self.keepdim)


class FuzzyOr(base.Or):

    def __init__(
        self, in_features: int, out_features: int, n_terms: int=None, 
        f: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]="max_min",
        wf: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]="clamp"
    ):
        super().__init__()
        if n_terms is not None:
            shape = (n_terms, in_features, out_features)
        else:
            shape = (in_features,  out_features)
        self.weight = nn.parameter.Parameter(_generate.positives(*shape))
        self._f = OrEnum.factory(f)
        self._wf = weight_func(wf)
        self._n_terms = n_terms
        self._in_features = in_features
        self._out_features = out_features
    

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        
        weight = self._wf(self.weight)
        return self._f(m, weight)


class FuzzyAnd(base.Or):

    def __init__(
        self, in_features: int, out_features: int, n_terms: int=None, 
        f: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]="minmax",
        wf: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]="clamp"
    ):
        super().__init__()
        if n_terms is not None:
            shape = (n_terms, in_features, out_features)
        else:
            shape = (in_features,  out_features)
        self.weight = nn.parameter.Parameter(_generate.negatives(*shape))
        self._wf = weight_func(wf)
        self._n_terms = n_terms
        self._in_features = in_features
        self._out_features = out_features
        self._f = AndEnum.factory(f)
    
    def forward(self, m: torch.Tensor) -> torch.Tensor:
        
        weight = self._wf(self.weight)
        return self._f(m, weight)


class FuzzyElse(base.Else):

    def forward(self, m: torch.Tensor):

        return torch.clamp(1 - m.sum(self.dim, keepdim=self.keepdim), 0, 1)
