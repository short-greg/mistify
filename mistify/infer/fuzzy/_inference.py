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
from abc import abstractmethod

# 3rd party
import torch
import torch.nn as nn

# local
from .._base import (
    UnionOn, IntersectionOn, Complement
)
from .. import _base as base
from .. import functional
from . import _generate
from ..utils import weight_func


class FuzzyComplement(Complement):

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return 1 - m


class FuzzyIntersectionOn(IntersectionOn):

    def __init__(self, f: str='min', dim: int=-1, keepdim: bool=False):
        print(functional)
        super().__init__()
        if f == 'min':
            self._f = functional.min_on
        elif f == 'min_ada':
            self._f = functional.smooth_min_on
        elif f == 'prod':
            self._f = functional.prod_on
        elif isinstance(f, typing.Callable):
            self._f = f
        else:
            raise ValueError(f'Invalid intersection')
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return self._f(m, dim=self.dim, keepdim=self.keepdim)


class FuzzyUnionOn(UnionOn):

    def __init__(self, f: str='max', dim: int=-1, keepdim: bool=False):
        super().__init__()
        if f == 'max':
            self._f = functional.max_on
        elif f == 'max_ada':
            self._f = functional.smooth_max_on
        elif isinstance(f, typing.Callable):
            self._f = f
        else:
            raise ValueError(f'Invalid union')
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return self._f(m, dim=self.dim, keepdim=self.keepdim)


class FuzzyOr(base.Or):

    def __init__(
        self, in_features: int, out_features: int, n_terms: int=None, 
        f: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]="maxmin",
        wf: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]="clamp"
    ):
        super().__init__()
        if n_terms is not None:
            shape = (n_terms, in_features, out_features)
        else:
            shape = (in_features,  out_features)
        self.weight = nn.parameter.Parameter(_generate.positives(*shape))
        self._wf = weight_func(wf)
        self._n_terms = n_terms
        self._in_features = in_features
        self._out_features = out_features
    
        if f == "maxmin":
            self._f = functional.maxmin
        elif f == "maxprod":
            self._f = functional.maxprod
        elif f == "maxmin_ada":
            self._f = functional.ada_minmax
        else:
            self._f = f

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
    
        if f == "minmax":
            self._f = functional.minmax
        elif f == "minmax_ada":
            self._f = functional.ada_minmax
        else:
            self._f = f

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        
        weight = self._wf(self.weight)
        return self._f(m, weight)


class FuzzyElse(base.Else):

    def forward(self, m: torch.Tensor):

        return torch.clamp(1 - m.sum(self.dim, keepdim=self.keepdim), 0, 1)



# class FuzzyComposition(CompositionBase):
#     """Base class for calculating relationship between two fuzzy sets
#     """

#     @abstractmethod
#     def forward(self, m: torch.Tensor) -> torch.Tensor:
#         raise NotImplementedError

#     def clamp_weights(self):
#         """Ensure the weights of the fuzzy set are between 0 and 1
#         """
#         self.weight.data = torch.clamp(self.weight.data, 0, 1)


# class MaxMin(FuzzyComposition):
#     """OrNeuron that uses MaxMinComposition
#     """

#     def init_weight(self, in_features: int, out_features: int, in_variables: int = None) -> torch.Tensor:
#         return positives(get_comp_weight_size(in_features, out_features, in_variables))
    
#     def forward(self, m: torch.Tensor) -> torch.Tensor:
#         return maxmin(m, self.weight)


# class MaxProd(FuzzyComposition):
#     """Or Neuron that uses MaxProduct Composition
#     """

#     def init_weight(self, in_features: int, out_features: int, in_variables: int = None) -> torch.Tensor:
#         return positives(get_comp_weight_size(in_features, out_features, in_variables))

#     def forward(self, m: torch.Tensor) -> torch.Tensor:
#         # assume inputs are binary
#         # binarize the weights
#         return maxprod(m, self.weight)


# class MinMax(FuzzyComposition):
#     """And Neuron that uses the minmax operation
#     """

#     def init_weight(self, in_features: int, out_features: int, in_variables: int = None) -> torch.Tensor:
#         return negatives(get_comp_weight_size(in_features, out_features, in_variables))
    
#     def forward(self, m: torch.Tensor):
#         # assume inputs are binary
#         # binarize the weights
#         return minmax(m, self.weight)


# class Inner(nn.Module):
#     """
#     """

#     def __init__(self, f: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
#         """_summary_

#         Args:
#             f (typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): _description_
#         """
#         super().__init__()
#         self._f = f
    
#     def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#         return self._f(x.unsqueeze(-1), y[None])


# class Outer(nn.Module):
        
#     def __init__(self, f: typing.Callable[[torch.Tensor], torch.Tensor], agg_dim: int=-2, idx: int=None):
#         super().__init__()
#         if idx is None:
#             self._f =  partial(f, dim=agg_dim)
#         if idx is not None:
#             self._f = lambda x: f(x, dim=agg_dim)[idx]
#         self._idx = idx
#         # change from being manually defined
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self._f(x)


# class FuzzyRelation(FuzzyComposition):

#     def __init__(
#         self, in_features: int, out_features: int, 
#         in_variables: int=None, 
#         inner: typing.Union[Inner, typing.Callable]=None, 
#         outer: typing.Union[Outer, typing.Callable]=None
#     ):
#         super().__init__(in_features, out_features, in_variables)
#         inner = inner or Inner(torch.min)
#         outer = outer or Outer(torch.max, idx=0)
#         if not isinstance(inner, Inner):
#             inner = Inner(inner)
#         if not isinstance(outer, Outer):
#             outer = Outer(outer)
#         self.inner = inner
#         self.outer = outer

#     def init_weight(self, in_features: int, out_features: int, in_variables: int = None) -> torch.Tensor:
#         return positives(get_comp_weight_size(in_features, out_features, in_variables))
                         
#     def forward(self, m: torch.Tensor):
#         return self.outer(self.inner(m, self.weight))



# class FuzzyAggregator(nn.Module):

#     def __init__(self, dim: int, keepdim: bool=False):
#         super().__init__()
#         self.dim = dim
#         self.keepdim = keepdim

#     @abstractmethod
#     def forward(self, m: torch.Tensor) -> torch.Tensor:
#         pass

# class MaxMinAgg(nn.Module):

#     def __init__(self, in_variables: int, in_features: int, out_features: int, agg_features: int):
#         super().__init__()
#         self._max_min = MaxMin(in_features, out_features * agg_features, in_variables)
#         self._agg_features = agg_features
    
#     def forward(self, m: torch.Tensor):
#         data = self._max_min.forward(m)
#         return data.view(*data.shape[:-1], -1, self._agg_features).max(dim=-1)[0]


# class WithFuzzyElse(nn.Module):

#     def __init__(self, dim=-1):
#         super().__init__()
#         self.else_ = FuzzyElse(dim)
    
#     @property
#     def dim(self) -> int:
#         return self.else_.dim

#     @dim.setter
#     def dim(self, dim: int) -> None:
#         self.else_.dim = dim

#     def forward(self, m: torch.Tensor):

#         else_ = self.else_.forward(m)
#         return torch.cat([m, else_], dim=self.else_.dim)



