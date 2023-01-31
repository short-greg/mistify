import typing
import torch
import torch.nn as nn
from .utils import reduce, get_comp_weight_size,  maxmin, minmax, maxprod
from abc import abstractmethod
from .base import Set, SetParam, CompositionBase
import math


class FuzzySet(Set):

    def transpose(self, first_dim, second_dim) -> 'FuzzySet':
        assert self._value_size is not None
        return FuzzySet(self._data.transpose(first_dim, second_dim), self._is_batch)

    def intersect_on(self, dim: int=-1):
        return FuzzySet(torch.min(self.data, dim=dim)[0], self.is_batch)

    def unify_on(self, dim: int=-1):
        return FuzzySet(torch.max(self.data, dim=dim)[0], self.is_batch)

    def differ(self, other: 'FuzzySet'):
        return FuzzySet(torch.clamp(self.data - other.data, 0, 1), self.is_batch)
    
    def unify(self, other: 'FuzzySet'):
        return FuzzySet(torch.max(self.data, other.data), self.is_batch)

    def intersect(self, other: 'FuzzySet'):
        return FuzzySet(torch.min(self.data, other.data), self._is_batch)

    def inclusion(self, other: 'FuzzySet') -> 'FuzzySet':
        return FuzzySet(
            (1 - other.data) + torch.min(self.data, other.data), self._is_batch
        )

    def exclusion(self, other: 'FuzzySet') -> 'FuzzySet':
        return FuzzySet(
            (1 - self.data) + torch.min(self.data, other.data), self._is_batch
        )
    
    def __sub__(self, other: 'FuzzySet'):
        return self.differ(other)

    def __mul__(self, other: 'FuzzySet'):
        return intersect(self, other)

    def __add__(self, other: 'FuzzySet'):
        return self.unify(other)
    
    def convert_variables(self, *size_after: int):
        
        if self._is_batch:
            return FuzzySet(
                self._data.view(self._data.size(0), *size_after, -1), True
            )
        return self.__class__(
            self._data.view(*size_after, -1), False
        )

    def reshape(self, *size: int):
        return FuzzySet(
            self.data.reshape(*size), self.is_batch
        )

    @classmethod
    def negatives(cls, *size: int, is_batch: bool=None, dtype=torch.float32, device='cpu'):

        return FuzzySet(
            torch.zeros(*size, dtype=dtype, device=device), is_batch
        )
    
    @classmethod
    def positives(cls, *size: int, dtype=torch.float32, is_batch: bool=None, device='cpu'):

        return FuzzySet(
            torch.ones(*size, dtype=dtype, device=device), 
            is_batch
        )

    @classmethod
    def rand(cls, *size: int, is_batch: bool=None, dtype=torch.float32, device='cpu'):

        return FuzzySet(
            (torch.rand(*size, device=device)).type(dtype), 
            is_batch
        )

    def transpose(self, first_dim, second_dim) -> 'FuzzySet':
        assert self._value_size is not None
        return FuzzySet(self._data.transpose(first_dim, second_dim), self._is_batch)

    @property
    def shape(self) -> torch.Size:
        return self._data.shape

    def __getitem__(self, key):
        return FuzzySet(self.data[key])


class FuzzyCalcApprox(object):

    def intersect(self, x: FuzzySet, y: FuzzySet):
        pass

    def union(self, x: FuzzySet, y: FuzzySet):
        pass


def intersect(m: FuzzySet, m2: FuzzySet):
    return FuzzySet(torch.min(m.data, m2.data))


def unify(m: FuzzySet, m2: FuzzySet):
    return FuzzySet(torch.max(m.data, m2.data))


def differ(m: FuzzySet, m2: FuzzySet):
    return FuzzySet((m.data - m2._data).clamp(0.0, 1.0))


class FuzzySetParam(SetParam):

    def __init__(self, set_: typing.Union[FuzzySet, torch.Tensor], requires_grad: bool=True):
 
        if isinstance(set_, torch.Tensor):
            set_ = FuzzySet(set_)
        super().__init__(set_, requires_grad=requires_grad)


class MaxMin(CompositionBase):

    def init_weight(self, in_features: int, out_features: int, in_variables: int = None) -> SetParam:
        return FuzzySetParam(
            FuzzySet.positives(get_comp_weight_size(in_features, out_features, in_variables))
        )

    def forward(self, m: FuzzySet):
        # assume inputs are binary
        # binarize the weights
        return FuzzySet(
            maxmin(self.prepare_inputs(m), self.weight.data
        ), m.is_batch)


class MaxProd(CompositionBase):

    def init_weight(self, in_features: int, out_features: int, in_variables: int = None) -> SetParam:
        return FuzzySetParam(
            FuzzySet.positives(get_comp_weight_size(in_features, out_features, in_variables))
        )

    def forward(self, m: FuzzySet):
        # assume inputs are binary
        # binarize the weights
        return FuzzySet(
            maxprod(self.prepare_inputs(m), self.weight.data), m.is_batch
        )



class MinMax(CompositionBase):

    def init_weight(self, in_features: int, out_features: int, in_variables: int = None) -> SetParam:
        return FuzzySetParam(
            FuzzySet.negatives(get_comp_weight_size(in_features, out_features, in_variables))
        )
 
    def inner(self):
        return 
    
    def forward(self, m: FuzzySet):
        # assume inputs are binary
        # binarize the weights
        return FuzzySet(
            minmax(self.prepare_inputs(m), self.weight.data), m.is_batch
        )


class FuzzyRelation(CompositionBase):

    def __init__(
        self, in_features: int, out_features: int, 
        in_variables: int=None, 
        inner=None, outer=None
    ):
        super().__init__()
        self.inner = inner or torch.min
        self.outer = outer or (lambda x: torch.max(x, dim=-2)[0])

        self.weight = FuzzySetParam(
            FuzzySet.positives(get_comp_weight_size(in_features, out_features, in_variables))
        )

    def init_weight(self, in_features: int, out_features: int, in_variables: int = None) -> SetParam:
        return FuzzySetParam(
            FuzzySet.positives(get_comp_weight_size(in_features, out_features, in_variables))
        )

    def forward(self, m: FuzzySet):

        return FuzzySet(
            self.outer(
                self.inner(self.prepare_inputs(m), self.weight.data[None])
            ), m.is_batch
        )


class IntersectOn(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, m: FuzzySet) -> FuzzySet:
        return FuzzySet(torch.min(m.data, dim=self.dim), m.is_batch)


class UnionOn(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, m: FuzzySet) -> FuzzySet:
        return FuzzySet(torch.max(m.data, dim=self.dim), m.is_batch)


class MaxMinAgg(nn.Module):

    def __init__(self, in_variables: int, in_features: int, out_features: int, agg_features: int, complement_inputs: bool=False):
        super().__init__()
        self._max_min = MaxMin(in_features, out_features * agg_features, complement_inputs, in_variables)
        self._agg_features = agg_features
    
    @property
    def to_complement(self) -> bool:
        return self._max_min._complement_inputs
    
    def forward(self, m: FuzzySet):
        data = self._max_min.forward(m).data
        data = data.view(*data.shape[:-1], -1, self._agg_features).max(dim=-1)[0]
        return FuzzySet(data, m.is_batch)


class FuzzyElse(nn.Module):

    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, m: FuzzySet):

        return FuzzySet(
            torch.clamp(1 - m.data.sum(self.dim, keepdim=True), 0, 1),
            is_batch=m.is_batch
        )


class WithFuzzyElse(nn.Module):

    def __init__(self, dim=-1):
        super().__init__()
        self.else_ = FuzzyElse(dim)
    
    @property
    def dim(self) -> int:
        return self.else_.dim

    @dim.setter
    def dim(self, dim: int) -> None:
        self.else_.dim = dim

    def forward(self, m: FuzzySet):

        else_ = self.else_.forward(m)
        return FuzzySet(
            torch.cat([m.data, else_.data], dim=self.else_.dim),
            is_batch=m.is_batch
        )
