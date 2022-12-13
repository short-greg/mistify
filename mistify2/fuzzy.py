import typing
import torch
import torch.nn as nn


class FuzzySet(object):

    def __init__(self, data: torch.Tensor, is_batch: bool=False, multiple_variables: bool=False):

        self._data = data
        self._n_samples = None
        self._n_variables = None
        self._multiple_variables = multiple_variables
        self._is_batch = is_batch
        if is_batch and multiple_variables:
            assert data.dim() == 3
            self._n_variables = data.size(1)
            self._batch_size = data.size(0)
        elif multiple_variables:
            assert data.dim() == 2
            self._n_samples = data.size(0)
        elif is_batch:
            assert data.dim() == 2
            self._batch_size = data.size(0)
        else:
            assert data.dim() == 1
    
    @property
    def is_batch(self) -> bool:
        return self._is_batch

    @property
    def multiple_variables(self) -> bool:
        return self._multiple_variables
    
    @property
    def data(self) -> torch.Tensor:
        return self._data

    def swap_variables(self) -> 'FuzzySet':
        assert self._multiple_variables
        if self._is_batch:
            dim1, dim2 = 1, 2
        else:
            dim1, dim2 = 0, 1
        return FuzzySet(self._data.transpose(dim2, dim1), self._is_batch, True)

    def convert_variables(self, n_after: int) -> 'FuzzySet':
        
        if self._is_batch:
            return FuzzySet(
                self._data.view(self._data.size(0), n_after, -1), True, True
            )
        return FuzzySet(
            self._data.view(n_after, -1), False, True
        )

    def differ(self, other: 'FuzzySet'):
        return differ(self, other)
    
    def unify(self, other: 'FuzzySet'):
        return unify(self, other)

    def intersect(self, other: 'FuzzySet'):
        return intersect(self, other)
    
    def __sub__(self, other: 'FuzzySet'):
        return self.differ(other)

    def __mul__(self, other: 'FuzzySet'):
        return intersect(self, other)

    def __add__(self, other: 'FuzzySet'):
        return self.unify(other)

    @classmethod
    def get_size(cls, n_features: int, batch_size: int=None, n_variables: int=None):

        if batch_size is not None and n_variables is not None:
            return (batch_size, n_variables, n_features)
        elif batch_size is not None:
            return (batch_size, n_features)
        elif n_variables is not None:
            return (n_variables, n_features)
        return (n_features,)
    
    @classmethod
    def zeros(cls, n_features: int, batch_size: int=None, n_variables: int=None, dtype=torch.float32, device='cpu'):

        size = cls.get_size(n_features, batch_size, n_variables)
        return FuzzySet(
            torch.zeros(*size, dtype=dtype, device=device), 
            batch_size is not None, n_variables is not None
        )

    @classmethod
    def ones(cls, n_features: int, batch_size: int=None, n_variables: int=None, dtype=torch.float32, device='cpu'):

        size = cls.get_size(n_features, batch_size, n_variables)
        return FuzzySet(
            torch.ones(*size, dtype=dtype, device=device), 
            batch_size is not None, n_variables is not None
        )

    @classmethod
    def rand(cls, n_features: int, batch_size: int=None, n_variables: int=None, dtype=torch.float32, device='cpu'):

        size = cls.get_size(n_features, batch_size, n_variables)
        return FuzzySet(
            torch.rand(*size, dtype=dtype, device=device), 
            batch_size is not None, n_variables is not None
        )


def intersect(m: FuzzySet, m2: FuzzySet):
    return FuzzySet(torch.min(m.data, m2.data))


def unify(m: FuzzySet, m2: FuzzySet):
    return FuzzySet(torch.max(m.data, m2.data))


def differ(m: FuzzySet, m2: FuzzySet):
    return FuzzySet((m.data - m2._data).clamp(0.0, 1.0))


class FuzzySetParam(nn.parameter.Parameter):

    def __init__(self, fuzzy_set: FuzzySet, requires_grad: bool=True):

        super().__init__(fuzzy_set.data, requires_grad=True)
        self._fuzzy_set = fuzzy_set

    @property
    def data(self) -> torch.Tensor:
        return self._fuzzy_set.data

    @data.setter
    def data(self, data: torch.Tensor):
        assert data.size() == self._fuzzy_set.data.size()
        super().data = data
        self._fuzzy_set.data = data

    @property
    def fuzzy_set(self) -> FuzzySet:
        return self._fuzzy_set
    
    @fuzzy_set.setter
    def fuzzy_set(self, fuzzy_set: 'FuzzySet'):
        self._fuzzy_set = fuzzy_set
        self.data = fuzzy_set.data
    

class MaxMinComp(nn.Module):

    def __init__(self, in_features: int, out_features: int, n_variables: int=None):

        super().__init__()
        self._n_variables = n_variables
        self._in_features = in_features
        self._out_features = out_features
        size = (in_features, out_features) if n_variables is None else (n_variables, in_features, out_features)
        self._weight_param = FuzzySetParam(
            FuzzySet.ones(
                *size
            )
        )
    
    def forward(self, m: FuzzySet):

        assert m.is_batch

        return FuzzySet(
            torch.max(torch.min(m.data, self._weight_param.data), dim=-2),
            is_batch=m.is_batch, multiple_variables=m.multiple_variables
        )


class MinMaxComp(nn.Module):

    def __init__(self, in_features: int, out_features: int, n_variables: int=None):

        super().__init__()
        self._n_variables = n_variables
        self._in_features = in_features
        self._out_features = out_features
        size = (in_features, out_features) if n_variables is None else (n_variables, in_features, out_features)
        self._weight_param = FuzzySetParam(
            FuzzySet.zeros(
                *size
            )
        )
    
    def forward(self, m: FuzzySet):

        return FuzzySet(
            torch.min(torch.max(m.data, self._weight_param.data), dim=-2),
            is_batch=m.is_batch, multiple_variables=m.multiple_variables
        )


class MaxProdComp(nn.Module):

    def __init__(self, in_features: int, out_features: int, n_variables: int=None):

        super().__init__()
        self._n_variables = n_variables
        self._in_features = in_features
        self._out_features = out_features
        size = (in_features, out_features) if n_variables is None else (n_variables, in_features, out_features)
        self._weight_param = FuzzySetParam(
            FuzzySet.ones(
                *size
            )
        )
    
    def forward(self, m: FuzzySet):

        return FuzzySet(
            torch.min(m.data * self._weight_param.data, dim=-2),
            is_batch=m.is_batch, multiple_variables=m.multiple_variables
        )
