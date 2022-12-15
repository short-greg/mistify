import typing
import torch
import torch.nn as nn
import typing
from abc import abstractmethod
from .base import ISet
from utils import reduce, get_comp_weight_size


class CrispSet(ISet):

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
    def data(self) -> torch.Tensor:
        return self._data

    def swap_variables(self) -> 'CrispSet':
        assert self._multiple_variables
        if self._is_batch:
            dim1, dim2 = 1, 2
        else:
            dim1, dim2 = 0, 1
        return CrispSet(self._data.transpose(dim2, dim1), self._is_batch, True)

    def differ(self, other: 'CrispSet') -> 'CrispSet':
        return CrispSet((self.data - other._data).clamp(0.0, 1.0))
    
    def unify(self, other: 'CrispSet') -> 'CrispSet':
        return CrispSet(torch.max(self.data, other.data))

    def intersect(self, other: 'CrispSet') -> 'CrispSet':
        return CrispSet(torch.min(self.data, other.data))

    def inclusion(self, other: 'CrispSet') -> 'CrispSet':
        return CrispSet(
            (1 - self.data) + torch.min(self.data, other.data), self._is_batch, self._multiple_variables
        )

    def exclusion(self, other: 'CrispSet') -> 'CrispSet':
        return CrispSet(
            (1 - other.data) + torch.min(self.data, other.data), self._is_batch, self._multiple_variables
        )

    def exclusion(self, other: 'CrispSet') -> 'CrispSet':
        # TODO: WRITE
        return CrispSet(
            torch.clamp((1 - self.data) - torch.min(self.data, other.data), 0, 1)
        )

    def __sub__(self, other: 'CrispSet'):
        return self.differ(other)

    def __mul__(self, other: 'CrispSet'):
        return self.intersect(self, other)

    def __add__(self, other: 'CrispSet'):
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
        return CrispSet(
            torch.zeros(*size, dtype=dtype, device=device), 
            batch_size is not None, n_variables is not None
        )

    @classmethod
    def ones(cls, n_features: int, batch_size: int=None, n_variables: int=None, dtype=torch.float32, device='cpu'):

        size = cls.get_size(n_features, batch_size, n_variables)
        return CrispSet(
            torch.ones(*size, dtype=dtype, device=device), 
            batch_size is not None, n_variables is not None
        )

    @classmethod
    def rand(cls, n_features: int, batch_size: int=None, n_variables: int=None, dtype=torch.float32, device='cpu'):

        size = cls.get_size(n_features, batch_size, n_variables)
        return CrispSet(
            (torch.rand(*size, device=device) > 0.5).type(dtype), 
            batch_size is not None, n_variables is not None
        )

class CrispComposition(nn.Module):

    def __init__(
        self, in_features: int, out_features: int, 
        complement_inputs: bool=False, in_variables: int=None
    ):
        super().__init__()
        self._in_features = in_features
        self._out_features = out_features
        self._in
        self._complement_inputs = complement_inputs
        if complement_inputs:
            in_features *= 2
        self._multiple_variables = in_variables is not None
        # store weights as values between 0 and 1
        self.weight = nn.parameter.Parameter(
            torch.ones(get_comp_weight_size(in_features, out_features, in_variables))
        )

    @property
    def to_complement(self) -> bool:
        return self._complement_inputs

    def forward(self, m: torch.Tensor):
        # assume inputs are binary
        # binarize the weights
        if self._complement_inputs:
            m = torch.cat([m, 1 - m], dim=1)
        return CrispSet(torch.max(
            torch.min(m[:,:,None], self.weight[None]), dim=-2
        )[0], True, self._multiple_variables)


class CrispSetParam(nn.parameter.Parameter):

    def __init__(self, crisp_set: CrispSet, requires_grad: bool=True):

        super().__init__(crisp_set.data, requires_grad=True)
        self._crisp_set = crisp_set

    @property
    def data(self) -> torch.Tensor:
        return self._crisp_set.data

    @data.setter
    def data(self, data: torch.Tensor):
        assert data.size() == self._crisp_set.data.size()
        super().data = data
        self._crisp_set.data = data

    @property
    def crisp_set(self) -> CrispSet:
        return self._crisp_set
    
    @crisp_set.setter
    def crisp_set(self, crisp_set: 'CrispSet'):
        self._crisp_set = crisp_set
        self.data = crisp_set.data
    
