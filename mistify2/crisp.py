import typing
import torch
import torch.nn as nn
import typing
from abc import abstractmethod
from .base import ISet
from .utils import reduce, get_comp_weight_size


class CrispSet(ISet):

    def __init__(self, data: torch.Tensor, is_batch: bool=None):

        if is_batch is None:
            is_batch = False if data.dim() <= 1 else True

        self._data = data
        self._n_samples = None
        self._n_variables = None
        if is_batch and data.dim() == 1:
            raise ValueError(f'Is batch cannot be set to true if data dimensionality is 1')
        
        if is_batch: 
            self._value_size = None if data.dim() == 2 else data.shape[1:-1]
        else:
            self._value_size = None if data.dim() == 1 else data.shape[1:-1]

        self._is_batch = is_batch
        self._n_values = data.shape[-1]
    
    @property
    def data(self) -> torch.Tensor:
        return self._data

    def dim(self) -> int:
        return self.data.dim()

    def transpose(self, first_dim, second_dim) -> 'CrispSet':
        assert self._value_size is not None
        return CrispSet(self._data.transpose(first_dim, second_dim), self._is_batch)

    def differ(self, other: 'CrispSet') -> 'CrispSet':
        return CrispSet((self.data - other._data).clamp(0.0, 1.0))
    
    def unify(self, other: 'CrispSet') -> 'CrispSet':
        return CrispSet(torch.max(self.data, other.data))

    def intersect(self, other: 'CrispSet') -> 'CrispSet':
        return CrispSet(torch.min(self.data, other.data))

    def inclusion(self, other: 'CrispSet') -> 'CrispSet':
        return CrispSet(
            (1 - other.data) + torch.min(self.data, other.data), self._is_batch
        )

    def exclusion(self, other: 'CrispSet') -> 'CrispSet':
        return CrispSet(
            (1 - self.data) + torch.min(self.data, other.data), self._is_batch
        )

    def __sub__(self, other: 'CrispSet') -> 'CrispSet':
        return self.differ(other)

    def __mul__(self, other: 'CrispSet') -> 'CrispSet':
        return self.intersect(other)

    def __add__(self, other: 'CrispSet') -> 'CrispSet':
        return self.unify(other)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    @classmethod
    def zeros(cls, *size: int, is_batch: bool=None, dtype=torch.float32, device='cpu'):

        return CrispSet(
            torch.zeros(*size, dtype=dtype, device=device), is_batch
        )

    @classmethod
    def ones(cls, *size: int, dtype=torch.float32, is_batch: bool=None, device='cpu'):

        return CrispSet(
            torch.ones(*size, dtype=dtype, device=device), 
            is_batch
        )

    @classmethod
    def rand(cls, *size: int, is_batch: bool=None, dtype=torch.float32, device='cpu'):

        return CrispSet(
            (torch.rand(*size, device=device) > 0.5).type(dtype), 
            is_batch
        )


class CrispComposition(nn.Module):

    def __init__(
        self, in_features: int, out_features: int, 
        complement_inputs: bool=False, in_variables: int=None
    ):
        super().__init__()
        self._in_features = in_features
        self._out_features = out_features
        self._complement_inputs = complement_inputs
        if complement_inputs:
            in_features *= 2
        self._multiple_variables = in_variables is not None
        # store weights as values between 0 and 1
        self.weight = CrispSetParam(CrispSet.ones(get_comp_weight_size(in_features, out_features, in_variables)))
        # self._weight = nn.parameter.Parameter(
        #     torch.ones(get_comp_weight_size(in_features, out_features, in_variables))
        # )

    @property
    def to_complement(self) -> bool:
        return self._complement_inputs

    def forward(self, m: CrispSet):
        if self._complement_inputs:
            m = torch.cat([m.data, 1 - m.data], dim=-1)

        return CrispSet(torch.max(
            torch.min(m.data.unsqueeze(-1), self.weight.data[None]), dim=-1
        )[0], True)


class CrispSetParam(nn.Module):

    def __init__(self, crisp_set: typing.Union[CrispSet, torch.Tensor], requires_grad: bool=True):
        super().__init__()
        if isinstance(crisp_set, torch.Tensor):
            crisp_set= CrispSet(crisp_set, False)
        self._crisp_set = crisp_set
        self._data = nn.parameter.Parameter(
            crisp_set.data, requires_grad=requires_grad
        )

    @property
    def data(self) -> torch.Tensor:
        return self._data

    @data.setter
    def data(self, data: torch.Tensor):
        self._crisp_set.data = data
        self._data = nn.parameter.Parameter(
            self._crisp_set.data.data, 
            requires_grad=self._data.requires_grad
        )

    @property
    def crisp_set(self) -> CrispSet:
        return self._crisp_set
    
    @crisp_set.setter
    def crisp_set(self, crisp_set: 'CrispSet'):
        self.data = crisp_set.data
        # self._crisp_set = crisp_set
        # self.data = crisp_set.data
    
