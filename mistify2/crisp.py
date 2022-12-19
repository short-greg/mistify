import typing
import torch
import torch.nn as nn
import typing
from abc import abstractmethod
from .base import Set, SetParam
from .utils import reduce, get_comp_weight_size


class CrispSet(Set):

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
    
    def convert_variables(self, *size_after: int):
        
        if self._is_batch:
            return CrispSet(
                self._data.view(self._data.size(0), *size_after, -1), True
            )
        return self.__class__(
            self._data.view(*size_after, -1), False
        )

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

    def transpose(self, first_dim, second_dim) -> 'Set':
        assert self._value_size is not None
        return CrispSet(self._data.transpose(first_dim, second_dim), self._is_batch)


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

    def prepare_inputs(self, m: CrispSet) -> torch.Tensor:
        if self._complement_inputs:
            return torch.cat([m.data, 1 - m.data], dim=-1).unsqueeze(-1)
        return m.data.unsqueeze(-1)
    
    @property
    def to_complement(self) -> bool:
        return self._complement_inputs

    def forward(self, m: CrispSet):
        print(self.prepare_inputs(m).unsqueeze(-1).size())
        print(self.weight.data[None].size())
        return CrispSet(torch.max(
            torch.min(self.prepare_inputs(m), self.weight.data[None]), dim=-2
        )[0], True)


class CrispSetParam(SetParam):

    def __init__(self, set_: typing.Union[CrispSet, torch.Tensor], requires_grad: bool=True):

        if isinstance(set_, torch.Tensor):
            set_ = CrispSet(set_)
        super().__init__(set_, requires_grad=requires_grad)
