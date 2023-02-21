import typing
import torch
import torch.nn as nn
import typing
from .base import CompositionBase, maxmin, ComplementBase
from .utils import get_comp_weight_size

# Add in TernarySet as a type of crisp set
# with


def differ(m1: torch.Tensor, m2: 'torch.Tensor') -> 'torch.Tensor':
    return (m1 - m2).clamp(-1.0, 1.0)

def unify(m1: torch.Tensor, m2: 'torch.Tensor') -> 'torch.Tensor':
    return torch.max(m1, m2)

def intersect(m1: torch.Tensor, m2: 'torch.Tensor') -> 'torch.Tensor':
    return torch.min(m1, m2)

def inclusion(m1: torch.Tensor, m2: 'torch.Tensor') -> 'torch.Tensor':
    return (1 - m2) + torch.min(m1, m2)

def exclusion(m1: torch.Tensor, m2: 'torch.Tensor') -> 'torch.Tensor':
    return (1 - m1) + torch.min(m1, m2)


def negatives(*size: int, dtype=torch.float32, device='cpu'):

    return -torch.ones(*size, dtype=dtype, device=device)

def positives(*size: int, dtype=torch.float32, device='cpu'):

    return torch.ones(*size, dtype=dtype, device=device)


def unknowns(*size: int, dtype=torch.float32, device='cpu'):

    return torch.zeros(*size, dtype=dtype, device=device)


def rand(*size: int, dtype=torch.float32, device='cpu'):

    return ((torch.rand(*size, device=device, dtype=dtype)) * 3).floor() - 1


class TernaryComposition(CompositionBase):

    def init_weight(self, in_features: int, out_features: int, in_variables: int = None) -> torch.Tensor:
        return positives(get_comp_weight_size(in_features, out_features, in_variables))

    def forward(self, m: torch.Tensor):
        return maxmin(m, self.weight.data[None]).round()


class TernaryComplement(ComplementBase):

    def complement(self, m: torch.Tensor):
        return -m


# class TernarySet(Set):



#     def __sub__(self, other: 'TernarySet') -> 'TernarySet':
#         return self.differ(other)

#     def __mul__(self, other: 'TernarySet') -> 'TernarySet':
#         return self.intersect(other)

#     def __add__(self, other: 'TernarySet') -> 'TernarySet':
#         return self.unify(other)
    
#     def __getitem__(self, idx):
#         return self.data[idx]
    
#     def convert_variables(self, *size_after: int):
        
#         if self._is_batch:
#             return TernarySet(
#                 self._data.reshape(self._data.size(0), *size_after, -1), True
#             )
#         return self.__class__(
#             self._data.reshape(*size_after, -1), False
#         )

#     def reshape(self, *size: int):
#         return BinarySet(
#             self.data.reshape(*size), self.is_batch
#         )



#     def transpose(self, first_dim, second_dim) -> 'TernarySet':
#         assert self._value_size is not None
#         return TernarySet(self._data.transpose(first_dim, second_dim), self._is_batch)

#     @property
#     def shape(self) -> torch.Size:
#         return self._data.shape

