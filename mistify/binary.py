import torch

from .core import CompositionBase, maxmin, ComplementBase
from .utils import get_comp_weight_size


def rand(*size: int, dtype=torch.float32, device='cpu'):

    return (torch.rand(*size, device=device, dtype=dtype)).round()

def negatives(*size: int, dtype=torch.float32, device='cpu'):

    return torch.zeros(*size, dtype=dtype, device=device)

def positives(*size: int, dtype=torch.float32, device='cpu'):

    return torch.ones(*size, dtype=dtype, device=device)

def differ(m1: torch.Tensor, m2: torch.Tensor) -> 'torch.Tensor':
    return (m1 - m2).clamp(0.0, 1.0)

def unify(m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
    return torch.max(m1, m2)

def intersect(m1: torch.Tensor, m2: torch.Tensor) -> 'torch.Tensor':
    return torch.min(m1, m2)

def inclusion(m1: torch.Tensor, m2: torch.Tensor) -> 'torch.Tensor':
    return (1 - m2) + torch.min(m1, m2)

def exclusion(m1: torch.Tensor, m2: torch.Tensor) -> 'torch.Tensor':
    return (1 - m1) + torch.min(m1, m2)


class BinaryComposition(CompositionBase):

    def init_weight(self, in_features: int, out_features: int, in_variables: int = None) -> torch.Tensor:
        return positives(get_comp_weight_size(in_features, out_features, in_variables))

    def forward(self, m: torch.Tensor):
        return maxmin(m, self.weight).round()


class BinaryComplement(ComplementBase):

    def complement(self, m: torch.Tensor):
        return 1 - m


# class BinarySet(Set):


#     def __sub__(self, other: 'BinarySet') -> 'BinarySet':
#         return self.differ(other)

#     def __mul__(self, other: 'BinarySet') -> 'BinarySet':
#         return self.intersect(other)

#     def __add__(self, other: 'BinarySet') -> 'BinarySet':
#         return self.unify(other)
    
#     def __getitem__(self, idx):
#         return self.data[idx]
    
#     def convert_variables(self, *size_after: int):
        
#         if self._is_batch:
#             return BinarySet(
#                 self._data.view(self._data.size(0), *size_after, -1), True
#             )
#         return self.__class__(
#             self._data.view(*size_after, -1), False
#         )

#     @classmethod
#     def negatives(cls, *size: int, is_batch: bool=None, dtype=torch.float32, device='cpu'):

#         return BinarySet(
#             torch.zeros(*size, dtype=dtype, device=device), is_batch
#         )
    
#     @classmethod
#     def positives(cls, *size: int, dtype=torch.float32, is_batch: bool=None, device='cpu'):

#         return BinarySet(
#             torch.ones(*size, dtype=dtype, device=device), 
#             is_batch
#         )

#     def reshape(self, *size: int):
#         return BinarySet(
#             self.data.reshape(*size), self.is_batch
#         )

#     @classmethod
#     def rand(cls, *size: int, is_batch: bool=None, dtype=torch.float32, device='cpu'):

#         return BinarySet(
#             (torch.rand(*size, device=device)).round(), 
#             is_batch
#         )

#     def transpose(self, first_dim, second_dim) -> 'BinarySet':
#         assert self._value_size is not None
#         return BinarySet(self._data.transpose(first_dim, second_dim), self._is_batch)

#     @property
#     def shape(self) -> torch.Size:
#         return self._data.shape

