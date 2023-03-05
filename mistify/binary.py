import torch

from ._core import CompositionBase, maxmin, ComplementBase,get_comp_weight_size
from torch import nn


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


# class BinaryWeightLoss(nn.Module):

#     def __init__(self, to_binary: conversion.StepCrispConverter):
#         """initialzier

#         Args:
#             linear (nn.Linear): Linear layer to optimize
#             act_inverse (Reversible): The invertable activation of the layer
#         """
#         self._to_binary = to_binary

#     def step(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor):

#         # assessment, y, result = get_y_and_assessment(objective, x, t, result)
#         # y = to_binary.forward(x)
#         change = (y != t).type_as(y)
#         if self._to_binary.same:
#             loss = (self._to_binary.weight[None,None,:] * change) ** 2
#         else:
#             loss = (self._to_binary.weight[None,:,:] * change) ** 2

#         # TODO: Reduce the loss
#         return loss


# class BinaryXLoss(nn.Module):

#     def __init__(self, to_binary: conversion.StepCrispConverter):
#         """initialzier

#         Args:
#             linear (nn.Linear): Linear layer to optimize
#             act_inverse (Reversible): The invertable activation of the layer
#         """
#         self._to_binary = to_binary

#     def step(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor):

#         # assessment, y, result = get_y_and_assessment(objective, x, t, result)
#         # y = to_binary.forward(x)
#         change = (y != t).type_as(y)
#         loss = (x[:,:,None] * change) ** 2

#         # TODO: Reduce the loss
#         return loss
