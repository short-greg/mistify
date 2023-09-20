"""
Functionality for crisp binary sets where 1 is True and 0 is False

"""

import torch

from .._base import CompositionBase, maxmin, ComplementBase,get_comp_weight_size
from torch import nn
from .utils import positives


class BinaryComposition(CompositionBase):

    def init_weight(self, in_features: int, out_features: int, in_variables: int = None) -> torch.Tensor:
        return positives(get_comp_weight_size(in_features, out_features, in_variables))

    def forward(self, m: torch.Tensor):
        return maxmin(m, self.weight).round()

    def clamp_weights(self):
        self.weight.data = self.weight.data.clamp(0, 1)


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
