import torch
import torch.nn as nn
from . import conversion


class BinaryWeightLoss(nn.Module):

    def __init__(self, to_binary: conversion.StepCrispConverter):
        """initialzier

        Args:
            linear (nn.Linear): Linear layer to optimize
            act_inverse (Reversible): The invertable activation of the layer
        """
        self._to_binary = to_binary

    def step(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor):

        # assessment, y, result = get_y_and_assessment(objective, x, t, result)
        # y = to_binary.forward(x)
        change = (y != t).type_as(y)
        if self._to_binary.same:
            loss = (self._to_binary.weight[None,None,:] * change) ** 2
        else:
            loss = (self._to_binary.weight[None,:,:] * change) ** 2

        # TODO: Reduce the loss
        return loss


class BinaryXLoss(nn.Module):

    def __init__(self, to_binary: conversion.StepCrispConverter):
        """initialzier

        Args:
            linear (nn.Linear): Linear layer to optimize
            act_inverse (Reversible): The invertable activation of the layer
        """
        self._to_binary = to_binary

    def step(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor):

        # assessment, y, result = get_y_and_assessment(objective, x, t, result)
        # y = to_binary.forward(x)
        change = (y != t).type_as(y)
        loss = (x[:,:,None] * change) ** 2

        # TODO: Reduce the loss
        return loss
