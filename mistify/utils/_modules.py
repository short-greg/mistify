# 1st party
from abc import ABC, abstractmethod
import typing

# 3rd party
import torch
from torch import nn
import pandas as pd


class Argmax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self._dim = dim

    def forward(self, x: torch.Tensor) -> torch.LongTensor:
        return torch.argmax(x, dim=-1)


class Sign(nn.Module):

    def __init__(self, grad: bool = True):
        super().__init__()
        self._grad = grad

    def forward(self, x: torch.Tensor):
        if self._grad:
            return SignSTE.apply(x)
        return torch.sign(x)


class Boolean(nn.Module):

    def __init__(self, grad: bool = True):
        super().__init__()
        self._grad = grad

    def forward(self, x: torch.Tensor):
        if self._grad:
            return BooleanSTE.apply(x)
        return torch.clamp(x, 0, 1).round()


class Clamp(nn.Module):

    def __init__(self, lower: float=-1.0, upper: float=1.0, grad: bool = True):
        super().__init__()
        self._lower = lower
        self._upper = upper
        self._grad = grad

    def forward(self, x: torch.Tensor):
        if self._grad:
            return ClampSTE.apply(x, self._lower, self._upper, -0.01, 0.01)
        return torch.clamp(x)


class SignSTE(torch.autograd.Function):
    """Use to clip the grad between two values
    Useful for smooth maximum/smooth minimum
    """

    @staticmethod
    def forward(ctx, x):
        """
        Forward pass of the Binary Step function.
        """
        ctx.save_for_backward(x)
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the Binary Step function using the Straight-Through Estimator.
        """
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # return grad_input.clamp(-1, 1)
        grad_input[(x < -1) | (x > 1)] = 0
        return grad_input


class BooleanSTE(torch.autograd.Function):
    """Use to clip the grad between two values
    Useful for smooth maximum/smooth minimum
    """

    @staticmethod
    def forward(ctx, x):
        """
        Forward pass of the Binary Step function.
        """
        ctx.save_for_backward(x)
        return torch.clamp(x, 0, 1).round()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the Binary Step function using the Straight-Through Estimator.
        """
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # return grad_input.clamp(-1, 1)
        grad_input[(x < -1) | (x > 1)] = 0
        return grad_input


class ClampSTE(torch.autograd.Function):
    """Use to clip the grad between two values
    Useful for smooth maximum/smooth minimum
    """

    @staticmethod
    def forward(ctx, x, lower: float=0, upper: float=1, backward_lower: float=-1, backward_upper: float=1):
        """
        Forward pass of the Binary Step function.
        """
        ctx.save_for_backward(x)
        ctx.lower = lower
        ctx.upper = upper
        ctx.backward_lower = backward_lower
        ctx.backward_upper = backward_upper
        return torch.clamp(x, lower, upper)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the Binary Step function using the Straight-Through Estimator.
        """
        grad_input = grad_output.clone()
        return grad_input.clamp(ctx.backward_lower, ctx.backward_upper), None, None, None, None


def clamp(x: torch.Tensor) -> torch.Tensor:
    return Clamp.apply(x)


def binary_ste(x: torch.Tensor) -> torch.Tensor:
    return BooleanSTE.apply(x)


def sign_ste(x: torch.Tensor) -> torch.Tensor:
    return SignSTE.apply(x)


class Dropout(nn.Module):
    """Dropout is designed to work with logical neurons
    It does not divide the output by p and can be set to "dropout" to
    any value. This is because for instance And neurons should
    dropout to 1 not to 0.
    """

    def __init__(self, p: float, val: float=0.0):
        """Create a dropout neuron to dropout logical inputs

        Args:
            p (float): The dropout probability
            val (float, optional): The value to dropout to. Defaults to 0.0.
        """

        super().__init__()
        self.p = p
        self.val = val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): the input

        Returns:
            torch.Tensor: the dropped out tensor
        """
        if self.training:
            x[(torch.rand_like(x) > self.p)] = self.val
        
        return x
