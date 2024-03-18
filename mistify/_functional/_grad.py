# 1st party
import typing

# 3rd party
import torch

import torch
from ..utils import reduce_as


class SignG(torch.autograd.Function):
    """Use to clip the grad between two values
    Useful for smooth maximum/smooth minimum
    """

    @staticmethod
    def forward(ctx, x, clip: float=1.0):
        """
        Forward pass of the Binary Step function.
        """
        ctx.clip = clip
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the Binary Step function using the Straight-Through Estimator.
        """
        grad_input = grad_output.clone()
        if ctx.clip is None:
            return grad_input, None
        
        return grad_input.clamp(-ctx.clip, ctx.clip), None


class BinaryG(torch.autograd.Function):
    """Use to clip the grad between two values
    Useful for smooth maximum/smooth minimum
    """

    @staticmethod
    def forward(ctx, x, clip: float=1.0):
        """
        Forward pass of the Binary Step function.
        """
        ctx.clip = clip
        return torch.clamp(x, 0, 1).round()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the Binary Step function using the Straight-Through Estimator.
        """
        grad_input = grad_output.clone()
        if ctx.clip is None:
            return grad_input, None
        return grad_input.clamp(-ctx.clip, ctx.clip), None


class ClampG(torch.autograd.Function):
    """Use to clip the grad between two values
    Useful for smooth maximum/smooth minimum
    """

    @staticmethod
    def forward(ctx, x, min=None, max=None, clip=1.0):
        """
        Forward pass of the Binary Step function.
        """
        if min is not None or max is not None:
            y = torch.clamp(x, min, max)
        else:
            y = x
        ctx.save_for_backward(x)
        ctx.min = min
        ctx.max = max
        ctx.clip = clip
        return y

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the Binary Step function using the Straight-Through Estimator.
        """
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()

        if ctx.clip == 0.0:
            return grad_input, None, None, None

        if ctx.min is not None and ctx.max is not None:
            x_range = (x < ctx.min) | (x > ctx.max)
        elif ctx.min is not None:
            x_range = (x < ctx.min)
        elif ctx.max is not None:
            x_range = (x > ctx.max)
        else: x_range = None

        if x_range is not None:
            grad_input[x_range] = grad_input[x_range].clamp(-ctx.clip, ctx.clip)
        return grad_input, None, None, None


class MaxG(torch.autograd.Function):
    """Use to clip the grad between two values
    Useful for smooth maximum/smooth minimum
    """

    @staticmethod
    def forward(ctx, x1, x2):
        """
        Forward pass of the Max function
        """
        y = torch.max(x1, x2)
        ctx.save_for_backward(x1, x2, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the Binary Step function using the Straight-Through Estimator.
        """
        x1, x2, y = ctx.saved_tensors
        grad_input = grad_output.clone()
        t = y - grad_input

        x1_grad = torch.zeros_like(grad_output)
        x2_grad = torch.zeros_like(grad_output)

        condition = (x1 > t) | (x1 >= x2)
        x1_grad[condition] = (x1 - t)[condition]
        x1_grad = reduce_as(x1_grad, x1)
        
        x2_grad = grad_output.clone()
        condition = (x2 > t) | (x2 >= x1)
        x2_grad[condition] = (x2 - t)[condition]
        x2_grad = reduce_as(x2_grad, x2)

        return x1_grad, x2_grad, None


class MinG(torch.autograd.Function):
    """Use to clip the grad between two values
    Useful for smooth maximum/smooth minimum
    """

    @staticmethod
    def forward(ctx, x1, x2):
        """
        Forward pass of the Max function
        """
        y = torch.min(x1, x2)
        ctx.save_for_backward(x1, x2, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the Binary Step function using the Straight-Through Estimator.
        """
        x1, x2, y = ctx.saved_tensors
        grad_input = grad_output.clone()
        t = y - grad_input

        x1_grad = torch.zeros_like(grad_output)
        x2_grad = torch.zeros_like(grad_output)

        condition = (x1 < t) | (x1 <= x2)
        x1_grad[condition] = (x1 - t)[condition]
        x1_grad = reduce_as(x1_grad, x1)
        
        x2_grad = grad_output.clone()
        condition = (x2 < t) | (x2 <= x1)
        x2_grad[condition] = (x2 - t)[condition]
        x2_grad = reduce_as(x2_grad, x2)

        return x1_grad, x2_grad


class MaxOnG(torch.autograd.Function):
    """Use to clip the grad between two values
    Useful for smooth maximum/smooth minimum
    """

    @staticmethod
    def forward(ctx, x, dim: int=-1, keepdim: bool=False):
        """
        Forward pass of the Max function
        """
        y = torch.max(x, dim, keepdim)
        ctx.save_for_backward(x, y[0])
        ctx.keepdim = keepdim
        ctx.dim = dim
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, ind: torch.Tensor):
        """
        Backward pass of the Binary Step function using the Straight-Through Estimator.
        """
        
        x, y = ctx.saved_tensors
        t = y - grad_output
        if not ctx.keepdim:
            grad_output = grad_output.unsqueeze(ctx.dim)
            y = y.unsqueeze(ctx.dim)
            t = t.unsqueeze(ctx.dim)
        condition = (x < t) & (x < y)
        r = [1] * x.dim()
        r[ctx.dim] = x.size(ctx.dim)
        grad_input = grad_output.repeat(r)
        grad_input[condition] = 0.0
        return grad_input, None, None


class MinOnG(torch.autograd.Function):
    """Use to clip the grad between two values
    Useful for smooth maximum/smooth minimum
    """

    @staticmethod
    def forward(ctx, x, dim: int=-1, keepdim: bool=False):
        """
        Forward pass of the Max function
        """
        y = torch.min(x, dim, keepdim)
        ctx.save_for_backward(x, y[0])
        ctx.keepdim = keepdim
        ctx.dim = dim
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, ind: torch.Tensor):
        """
        Backward pass of the Binary Step function using the Straight-Through Estimator.
        """
        
        x, y = ctx.saved_tensors
        t = y - grad_output
        if not ctx.keepdim:
            grad_output = grad_output.unsqueeze(ctx.dim)
            y = y.unsqueeze(ctx.dim)
            t = t.unsqueeze(ctx.dim)
        condition = (x > t) & (x > y)
        r = [1] * x.dim()
        r[ctx.dim] = x.size(ctx.dim)
        grad_input = grad_output.repeat(r)
        grad_input[condition] = 0.0
        return grad_input, None, None
