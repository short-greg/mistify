import torch
from ._functional import triangle, trapezoid, TENSOR_FLOAT
from functools import partial
from ..utils import reduce_as


class SignSTE(torch.autograd.Function):
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


class BinarySTE(torch.autograd.Function):
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


class RampSTE(torch.autograd.Function):
    """Use to clip the grad between two values
    Useful for smooth maximum/smooth minimum
    """

    @staticmethod
    def forward(ctx, x, min=None, max=None, clip=1.0):
        """
        Forward pass of the Binary Step function.
        """
        y = torch.clamp(x, min, max)
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

        if ctx.clip is not None:
            return grad_input, None, None, None

        if ctx.min is not None and ctx.max is not None:
            x_range = (x < ctx.min) | (x > ctx.max)
        elif ctx.min is not None:
            x_range = (x < ctx.min)
        elif ctx.max is not None:
            x_range = (x > ctx.max)
        else: x_range = None

        if x_range is not None:
            grad_input[x_range].clamp_(-ctx.clip, ctx.clip)
        return grad_input, None, None, None


class MaxSTE(torch.autograd.Function):
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
        x1_grad[condition] = (x1 - t)
        x1_grad = reduce_as(x1_grad, x1)
        
        x2_grad = grad_output.clone()
        condition = (x2 > t) | (x2 >= x1)
        x2_grad[condition] = x2 - t
        x2_grad = reduce_as(x2_grad, x2)

        return x1_grad, x2_grad, None


class MinSTE(torch.autograd.Function):
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
        x1_grad[condition] = (x1 - t)
        x1_grad = reduce_as(x1_grad, x1)
        
        x2_grad = grad_output.clone()
        condition = (x2 < t) | (x2 <= x1)
        x2_grad[condition] = x2 - t
        x2_grad = reduce_as(x2_grad, x2)

        return x1_grad, x2_grad


class MaxOnSTE(torch.autograd.Function):
    """Use to clip the grad between two values
    Useful for smooth maximum/smooth minimum
    """

    @staticmethod
    def forward(ctx, x, dim: int=-1, keepdim: bool=False):
        """
        Forward pass of the Max function
        """
        y = torch.max(x, dim, keepdim)
        ctx.save_for_backward(x, y)
        ctx.keepdim = keepdim
        ctx.dim = dim
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass of the Binary Step function using the Straight-Through Estimator.
        """
        
        t = y - grad_output
        x, y = ctx.saved_tensors
        if not ctx.keepdim:
            grad_output = grad_output.unsqueeze(ctx.dim)
            y = y.unsqueeze(ctx.dim)
        condition = (x < t) & (x < y)
        r = [1] * len(x)
        r[ctx.dim] = x.size(ctx.dim)
        grad_input = grad_output.repeat(r)
        grad_input[condition] = 0.0
        return grad_input


class MinOnSTE(torch.autograd.Function):
    """Use to clip the grad between two values
    Useful for smooth maximum/smooth minimum
    """

    @staticmethod
    def forward(ctx, x, dim: int=-1, keepdim: bool=False):
        """
        Forward pass of the Max function
        """
        y = torch.min(x, dim, keepdim)
        ctx.save_for_backward(x, y)
        ctx.keepdim = keepdim
        ctx.dim = dim
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass of the Binary Step function using the Straight-Through Estimator.
        """
        
        t = y - grad_output
        x, y = ctx.saved_tensors
        if not ctx.keepdim:
            grad_output = grad_output.unsqueeze(ctx.dim)
            y = y.unsqueeze(ctx.dim)
        condition = (x > t) & (x > y)
        r = [1] * len(x)
        r[ctx.dim] = x.size(ctx.dim)
        grad_input = grad_output.repeat(r)
        grad_input[condition] = 0.0
        return grad_input


def _triangle_post_hook(grad, state):
    
    state['grad'] = grad
    return grad

def _triangle_pre_hook(grad, x, left, right, clip, state):
    
    out_grad = state['grad']
    oob = (x < left) | (x > right)
    grad = grad.clone()

    if clip is not None:
        grad[oob] = out_grad[oob].clamp(-clip, clip)
    else:
        grad[oob] = out_grad[oob]
    return grad


def triangle_ste(
    x: torch.Tensor, left: TENSOR_FLOAT, mid: TENSOR_FLOAT, 
    right: TENSOR_FLOAT, height: TENSOR_FLOAT=1., 
    grad_clip: float=None
) -> torch.Tensor:

    state = {}
    if grad_clip != 0.0:
        x.register_hook(partial(_triangle_pre_hook, x=x, left=left, right=right, state=state, clip=grad_clip))
    y = triangle(x, left, mid, right, height)
    if grad_clip != 0.0:
        y.register_hook(partial(_triangle_post_hook, x=x, state=state))
    return y


def isosceles_ste(
    x: torch.Tensor, left: TENSOR_FLOAT, mid: TENSOR_FLOAT, 
    height: TENSOR_FLOAT=1., 
    grad_clip: float=None
) -> torch.Tensor:
    
    dl = mid - left
    right = mid + dl
    return trapezoid_ste(
        x, left, mid, right, height, grad_clip
    )


def _trap_post_hook(grad, state):
    
    state['grad'] = grad
    return grad


def _trap_pre_hook(grad, x, left, mid1, mid2, right, clip, state):
    
    out_grad = state['grad']
    oob = (x < left) | (x > right) | ( (x > mid1) & (x < mid2))
    grad = grad.clone()

    if clip is not None:
        grad[oob] = out_grad[oob].clamp(-clip, clip)
    else:
        grad[oob] = out_grad[oob]
    return grad


def trapezoid_ste(
    x: torch.Tensor, left: TENSOR_FLOAT, mid1: TENSOR_FLOAT, 
    mid2: TENSOR_FLOAT, right: TENSOR_FLOAT, height: TENSOR_FLOAT=1., 
    grad_clip: float=0.0
) -> torch.Tensor:

    state = {}
    if grad_clip != 0.0:
        x.register_hook(partial(_trap_pre_hook, x=x, left=left, mid1=mid1, mid2=mid2, right=right, state=state))
    y = trapezoid(x, left, mid1, mid2, right, height)

    if grad_clip != 0.0:
        y.register_hook(partial(_trap_post_hook, x=x, state=state))
    return y


def isosceles_trapezoid_ste(
    x: torch.Tensor, left: TENSOR_FLOAT, mid1: TENSOR_FLOAT, 
    mid2: TENSOR_FLOAT, height: TENSOR_FLOAT=1., 
    grad_clip: float=0.0
) -> torch.Tensor:
    
    dl = mid1 - left
    right = mid2 + dl
    return trapezoid_ste(
        x, left, mid1, mid2, right, height, grad_clip
    )


def binary_ste(x: torch.Tensor, clip: float=1.0) -> torch.Tensor:
    """Convenience function to use the straight through estimator for binary

    Args:
        x1 (torch.Tensor): First tensor
        x2 (torch.Tensor): Second tensor

    Returns:
        torch.Tensor: The binarized tensor
    """
    return BinarySTE.apply(x, clip=clip)


def sign_ste(x: torch.Tensor, clip: float=1.0) -> torch.Tensor:
    """Convenience function to use the straight through estimator for sign

    Args:
        x1 (torch.Tensor): First tensor

    Returns:
        torch.Tensor: The signed tensor
    """
    return SignSTE.apply(x, clip=clip)


def ramp_ste(x: torch.Tensor, clip: float=1.0) -> torch.Tensor:
    """Convenience function to use the straight through estimator for ramp

    Args:
        x1 (torch.Tensor): First tensor

    Returns:
        torch.Tensor: The signed tensor
    """
    return RampSTE.apply(x, clip=clip)


def max_ste(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Convenience function to use the straight through estimator for max

    Args:
        x1 (torch.Tensor): First tensor
        x2 (torch.Tensor): Second tensor

    Returns:
        torch.Tensor: The max tensor
    """

    return MaxSTE.apply(x1, x2)


def min_ste(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Convenience function to use the straight through estimator for min

    Args:
        x1 (torch.Tensor): First tensor
        x2 (torch.Tensor): Second tensor

    Returns:
        torch.Tensor: The min tensor
    """
    return MinSTE.apply(x1, x2)


def max_on_ste(x: torch.Tensor, dim: int=-1, keepdim: bool=False) -> torch.Tensor:
    """Convenience function to use the straight through estimator for max

    Args:
        x (torch.Tensor): The input
        dim (int, optional): The dimension. Defaults to -1.
        keepdim (bool, optional): Whether to keep the dimension. Defaults to False.

    Returns:
        torch.Tensor: The max
    """
    return MaxOnSTE.apply(x, dim, keepdim)


def min_on_ste(x: torch.Tensor, dim: int=-1, keepdim: bool=False) -> torch.Tensor:
    """Convenience function to use the straight through estimator for min

    Args:
        x (torch.Tensor): The input
        dim (int, optional): The dimension. Defaults to -1.
        keepdim (bool, optional): Whether to keep the dimension. Defaults to False.

    Returns:
        torch.Tensor: The min
    """
    return MinOnSTE.apply(x, dim, keepdim)


def bounded_min_ste(x1: torch.Tensor, x2: torch.Tensor, clip: float=1.0) -> torch.Tensor:
    """Convenience function to use the straight through estimator for bounded min

    Args:
        x1 (torch.Tensor): First tensor
        x2 (torch.Tensor): Second tensor

    Returns:
        torch.Tensor: The max tensor
    """
    return RampSTE.apply(
        x1 + x2 - 1, max=torch.tensor(0.0, dtype=x1.dtype, device=x1.device),
        clip=clip
    )


def bounded_max_ste(x1: torch.Tensor, x2: torch.Tensor, clip: float=1.0) -> torch.Tensor:
    """Convenience function to use the straight through estimator for bounded max

    Args:
        x1 (torch.Tensor): First tensor
        x2 (torch.Tensor): Second tensor

    Returns:
        torch.Tensor: The max tensor
    """
    return RampSTE.apply(
        x1 + x2, min=torch.tensor(1.0, dtype=x1.dtype, device=x1.device),
        clip=clip
    )


def bounded_max_on_ste(m: torch.Tensor, dim=-1, keepdim: bool=False, clip: float=1.0) -> torch.Tensor:
    """Take the bounded max on a given dimension

    Args:
        x (torch.Tensor): Tensor to take the bounded max of
        dim (int, optional): The dimension to take the bounded max on. Defaults to -1.
        keepdim (bool, optional): Whether to keep the dim. Defaults to False.

    Returns:
        torch.Tensor: The bounded max
    """
    return RampSTE.apply(
        m.sum(dim=dim, keepdim=keepdim),
        max=torch.tensor(1.0, device=m.device, dtype=m.dtype),
        clip=clip
    )


def bounded_min_on_ste(x: torch.Tensor, dim=-1, keepdim: bool=False, clip: float=1.0) -> torch.Tensor:
    """Take the bounded min on a given dimension

    Args:
        x (torch.Tensor): Tensor to take the bounded min of
        dim (int, optional): The dimension to take the bounded min on. Defaults to -1.
        keepdim (bool, optional): Whether to keep the dim. Defaults to False.

    Returns:
        torch.Tensor: The bounded min
    """
    return RampSTE.apply(
        x.sum(dim=dim, keepdim=keepdim) 
        - x.size(dim) + 1, 
        min=torch.tensor(0.0, device=x.device, dtype=x.dtype),
        clip=clip
    )
