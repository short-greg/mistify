from ._m import TENSOR_FLOAT
from functools import partial
import torch


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


def triangle(
    x: torch.Tensor, left: TENSOR_FLOAT, mid: TENSOR_FLOAT, 
    right: TENSOR_FLOAT, height: TENSOR_FLOAT=1., g: bool=False,
    clip: float=None
) -> torch.Tensor:

    state = {}
    if clip != 0.0 and g is True:
        x.register_hook(partial(_triangle_pre_hook, x=x, left=left, right=right, state=state, clip=clip))

    left_val = height / (mid - left) * (x - left)
    right_val = -height / (right - mid) * (x - mid) + height
    
    right_side = x >= mid
    left_val[right_side] = right_val[right_side]
    if clip != 0.0 and g is True:
        left_val.register_hook(partial(_triangle_post_hook, x=x, state=state))
    return left_val


def isosceles(
    x: torch.Tensor, left: TENSOR_FLOAT, mid: TENSOR_FLOAT, height: TENSOR_FLOAT=1.,
    g: bool=False, clip: float=None
) -> torch.Tensor:

    dx = mid - left
    return triangle(x, left, mid, mid + dx, height, g, clip)


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


def trapezoid(
    x: torch.Tensor, left: TENSOR_FLOAT, mid1: TENSOR_FLOAT, 
    mid2: TENSOR_FLOAT, right: TENSOR_FLOAT, height: TENSOR_FLOAT=1., 
    g: bool=False, clip: float=None
) -> torch.Tensor:

    state = {}
    if clip != 0.0 and g is True:
        x.register_hook(partial(_trap_pre_hook, x=x, left=left, mid1=mid1, mid2=mid2, right=right, state=state))
    left_val = height / (mid1 - left) * (x - left)
    right_val = -height / (right - mid2) * (x - mid2) + height
    
    right_side = x >= mid2
    mid_val = (x >= mid1) & (x <= mid2)
    left_val[right_side] = right_val[right_side]
    left_val[mid_val] = height
    y = left_val

    if clip != 0.0 and g is True:
        y.register_hook(partial(_trap_post_hook, x=x, state=state))
    return y


def isosceles_trapezoid(
    x: torch.Tensor, left: TENSOR_FLOAT, mid1: TENSOR_FLOAT, 
    mid2: TENSOR_FLOAT, height: TENSOR_FLOAT=1., 
    grad_clip: float=0.0
) -> torch.Tensor:
    
    dl = mid1 - left
    right = mid2 + dl
    return trapezoid(
        x, left, mid1, mid2, right, height, grad_clip
    )

