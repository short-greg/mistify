from ._m import TENSOR_FLOAT
from functools import partial
import torch


def _triangle_post_hook(grad, state):
    
    state['grad'] = grad
    return grad


def _triangle_pre_hook(grad, x, left, right, clip, state):
    
    out_grad = state['grad']
    if right is not None:
        oob = (x < left) | (x > right)
    else:
        oob = (x < left)
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
    """The right triangle function

    Args:
        x (torch.Tensor): The input
        left (TENSOR_FLOAT): The leftmost point
        mid (TENSOR_FLOAT): The midpoint
        right (TENSOR_FLOAT): The rightmost point
        increasing (bool, optional): Whether the triangle is increasing. Defaults to False.
        height (TENSOR_FLOAT, optional): The height of the triangle. Defaults to 1..
        g (bool, optional): WHether to use a straight-through estimator. Defaults to False.
        clip (float, optional): Whether to clip if g is True. Defaults to None.

    Returns:
        torch.Tensor: The output
    """
    state = {}
    x_base = x
    x = x_base.clone()
    if clip != 0.0 and g is True:
        x.register_hook(partial(_triangle_pre_hook, x=x_base, left=left, right=right, state=state, clip=clip))

    left_val = height / (mid - left) * (x - left)
    right_val = -height / (right - mid) * (x - mid) + height
    
    right_side = x >= mid
    left_val[right_side] = right_val[right_side]
    if clip != 0.0 and g is True:
        left_val.register_hook(partial(_triangle_post_hook, state=state))
    return left_val


def right_triangle(
    x: torch.Tensor, left: TENSOR_FLOAT, mid: TENSOR_FLOAT, 
    increasing: bool=False, height: TENSOR_FLOAT=1., g: bool=False,
    clip: float=None
) -> torch.Tensor:
    """The right triangle function

    Args:
        x (torch.Tensor): The input
        left (TENSOR_FLOAT): The leftmost point
        mid (TENSOR_FLOAT): The midpoint
        increasing (bool, optional): Whether the triangle is increasing. Defaults to False.
        height (TENSOR_FLOAT, optional): The height of the triangle. Defaults to 1..
        g (bool, optional): WHether to use a straight-through estimator. Defaults to False.
        clip (float, optional): Whether to clip if g is True. Defaults to None.

    Returns:
        torch.Tensor: The output
    """

    state = {}
    x_base = x
    x = x_base.clone()
    if clip != 0.0 and g is True:
        x.register_hook(partial(_triangle_pre_hook, x=x_base, left=left, right=None, state=state, clip=clip))

    if increasing:
        val = height / (mid - left) * (x - left)
    else:
        val = -height / (mid - left) * (x - left) + height
    
    if clip != 0.0 and g is True:
        val.register_hook(partial(_triangle_post_hook, state=state))
    return val



def triangle_area(
    left: TENSOR_FLOAT, 
    right: TENSOR_FLOAT, height: TENSOR_FLOAT=1.
) -> torch.Tensor:
    """Calculate the area of the triangle shape

    Args:
        left (TENSOR_FLOAT): The leftmost point
        right (TENSOR_FLOAT): The midpoint
        height (TENSOR_FLOAT, optional): The height of the area.. Defaults to 1..

    Returns:
        torch.Tensor: The area of the triangle
    """
    return (right - left) * height / 2.0


def isosceles_area(
    left: TENSOR_FLOAT, 
    mid: TENSOR_FLOAT, height: TENSOR_FLOAT=1.
) -> torch.Tensor:
    """Calculate the area of the isosceles shape

    Args:
        left (TENSOR_FLOAT): The leftmost point
        mid (TENSOR_FLOAT): The midpoint 
        height (TENSOR_FLOAT, optional): The height of the area. Defaults to 1..

    Returns:
        torch.Tensor: The area of the isosceles
    """

    return (2 * mid - 2 * left) * height / 2.0


def isosceles(
    x: torch.Tensor, left: TENSOR_FLOAT, mid: TENSOR_FLOAT, height: TENSOR_FLOAT=1.,
    g: bool=False, clip: float=None
) -> torch.Tensor:

    """The isosceles triangle function

    Args:
        x (torch.Tensor): The input
        left (TENSOR_FLOAT): The leftmost point
        mid (TENSOR_FLOAT): The midpoint
        increasing (bool, optional): Whether the triangle is increasing. Defaults to False.
        height (TENSOR_FLOAT, optional): The height of the triangle. Defaults to 1..
        g (bool, optional): WHether to use a straight-through estimator. Defaults to False.
        clip (float, optional): Whether to clip if g is True. Defaults to None.

    Returns:
        torch.Tensor: The output
    """
    dx = mid - left
    return triangle(x, left, mid, mid + dx, height, g, clip)


def _trap_post_hook(grad, state):
    
    state['grad'] = grad
    return grad


def _trap_pre_hook(grad, x, left, mid1, mid2, right, clip, state):
    
    out_grad = state['grad']

    if right is not None:
        oob = (x < left) | (x > right) | ( (x > mid1) & (x < mid2))
    else:
        oob = (x < left) | ( (x > mid1) & (x < mid2))

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
    """The trapezoid function

    Args:
        x (torch.Tensor): The x value
        left (TENSOR_FLOAT): The left value
        mid1 (TENSOR_FLOAT): The left midpoint
        mid2 (TENSOR_FLOAT): The right midpoint
        right (TENSOR_FLOAT): The right point
        height (TENSOR_FLOAT, optional): The height. Defaults to 1..
        g (bool, optional): WHether to use a straight-through estimator. Defaults to False.
        clip (float, optional): Whether to clip if g is True. Defaults to None.

    Returns:
        torch.Tensor: The trapezoid output
    """

    state = {}
    x_base = x
    x = x_base.clone()
    if clip != 0.0 and g is True:
        x.register_hook(
            partial(_trap_pre_hook, 
                    x=x_base, left=left,
                    mid1=mid1, mid2=mid2, 
                    right=right, state=state, 
                    clip=clip))
    left_val = height / (mid1 - left) * (x - left)
    right_val = -height / (right - mid2) * (x - mid2) + height
    
    right_side = x >= mid2
    mid_val = (x >= mid1) & (x <= mid2)
    left_val[right_side] = right_val[right_side]
    left_val[mid_val] = height
    y = left_val

    if clip != 0.0 and g is True:
        y.register_hook(partial(_trap_post_hook, state=state))
    return y


def isosceles_trapezoid(
    x: torch.Tensor, left: TENSOR_FLOAT, mid1: TENSOR_FLOAT, 
    mid2: TENSOR_FLOAT, height: TENSOR_FLOAT=1., 
    g: bool=False,
    clip: float=0.0
) -> torch.Tensor:
    """An isosceles trapezoid

    Args:
        x (torch.Tensor): The x 
        left (TENSOR_FLOAT): The leftmost point
        mid1 (TENSOR_FLOAT): The first mid point
        mid2 (TENSOR_FLOAT): The second mid point
        height (TENSOR_FLOAT, optional): The height. Defaults to 1..
        g (bool, optional): Whether to have a ste gradient. Defaults to False.
        clip (float, optional): . Defaults to 0.0.

    Returns:
        torch.Tensor: The output
    """
    dl = mid1 - left
    right = mid2 + dl
    return trapezoid(
        x, left, mid1, mid2, right, height, g, clip
    )


def right_trapezoid(
    x: torch.Tensor, left: TENSOR_FLOAT, mid: TENSOR_FLOAT, right: TENSOR_FLOAT, 
    increasing: bool=False, height: TENSOR_FLOAT=1., g: bool=False,
    clip: float=None
) -> torch.Tensor:
    """A right trapezoid

    Args:
        x (torch.Tensor): _description_
        left (TENSOR_FLOAT): The leftmost value
        mid (TENSOR_FLOAT): The midpoint
        right (TENSOR_FLOAT): The rightmost value
        increasing (bool, optional): whether it is increasing or decreasing. Defaults to False.
        height (TENSOR_FLOAT, optional): The height of the trapezoid. Defaults to 1..
        g (bool, optional): Whether to use straight-through grads. Defaults to False.
        clip (float, optional): Amount to clip if using grads. Defaults to None.

    Returns:
        torch.Tensor: The point on the trapezoid
    """

    state = {}
    x_base = x
    x = x_base.clone()
    if clip != 0.0 and g is True:
        x.register_hook(
            partial(_trap_pre_hook, 
                    x=x_base, left=left,
                    mid1=mid, mid2=right, 
                    right=None, state=state, 
                    clip=clip))
    if increasing:
        val = height / (mid - left) * (x - left)
        mid_val = (x >= mid) & (x <= right)
    else:
        val = -height / (right - mid) * (x - mid) + height
        mid_val = (x >= left) & (x <= mid)
    
    val[mid_val] = height

    if clip != 0.0 and g is True:
        val.register_hook(partial(_trap_post_hook, state=state))
    return val


def trapezoid_area(
    left: TENSOR_FLOAT, mid1: TENSOR_FLOAT, mid2: TENSOR_FLOAT,
    right: TENSOR_FLOAT, height: TENSOR_FLOAT=1.
) -> torch.Tensor:
    """Calculate the area of a trapezoid

    Args:
        left (TENSOR_FLOAT): The leftmost point on the trapezoid
        mid1 (TENSOR_FLOAT): The midpoint on the trapaezoid
        mid2 (TENSOR_FLOAT): The second midpoint on the trapezoid
        right (TENSOR_FLOAT): The right point on the trapezoid
        height (TENSOR_FLOAT, optional): The height of the trapezoid. Defaults to 1..

    Returns:
        torch.Tensor: The area of the trapezoid
    """

    return ((right - left) + (mid2 - mid1)) * height / 2.0


def isosceles_trapezoid_area(
    left: TENSOR_FLOAT, mid2: TENSOR_FLOAT,
    height: TENSOR_FLOAT=1.
) -> torch.Tensor:
    """Calculate the area of an isosceles trapezoid

    Args:
        left (TENSOR_FLOAT): The leftmost point
        mid2 (TENSOR_FLOAT): The second point 
        height (TENSOR_FLOAT, optional): _description_. Defaults to 1..

    Returns:
        torch.Tensor: _description_
    """

    return (2 * mid2 - left) * height / 2
