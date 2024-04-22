from ._m import TENSOR_FLOAT
from functools import partial
import torch
from ._grad import G, ClipG


def _shape_post_hook(grad, state):
    
    state['grad'] = grad
    return grad


def _shape_pre_hook(grad, x: torch.Tensor, oob, g: G, state):
    
    out_grad = state['grad']
    grad = grad.clone()
    if oob is True:
        grad[:] = out_grad[:]
    else:
        grad[oob] = out_grad[oob]

    if g is not None:
        grad = g(x, grad, oob)
    return grad


def triangle(
    x: torch.Tensor, left: TENSOR_FLOAT, mid: TENSOR_FLOAT, 
    right: TENSOR_FLOAT, height: TENSOR_FLOAT=1., g: G=None,
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
    oob = (x < left) | (x > right)
    if g is not None:
        x.register_hook(partial(_shape_pre_hook, x=x, oob=oob, state=state, g=g))

    left_val = height / (mid - left) * (x - left)
    right_val = -height / (right - mid) * (x - mid) + height
    
    right_side = x >= mid
    left_val[right_side] = right_val[right_side]

    left_val[oob] = 0.0
    if g is not None:
        left_val.register_hook(partial(_shape_post_hook, state=state))
    return left_val


def right_triangle(
    x: torch.Tensor, left: TENSOR_FLOAT, mid: TENSOR_FLOAT, 
    increasing: bool=False, height: TENSOR_FLOAT=1., g: G=None
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
    oob = (x < left) | (x > mid)
    if g is not None:
        x.register_hook(partial(_shape_pre_hook, x=x, oob=oob, state=state, g=g))

    if increasing:
        val = height / (mid - left) * (x - left)
    else:
        val = -(height / (mid - left)) * (x - left) + height
    
    val[oob] = 0.0

    if g is not None:
        val.register_hook(partial(_shape_post_hook, state=state))
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
        torch.Tensor: The area
    """
    return (right - left) * height / 2.0


def isosceles_area(
    left: TENSOR_FLOAT, 
    mid: TENSOR_FLOAT, height: TENSOR_FLOAT=1.
) -> torch.Tensor:
    """Calculate the area of the isosceles triangle

    Args:
        left (TENSOR_FLOAT): The leftmost point
        mid (TENSOR_FLOAT): The midpoint 
        height (TENSOR_FLOAT, optional): The height of the area. Defaults to 1..

    Returns:
        torch.Tensor: The area
    """
    return (2 * mid - 2 * left) * height / 2.0


def isosceles(
    x: torch.Tensor, left: TENSOR_FLOAT, mid: TENSOR_FLOAT, height: TENSOR_FLOAT=1.,
    g: G=None
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
    return triangle(x, left, mid, mid + dx, height, g)


def triangle_centroid(
    left: TENSOR_FLOAT, mid: TENSOR_FLOAT, 
    right: TENSOR_FLOAT
) -> torch.Tensor:
    """Calculate the centroid of an triangle

    Args:
        left (TENSOR_FLOAT): The leftmost point
        mid (TENSOR_FLOAT): The mid point 
        right (TENSOR_FLOAT): The second point 
    Returns:
        torch.Tensor: The centroid
    """
    return (left + mid + right) / 3.0


def right_triangle_centroid(
    left: TENSOR_FLOAT, 
    right: TENSOR_FLOAT,
    increasing: bool=True
) -> torch.Tensor:
    """Calculate the centroid of a right trapezoid

    Args:
        left (TENSOR_FLOAT): The leftmost point
        right (TENSOR_FLOAT): The right-hand point 
        increasing (bool): Whether the right triangle is increasing or decreasing
    Returns:
        torch.Tensor: The centroid
    """
    if increasing:
        return (2 / 3) * right + (1 / 3) * left
    return (1 / 3) * right + (2 / 3) * left


def isosceles_centroid(
    mid: TENSOR_FLOAT
) -> torch.Tensor:
    """Calculate the centroid of an isosceles trapezoid

    Args:
        mid1 (TENSOR_FLOAT): The leftmost point
        mid2 (TENSOR_FLOAT): The second point 
    Returns:
        torch.Tensor: The centroid
    """
    return mid


def trapezoid(
    x: torch.Tensor, left: TENSOR_FLOAT, mid1: TENSOR_FLOAT, 
    mid2: TENSOR_FLOAT, right: TENSOR_FLOAT, height: TENSOR_FLOAT=1., 
    g: G=None
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

    oob = (x < left) | (x > right)
    if g is not None:
        x.register_hook(
            partial(_shape_pre_hook, x=x,
                    oob=oob, state=state, 
                    g=g))
    left_val = height / (mid1 - left) * (x - left)
    right_val = -height / (right - mid2) * (x - mid2) + height
    
    right_side = x >= mid2
    mid_val = (x >= mid1) & (x <= mid2)
    left_val[right_side] = right_val[right_side]
    left_val[mid_val] = height
    y = left_val
    y[oob] = 0.0

    if g is not None:
        y.register_hook(partial(_shape_post_hook, state=state))
    return y


def isosceles_trapezoid(
    x: torch.Tensor, left: TENSOR_FLOAT, mid1: TENSOR_FLOAT, 
    mid2: TENSOR_FLOAT, height: TENSOR_FLOAT=1., 
    g: G=None
) -> torch.Tensor:
    """An isosceles trapezoid

    Args:
        x (torch.Tensor): The x 
        left (TENSOR_FLOAT): The leftmost point
        mid1 (TENSOR_FLOAT): The first mid point
        mid2 (TENSOR_FLOAT): The second mid point
        height (TENSOR_FLOAT, optional): The height. Defaults to 1..
        g (G, optional): Whether to have a ste gradient. Defaults to False.

    Returns:
        torch.Tensor: The output
    """
    dl = mid1 - left
    right = mid2 + dl
    return trapezoid(
        x, left, mid1, mid2, right, height, g
    )


def right_trapezoid(
    x: torch.Tensor, left: TENSOR_FLOAT, mid: TENSOR_FLOAT, right: TENSOR_FLOAT, 
    increasing: bool=False, height: TENSOR_FLOAT=1., g: G=None
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

    Returns:
        torch.Tensor: The point on the trapezoid
    """

    state = {}
    x_base = x
    x = x_base.clone()
    oob = (x < left) | (x > right)
    if g is not None:
        x.register_hook(
            partial(_shape_pre_hook, x=x,
                    oob=oob, state=state, 
                    g=g))
    if increasing:
        val = height / (mid - left) * (x - left)
        mid_val = (x >= mid) & (x <= right)
    else:
        val = -height / (right - mid) * (x - mid) + height
        mid_val = (x >= left) & (x <= mid)
    
    val[mid_val] = height
    val[oob] = 0.0

    if g is not None:
        val.register_hook(partial(_shape_post_hook, state=state))
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


def trapezoid_centroid(
    left: TENSOR_FLOAT, mid1: TENSOR_FLOAT, mid2: TENSOR_FLOAT,
    right: TENSOR_FLOAT, height: TENSOR_FLOAT=1.0
) -> torch.Tensor:
    """_summary_

    Args:
        left (TENSOR_FLOAT): The leftmost point
        mid1 (TENSOR_FLOAT): The first point at the top point
        mid2 (TENSOR_FLOAT): The second point at the top point 
        right (TENSOR_FLOAT): The rightmost point
        height (TENSOR_FLOAT, optional): the height of the trapezoid. Defaults to 1.0.

    Returns:
        torch.Tensor: The trapezoid's centroid
    """

    b = mid2 - mid1
    a = right - left

    return height * (b + 2 * a) / (3 * (a + b))


def right_trapezoid_area(
    left: TENSOR_FLOAT, mid: TENSOR_FLOAT, 
    right: TENSOR_FLOAT, height: TENSOR_FLOAT=1.,
    increasing: bool=True
) -> torch.Tensor:
    """Calculate the area of a right trapezoid

    Args:
        left (TENSOR_FLOAT): The leftmost point on the trapezoid
        mid (TENSOR_FLOAT): The midpoint on the trapaezoid
        right (TENSOR_FLOAT): The right point on the trapezoid
        height (TENSOR_FLOAT, optional): The height of the trapezoid. Defaults to 1..

    Returns:
        torch.Tensor: The area
    """
    if increasing:
        return trapezoid_area(
            left, mid, right, right, height
        )
    return trapezoid_area(
        left, left, mid, right
    )


def right_trapezoid_centroid(
    left: TENSOR_FLOAT, mid: TENSOR_FLOAT,
    right: TENSOR_FLOAT, height: TENSOR_FLOAT=1.0,
    increasing: bool=True
) -> torch.Tensor:
    """Calculate the centroid of a right trapezoid

    Args:
        left (TENSOR_FLOAT): The leftmost point
        mid (TENSOR_FLOAT): The midpoint
        right (TENSOR_FLOAT): the right-hand point
    Returns:
        torch.Tensor: The centroid
    """

    if increasing:
        b = right - mid
    else:
        b = mid - left

    a = right - left

    return height * (b + 2 * a) / (3 * (a + b))


def isosceles_trapezoid_area(
    left: TENSOR_FLOAT, mid2: TENSOR_FLOAT,
    height: TENSOR_FLOAT=1.
) -> torch.Tensor:
    """Calculate the area of an isosceles trapezoid

    Args:
        left (TENSOR_FLOAT): The leftmost point
        mid2 (TENSOR_FLOAT): The second point at the top
        height (TENSOR_FLOAT, optional): The height of the trapezoid. Defaults to 1..

    Returns:
        torch.Tensor: The area
    """

    return (2 * mid2 - left) * height / 2


def isosceles_trapezoid_centroid(
    mid1: TENSOR_FLOAT, mid2: TENSOR_FLOAT
) -> torch.Tensor:
    """Calculate the centroid of an isosceles trapezoid

    Args:
        mid1 (TENSOR_FLOAT): The leftmost point
        mid2 (TENSOR_FLOAT): The second point 
    Returns:
        torch.Tensor: Calculate the area of an isosceles trapezoid
    """
    return (mid2 - mid1) / 2.0


def square(
    x: torch.Tensor, left: TENSOR_FLOAT, 
    right: TENSOR_FLOAT, height: TENSOR_FLOAT=1., g: G=None,
) -> torch.Tensor:
    """The square function

    Args:
        x (torch.Tensor): The input
        left (TENSOR_FLOAT): The leftmost point
        right (TENSOR_FLOAT): The rightmost point
        height (TENSOR_FLOAT, optional): The height of the square. Defaults to 1..
        g (bool, optional): WHether to use a straight-through estimator. Defaults to False.
        clip (float, optional): Whether to clip if g is True. Defaults to None.

    Returns:
        torch.Tensor: The output
    """
    state = {}
    x_base = x
    x = x_base.clone()
    oob = True
    if g is not None:
        x.register_hook(partial(_shape_pre_hook, x=x, oob=oob, state=state, g=g))

    result = ((x >= left) & (x <= right)) * height

    if g is not None:
        result.register_hook(partial(_shape_post_hook, state=state))
    return result


def square_area(
    left: TENSOR_FLOAT, right: TENSOR_FLOAT,
    height: TENSOR_FLOAT=1.
) -> torch.Tensor:
    """Calculate the area of a square

    Args:
        left (TENSOR_FLOAT): The lefthand point
        right (TENSOR_FLOAT): The righthand point 
        height (TENSOR_FLOAT, optional): The heght of the square. Defaults to 1..

    Returns:
        torch.Tensor: Calculate the area of an isosceles trapezoid
    """
    return (right - left) * height


def square_centroid(
    left: TENSOR_FLOAT, right: TENSOR_FLOAT
) -> torch.Tensor:
    """Calculate the area of a square

    Args:
        left (TENSOR_FLOAT): The lefthand point
        right (TENSOR_FLOAT): The righthand point 

    Returns:
        torch.Tensor: Calculate the area of an isosceles trapezoid
    """
    return (right - left) / 2.0
