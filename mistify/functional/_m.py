# 1st party
import typing

# 3rd party
import torch

TENSOR_FLOAT = typing.Union[torch.Tensor, float]

from ._grad import (
    BinaryG, ClampG,
    SignG
)

def binary(x: torch.Tensor, g: bool=False, clip: float=None) -> torch.Tensor:
    """Convenience function to use the straight through estimator for binary

    Args:
        x1 (torch.Tensor): First tensor
        x2 (torch.Tensor): Second tensor

    Returns:
        torch.Tensor: The binarized tensor
    """
    if g is False:
        clip = None
    return BinaryG.apply(x, clip=clip)


def sign(x: torch.Tensor, g: bool=False, clip: float=None) -> torch.Tensor:
    """Convenience function to use the straight through estimator for sign

    Args:
        x1 (torch.Tensor): First tensor

    Returns:
        torch.Tensor: The signed tensor
    """
    if g is False:
        clip = None
    return SignG.apply(x, clip=clip)


def ramp(x: torch.Tensor, g: bool=False, clip: float=None) -> torch.Tensor:
    """Convenience function to use the straight through estimator for ramp

    Args:
        x1 (torch.Tensor): First tensor

    Returns:
        torch.Tensor: The signed tensor
    """
    if g is False:
        clip = None
    return ClampG.apply(x, clip=clip)


def to_binary(signed: torch.Tensor) -> torch.Tensor:
    """Convert a signed (neg ones/ones) tensor to a binary one (zeros/ones)

    Args:
        signed (torch.Tensor): The signed tensor

    Returns:
        torch.Tensor: The binary tensor
    """
    return (signed + 1) / 2


def to_signed(binary: torch.Tensor) -> torch.Tensor:
    """Convert a binary (zeros/ones) tensor to a signed one

    Args:
        binary (torch.Tensor): The binary tensor

    Returns:
        torch.Tensor: The signed tensor
    """
    return (binary * 2) - 1


# TODO: Test these functions



# def triangle(x: torch.Tensor, left: TENSOR_FLOAT, mid: TENSOR_FLOAT, right: TENSOR_FLOAT, height: TENSOR_FLOAT=1.) -> torch.Tensor:

#     left_val = height / (mid - left) * (x - left)
#     right_val = -height / (right - mid) * (x - mid) + height
    
#     right_side = x >= mid
#     left_val[right_side] = right_val[right_side]
#     return left_val


# ramp = torch.clamp


# def isosceles(x: torch.Tensor, left: TENSOR_FLOAT, mid: TENSOR_FLOAT, height: TENSOR_FLOAT=1.) -> torch.Tensor:

#     dx = mid - left
#     return triangle(x, left, mid, mid + dx, height)


# def trapezoid(x: torch.Tensor, left: TENSOR_FLOAT, mid1: TENSOR_FLOAT, mid2: TENSOR_FLOAT, right: TENSOR_FLOAT, height: TENSOR_FLOAT=1.) -> torch.Tensor:

#     left_val = height / (mid1 - left) * (x - left)
#     right_val = -height / (right - mid2) * (x - mid2) + height
    
#     right_side = x >= mid2
#     mid_val = (x >= mid1) & (x <= mid2)
#     left_val[right_side] = right_val[right_side]
#     left_val[mid_val] = height
#     return left_val


# def isosceles_trapezoid(x: torch.Tensor, left: TENSOR_FLOAT, mid1: TENSOR_FLOAT, mid2: TENSOR_FLOAT, height: TENSOR_FLOAT=1.) -> torch.Tensor:

#     dx = mid2 - left
    
#     return trapezoid(x, left, mid1, mid2, mid2 + dx, height)


# def max_on(x: torch.Tensor, dim: int=-1, keepdim: bool=False) -> torch.Tensor:
#     """Take the max on a given dimension

#     Args:
#         x (torch.Tensor): Tensor to take the max of
#         dim (int, optional): The dimension to take the max on. Defaults to -1.
#         keepdim (bool, optional): Whether to keep the dim. Defaults to False.

#     Returns:
#         torch.Tensor: The max
#     """
#     return torch.max(x, dim=dim, keepdim=keepdim)[0]


# def min_on(x: torch.Tensor, dim: int=-1, keepdim: bool=False) -> torch.Tensor:
#     """Take the min on a given dimension

#     Args:
#         x (torch.Tensor): Tensor to take the max of
#         dim (int, optional): The dimension to take the min on. Defaults to -1.
#         keepdim (bool, optional): Whether to keep the dim. Defaults to False.

#     Returns:
#         torch.Tensor: The min
#     """
#     return torch.min(x, dim=dim, keepdim=keepdim)[0]



# def bounded_union(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
#     """Sum up two tensors and output with a max of 1

#     Args:
#         x1 (torch.Tensor): Tensor 1 to take the bounded max of
#         x2 (torch.Tensor): Tensor 2 to take the bounded max of

#     Returns:
#         torch.Tensor: The bounded max
#     """
#     return torch.min(x1 + x2, torch.tensor(1.0, dtype=x1.dtype, device=x1.device))


# def bounded_min(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
#     """Sum up two tensors and subtract number of tensors with a min of 0

#     Args:
#         m1 (torch.Tensor): Tensor 1 to take the bounded min of
#         m2 (torch.Tensor): Tensor 2 to take the bounded min of

#     Returns:
#         torch.Tensor: The bounded min
#     """
#     return torch.max(x1 + x2 - 1, torch.tensor(0.0, dtype=x1.dtype, device=x1.device))


# def bounded_max_on(m: torch.Tensor, dim=-1, keepdim: bool=False) -> torch.Tensor:
#     """Take the bounded max on a given dimension

#     Args:
#         x (torch.Tensor): Tensor to take the bounded max of
#         dim (int, optional): The dimension to take the bounded max on. Defaults to -1.
#         keepdim (bool, optional): Whether to keep the dim. Defaults to False.

#     Returns:
#         torch.Tensor: The bounded max
#     """
#     return torch.min(
#         m.sum(dim=dim, keepdim=keepdim),
#         torch.tensor(1.0, device=m.device, dtype=m.dtype)
#     )


# def bounded_min_on(x: torch.Tensor, dim=-1, keepdim: bool=False) -> torch.Tensor:
#     """Take the bounded min on a given dimension

#     Args:
#         x (torch.Tensor): Tensor to take the bounded min of
#         dim (int, optional): The dimension to take the bounded min on. Defaults to -1.
#         keepdim (bool, optional): Whether to keep the dim. Defaults to False.

#     Returns:
#         torch.Tensor: The bounded min
#     """
#     return torch.max(
#         x.sum(dim=dim, keepdim=keepdim) 
#         - x.size(dim) + 1, 
#         torch.tensor(0.0, device=x.device, dtype=x.dtype)
#     )

