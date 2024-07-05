# 1st party
from enum import Enum
from functools import partial
import typing

# 3rd party
import torch


def check_contains(x: torch.Tensor, pt1: torch.Tensor, pt2: torch.Tensor) -> torch.BoolTensor:
    """Check if a tensor falls between two points

    Args:
        x (torch.Tensor): the tensor to check
        pt1 (torch.Tensor): the first point
        pt2 (torch.Tensor): the second boint

    Returns:
        torch.BoolTensor: whether the tensor is contained
    """
    return (x >= pt1) & (x <= pt2)


def resize_dim_to(x1: torch.Tensor, x2: torch.Tensor, dim=0) -> torch.Tensor:
    """resize a tensor to the same as 

    Args:
        x1 (torch.Tensor): the tensor to resize
        x2 (torch.Tensor): the tensor with the target size
        dim (int, optional): the dimension to resize. Defaults to 0.

    Raises:
        ValueError: If the sizes do not align

    Returns:
        torch.Tensor: the resized tensor
    """

    if x1.size(dim) == 1 and x2.size(dim) != 1:
        size = [1] * x1.dim()
        size[dim] = x2.size(dim)
        return x1.repeat(*size)
    elif x1.size(dim) != x2.size(dim):
        raise ValueError()
    return x1


def resize_to(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """resize a tensor to the same as 

    Args:
        x1 (torch.Tensor): the tensor to resize
        x2 (torch.Tensor): the tensor with the target size
        dim (int, optional): the dimension to resize. Defaults to 0.

    Raises:
        ValueError: If the sizes do not align

    Returns:
        torch.Tensor: the resized tensor
    """

    x1_shape = x1.shape
    x2_shape = x2.shape
    dim = max(len(x1_shape), len(x2_shape))

    repeat_to = []
    for i in range(dim):
        if i >= x1.dim():
            x1 = x1.unsqueeze(-1)
        if x1.size(i) == 1 and x2.size(i) != 1:
            repeat_to.append(x2.size(i))
        elif x1.size(i) != x2.size(i):
            raise ValueError('Cannot reshape')
        else:
            repeat_to.append(1)
    return x1.repeat(repeat_to)


def unsqueeze(x: torch.Tensor) -> torch.Tensor:
    """unsqueeze the final dimension in the tnesor

    Args:
        x (torch.Tensor): the tensor to unsqueeze

    Returns:
        torch.Tensor: the unsqueezed tensor
    """
    return x.unsqueeze(dim=x.dim())


class EnumFactory(dict):
    """Base class for creating an Enum
    """

    def f(self, f: typing.Union[str, typing.Callable], *args, **kwargs) -> typing.Callable[[typing.Any], torch.Tensor]:
        """Create the function based on f

        Args:
            f (typing.Union[str, typing.Callable]): The function to use. 
             If it is a Callable, that function will be used rather than doing a lookup

        Returns:
            typing.Callable[[typing.Any], torch.Tensor]: Create the class specified by Enum. 
        """
        if isinstance(f, typing.Callable):
            return partial(f, *args, **kwargs)
        return partial(self[f], *args, **kwargs)


def reduce_as(x: torch.Tensor, target: typing.Union[torch.Tensor, torch.Size]) -> torch.Tensor:
    """Reduce x like a target using the sum reduction

    Args:
        x (torch.Tensor): The tensor to reduce
        target (typing.Union[torch.Tensor, torch.Size]): The reference to reduce x by

    Returns:
        torch.Tensor: The output
    """
    if isinstance(target, torch.Tensor):
        target = target.shape

    for i, (sz1, sz2) in enumerate(zip(x.size(), target)):
        if sz1 != 1 and sz2 == 1:
            x = x.sum(dim=i, keepdim=True)
    return x
