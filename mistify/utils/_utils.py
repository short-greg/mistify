# 1st party
from enum import Enum
from functools import partial
import typing

# 3rd party

import torch.nn as nn
import torch


# TODO: Make this an enum
# should be in inference (?)
def weight_func(wf: typing.Union[str, typing.Callable]) -> typing.Callable:
    """_summary_

    Args:
        wf (typing.Union[str, typing.Callable]): 

    Raises:
        ValueError: 

    Returns:
        typing.Callable: 
    """
    if isinstance(wf, str):
        if wf == 'sigmoid':
            return torch.sigmoid
        if wf == 'clamp':
            return partial(torch.clamp, min=0, max=1)
        if wf == 'sign':
            return torch.sign
        if wf == 'binary':
            return lambda x: torch.clamp(torch.round(x), 0, 1)

        raise ValueError(f'Invalid weight function {wf}')

    return wf


def join(m: torch.Tensor, nn_module: nn.Module, dim=-1, unsqueeze_dim: int=None) -> torch.Tensor:
    """_summary_

    Args:
        m (torch.Tensor): a membership tensor
        nn_module (nn.Module): _description_
        dim (int, optional): _description_. Defaults to -1.
        unsqueeze_dim (int, optional): _description_. Defaults to None.

    Returns:
        torch.Tensor: _description_
    """

    m_out = nn_module(m)
    if unsqueeze_dim is not None:
        m_out = m_out.unsqueeze(unsqueeze_dim)
    return torch.cat(
        [m, m_out], dim=dim
    )


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


def resize_to(x1: torch.Tensor, x2: torch.Tensor, dim=0) -> torch.Tensor:
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


def unsqueeze(x: torch.Tensor) -> torch.Tensor:
    """unsqueeze the final dimension in the tnesor

    Args:
        x (torch.Tensor): the tensor to unsqueeze

    Returns:
        torch.Tensor: the unsqueezed tensor
    """
    return x.unsqueeze(dim=x.dim())


class EnumFactory(Enum):

    @classmethod
    def factory(cls, f: typing.Union[str, typing.Callable]) -> typing.Callable[[typing.Any], torch.Tensor]:

        if isinstance(f, typing.Callable):
            return f
        return cls[f].value
