import torch.nn as nn
import torch
# 1st party
from enum import Enum
from abc import abstractmethod


import torch
from functools import partial
# 3rd party
import torch
import torch.nn as nn
import typing

import torch
from functools import partial

# local

# Not sure why i have strides
# def get_strided_indices(n_points: int, stride: int, step: int=1):
#     initial_indices = torch.arange(0, n_points).as_strided((n_points - stride + 1, stride), (1, 1))
#     return initial_indices[torch.arange(0, len(initial_indices), step)]


# def stride_coordinates(coordinates: torch.Tensor, stride: int, step: int=1):

#     dim2_index = get_strided_indices(coordinates.size(2), stride, step)
#     return coordinates[:, :, dim2_index]

import typing


def weight_func(wf: typing.Union[str, typing.Callable]) -> typing.Callable:
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


def unsqueeze(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(dim=x.dim())



def join(m: torch.Tensor, nn_module: nn.Module, dim=-1, unsqueeze_dim: int=None):

    m_out = nn_module(m)
    if unsqueeze_dim is not None:
        m_out = m_out.unsqueeze(unsqueeze_dim)
    return torch.cat(
        [m, m_out], dim=dim
    )


def check_contains(x: torch.Tensor, pt1: torch.Tensor, pt2: torch.Tensor):
    
    return (x >= pt1) & (x <= pt2)


def resize_to(x1: torch.Tensor, x2: torch.Tensor, dim=0):

    if x1.size(dim) == 1 and x2.size(dim) != 1:
        size = [1] * x1.dim()
        size[dim] = x2.size(dim)
        return x1.repeat(*size)
    elif x1.size(dim) != x2.size(dim):
        raise ValueError()
    return x1


def unsqueeze(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(dim=x.dim())

