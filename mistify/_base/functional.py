import torch.nn as nn
import torch
# 1st party
from enum import Enum
from abc import abstractmethod

# 3rd party
import torch
import torch.nn as nn

# local

# Not sure why i have strides
# def get_strided_indices(n_points: int, stride: int, step: int=1):
#     initial_indices = torch.arange(0, n_points).as_strided((n_points - stride + 1, stride), (1, 1))
#     return initial_indices[torch.arange(0, len(initial_indices), step)]


# def stride_coordinates(coordinates: torch.Tensor, stride: int, step: int=1):

#     dim2_index = get_strided_indices(coordinates.size(2), stride, step)
#     return coordinates[:, :, dim2_index]


def join(m: torch.Tensor, nn_module: nn.Module, dim=-1, unsqueeze_dim: int=None):

    m_out = nn_module(m)
    if unsqueeze_dim is not None:
        m_out = m_out.unsqueeze(unsqueeze_dim)
    return torch.cat(
        [m, m_out], dim=dim
    )


def check_contains(x: torch.Tensor, pt1: torch.Tensor, pt2: torch.Tensor):
    
    return (x >= pt1) & (x <= pt2)


def maxmin(x: torch.Tensor, w: torch.Tensor, dim=-2) -> torch.Tensor:
    """Take max min between two tensors to compute the relation

    Args:
        x (torch.Tensor): Input tensor
        w (torch.Tensor): Weight tensor to calculate relation of
        dim (int, optional): Dimension to aggregate. Defaults to -2.

    Returns:
        torch.Tensor: The relation between two tensors
    """
    return torch.max(torch.min(x.unsqueeze(-1), w[None]), dim=dim)[0]


def ada_maxmin(x: torch.Tensor, w: torch.Tensor, dim=-2) -> torch.Tensor:
    """Take max min between two tensors to compute the relation

    Args:
        x (torch.Tensor): Input tensor
        w (torch.Tensor): Weight tensor to calculate relation of
        dim (int, optional): Dimension to aggregate. Defaults to -2.

    Returns:
        torch.Tensor: The relation between two tensors
    """
    return adamax_on(adamin(x.unsqueeze(-1), w[None]), dim=dim)


def ada_minmax(x: torch.Tensor, w: torch.Tensor, dim=-2) -> torch.Tensor:
    """Take max min between two tensors to compute the relation

    Args:
        x (torch.Tensor): Input tensor
        w (torch.Tensor): Weight tensor to calculate relation of
        dim (int, optional): Dimension to aggregate. Defaults to -2.

    Returns:
        torch.Tensor: The relation between two tensors
    """
    return adamin_on(adamax(x.unsqueeze(-1), w[None]), dim=dim)


def minmax(x: torch.Tensor, w: torch.Tensor, dim=-2) -> torch.Tensor:
    """Take min max between two tensors to compute the relation

    Args:
        x (torch.Tensor): Input tensor
        w (torch.Tensor): Weight tensor to calculate relation of
        dim (int, optional): Dimension to aggregate. Defaults to -2.

    Returns:
        torch.Tensor: The relation between two tensors
    """
    return torch.min(torch.max(x.unsqueeze(-1), w[None]), dim=dim)[0]


def maxprod(x: torch.Tensor, w: torch.Tensor, dim=-2) -> torch.Tensor:
    """Take max prod between two tensors to compute the relation

    Args:
        x (torch.Tensor): Input tensor
        w (torch.Tensor): Weight tensor to calculate relation of
        dim (int, optional): Dimension to aggregate. Defaults to -2.

    Returns:
        torch.Tensor: The relation between two tensors
    """
    return torch.max(x.unsqueeze(-1) * w[None], dim=dim)[0]


def smooth_max(x: torch.Tensor, x2: torch.Tensor, a: float) -> torch.Tensor:
    """Smooth approximation to the max function of two tensors

    Args:
        x (torch.Tensor): Tensor to take max of
        x2 (torch.Tensor): Other tensor to take max of
        a (float): Value to 

    Returns:
        torch.Tensor: Tensor containing the maximum of x1 and x2
    """
    z1 = ((x + 1) ** a).detach()
    z2 = ((x2 + 1) ** a).detach()
    return (x * z1 + x2 * z2) / (z1 + z2)


def smooth_max_on(x: torch.Tensor, dim: int, a: float, keepdim: bool=False) -> torch.Tensor:
    """Take smooth max over specified dimension

    Args:
        x (torch.Tensor): 
        dim (int): Dimension to take max over
        a (float): Smoothing value. The larger the value the smoother

    Returns:
        torch.Tensor: Result of the smooth max
    """
    z = ((x + 1) ** a).detach()
    return (x * z).sum(dim=dim, keepdim=keepdim) / z.sum(dim=dim, keepdim=keepdim)


def smooth_min(x: torch.Tensor, x2: torch.Tensor, a: float) -> torch.Tensor:
    """Take smooth m over specified dimension

    Args:
        x (torch.Tensor): 
        dim (int): Dimension to take max over
        a (float): Smoothing value. The larger the value the smoother

    Returns:
        torch.Tensor: Result of the smooth max
    """
    return smooth_max(x, x2, -a)


def smooth_min_on(
        x: torch.Tensor, dim: int, a: float, keepdim: bool=False
    ) -> torch.Tensor:
    """Take smooth min over specified dimension

    Args:
        x (torch.Tensor): 
        dim (int): Dimension to take max over
        a (float): Smoothing value. The larger the value the smoother
        keepdim (bool): Whether to keep the dimension or not

    Returns:
        torch.Tensor: Result of the smooth max
    """
    return smooth_max_on(x, dim, -a, keepdim=keepdim)


def adamax(x: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Smooth approximation to the max function of two tensors

    Args:
        x (torch.Tensor): Tensor to take max of
        x2 (torch.Tensor): Other tensor to take max of
    
    Returns:
        torch.Tensor: Tensor containing the maximum of x1 and x2
    """
    q = torch.clamp(-69 / torch.log(torch.min(x, x2)), max=1000, min=-1000).detach()  
    return ((x ** q + x2 ** q) / 2) ** (1 / q)


def adamin(x: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Smooth approximation to the min function of two tensors

    Args:
        x (torch.Tensor): Tensor to take max of
        x2 (torch.Tensor): Other tensor to take max of
    
    Returns:
        torch.Tensor: Tensor containing the maximum of x1 and x2
    """
    q = torch.clamp(69 / torch.log(torch.min(x, x2)).detach(), max=1000, min=-1000)
    result = ((x ** q + x2 ** q) / 2) ** (1 / q)
    return result


def adamax_on(x: torch.Tensor, dim: int, keepdim: bool=False) -> torch.Tensor:
    """Take smooth max over specified dimension

    Args:
        x (torch.Tensor): 
        dim (int): Dimension to take max over
        a (float): Smoothing value. The larger the value the smoother

    Returns:
        torch.Tensor: Result of the smooth max
    """
    q = torch.clamp(-69 / torch.log(torch.min(x, dim=dim)[0]).detach(), max=1000, min=-1000)
    return (torch.sum(x ** q.unsqueeze(dim), dim=dim, keepdim=keepdim) / x.size(dim)) ** (1 / q)


def adamin_on(x: torch.Tensor, dim: int, keepdim: bool=False) -> torch.Tensor:
    """Take smooth min over specified dimension

    Args:
        x (torch.Tensor): 
        dim (int): Dimension to take max over
        keepdim (bool): Whether to keep the dimension or not

    Returns:
        torch.Tensor: Result of the smooth max
    """
    q = torch.clamp(69 / torch.log(torch.min(x, dim=dim)[0]).detach(), max=1000, min=-1000)
    return (torch.sum(x ** q.unsqueeze(dim), dim=dim, keepdim=keepdim) / x.size(dim)) ** (1 / q)


def max_on(x: torch.Tensor, dim: int=-1, keepdim: bool=False) -> torch.Tensor:

    return torch.max(x, dim=dim, keepdim=keepdim)[0]


def min_on(x: torch.Tensor, dim: int=-1, keepdim: bool=False) -> torch.Tensor:

    return torch.min(x, dim=dim, keepdim=keepdim)[0]


def resize_to(x1: torch.Tensor, x2: torch.Tensor, dim=0):

    if x1.size(dim) == 1 and x2.size(dim) != 1:
        size = [1] * x1.dim()
        size[dim] = x2.size(dim)
        return x1.repeat(*size)
    elif x1.size(dim) != x2.size(dim):
        raise ValueError()
    return x1


def to_signed(binary: torch.Tensor) -> torch.Tensor:
    return (binary * 2) - 1


def to_binary(signed: torch.Tensor) -> torch.Tensor:
    return (signed + 1) / 2
