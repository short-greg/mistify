# 1st party
from enum import Enum
from abc import abstractmethod

# 3rd party
import torch
import torch.nn as nn

# local

class ToOptim(Enum):
    """
    Specify whehther to optimize x, theta or both
    """

    X = 'x'
    THETA = 'theta'
    BOTH = 'both'

    def x(self) -> bool:
        return self in (ToOptim.X, ToOptim.BOTH)

    def theta(self) -> bool:
        return self in (ToOptim.THETA, ToOptim.BOTH)


def unsqueeze(x: torch.Tensor):
    return x.unsqueeze(x.dim())


def calc_m_linear_increasing(x: torch.Tensor, pt1: torch.Tensor, pt2: torch.Tensor, m: torch.Tensor):
    return (x - pt1) * (m / (pt2 - pt1)) * check_contains(x, pt1, pt2).float() 


def calc_m_linear_decreasing(x: torch.Tensor, pt1: torch.Tensor, pt2: torch.Tensor, m: torch.Tensor):
    return ((x - pt1) * (-m / (pt2 - pt1)) + m) * check_contains(x, pt1, pt2).float()


def calc_x_linear_increasing(m0: torch.Tensor, pt1: torch.Tensor, pt2: torch.Tensor, m: torch.Tensor):
    # NOTE: To save on computational costs do not perform checks to see
    # if m0 is greater than m

    # TODO: use intersect function
    m0 = torch.min(m0, m)
    # m0 = m0.intersect(m)
    x = m0 * (pt2 - pt1) / m + pt1
    torch.nan_to_num_(x, 0.0, 0.0)
    return x


def calc_x_linear_decreasing(m0: torch.Tensor, pt1, pt2, m: torch.Tensor):

    # m0 = m0.intersect(m)
    m0 = torch.min(m0, m)
    x = -(m0 - 1) * (pt2 - pt1) / m + pt1
    torch.nan_to_num_(x, 0.0, 0.0)
    return x


def calc_m_logistic(x, b, s, m: torch.Tensor):

    z = s * (x - b)
    multiplier = 4 * m
    y = torch.sigmoid(z)
    return multiplier * y * (1 - y)


def calc_x_logistic(y, b, s):

    return -torch.log(1 / y - 1) / s + b


def calc_dx_logistic(m0: torch.Tensor, s: torch.Tensor, m_base: torch.Tensor):
    
    m = m0 / m_base
    dx = -torch.log((-m - 2 * torch.sqrt(1 - m) + 2) / (m)).float()
    dx = dx / s
    return dx


def calc_area_logistic(s: torch.Tensor, m_base: torch.Tensor, left=True):
    
    return 4 * m_base / s


def calc_area_logistic_one_side(x: torch.Tensor, b: torch.Tensor, s: torch.Tensor, m_base: torch.Tensor):
    
    z = s * (x - b)
    left = (z < 0).float()
    a = torch.sigmoid(z)
    a = left * a + (1 - left) * (1 - a)

    return a * m_base * 4 / s


def resize_to(x1: torch.Tensor, x2: torch.Tensor, dim=0):

    if x1.size(dim) == 1 and x2.size(dim) != 1:
        size = [1] * x1.dim()
        size[dim] = x2.size(dim)
        return x1.repeat(*size)
    elif x1.size(dim) != x2.size(dim):
        raise ValueError()
    return x1


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
