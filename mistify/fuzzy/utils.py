import torch
from ..functional import check_contains
import typing
import torch


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


def differ(m: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
    """
    Take the difference between two fuzzy sets
    
    Args:
        m (torch.Tensor): Fuzzy set to subtract from 
        m2 (torch.Tensor): Fuzzy set to subtract

    Returns:
        torch.Tensor: 
    """
    return (m - m2).clamp(0.0, 1.0)


def rand(*size: int,  dtype=torch.float32, device='cpu'):

    return (torch.rand(*size, device=device, dtype=dtype))


def positives(*size: int, dtype=torch.float32, device='cpu') -> torch.Tensor:
    """
    Generate a positive fuzzy set

    Args:
        dtype (_type_, optional): . Defaults to torch.float32.
        device (str, optional): . Defaults to 'cpu'.

    Returns:
        torch.Tensor: Positive fuzzy set
    """
    return torch.ones(*size, dtype=dtype, device=device)


def negatives(*size: int, dtype: torch.dtype=torch.float32, device='cpu') -> torch.Tensor:
    """
    Generate a negative fuzzy set

    Args:
        dtype (torch.dtype, optional): The data type for the fuzzys set. Defaults to torch.float32.
        device (str, optional): The device for the fuzzy set. Defaults to 'cpu'.

    Returns:
        torch.Tensor: Negative fuzzy set
    """
    return torch.zeros(*size, dtype=dtype, device=device)

