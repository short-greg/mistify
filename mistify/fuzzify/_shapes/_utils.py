# 1st party
import torch
from ...utils import check_contains


# TODO: Delete?

def calc_m_linear_increasing(x: torch.Tensor, pt1: torch.Tensor, pt2: torch.Tensor, m: torch.Tensor):
    m_result = (x - pt1) * (m / (pt2 - pt1)) * check_contains(x, pt1, pt2).float() 
    return m_result


def calc_m_linear_decreasing(x: torch.Tensor, pt1: torch.Tensor, pt2: torch.Tensor, m: torch.Tensor):

    m_result = ((x - pt1) * (-m / (pt2 - pt1)) + m) * check_contains(x, pt1, pt2).float()
    return m_result


def calc_x_linear_increasing(m0: torch.Tensor, pt1: torch.Tensor, pt2: torch.Tensor, m: torch.Tensor):
    # NOTE: To save on computational costs do not perform checks to see
    # if m0 is greater than m

    m0 = torch.min(m0, m)
    x = m0 * (pt2 - pt1) / m + pt1
    torch.nan_to_num_(x, 0.0, 0.0)
    return x


def calc_x_linear_decreasing(m0: torch.Tensor, pt1, pt2, m: torch.Tensor):

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


def calc_m_flat(x, pt1: torch.Tensor, pt2: torch.Tensor, m: torch.Tensor):

    return m * check_contains(x, pt1, pt2).float()
