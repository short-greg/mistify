import torch


def logistic_bell(x, b, s, m: torch.Tensor):

    z = s * (x - b)
    multiplier = 4 * m
    y = torch.sigmoid(z)
    return multiplier * y * (1 - y)


def logistic_cdf_inv(y, b, s):

    return -torch.log(1 / y - 1) / s + b


def logistic_diff(m0: torch.Tensor, s: torch.Tensor, m_base: torch.Tensor):
    
    m = m0 / m_base
    dx = -torch.log((-m - 2 * torch.sqrt(1 - m) + 2) / (m)).float()
    dx = dx / s
    return dx


def logistic_area(s: torch.Tensor, m_base: torch.Tensor, left=True):
    
    return 4 * m_base / s