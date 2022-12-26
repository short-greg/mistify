import torch
# NEED TO CHANGE THE NAME


def maxmin(x: torch.Tensor, w: torch.Tensor, dim=-2):
    return torch.max(torch.min(x.unsqueeze(-1), w[None]), dim=dim)[0]


def minmax(x: torch.Tensor, w: torch.Tensor, dim=-2):
    return torch.min(torch.max(x.unsqueeze(-1), w[None]), dim=dim)[0]


def maxprod(x: torch.Tensor, w: torch.Tensor, dim=-2):
    return torch.min(x.unsqueeze(-1) * w[None], dim=dim)[0]
