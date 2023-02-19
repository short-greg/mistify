import torch


def get_comp_weight_size(in_features: int, out_features: int, in_variables: int=None):

    if in_variables is None or in_variables == 0:
        return torch.Size([in_features, out_features])
    return torch.Size([in_variables, in_features, out_features])


def reduce(value: torch.Tensor, reduction: str):

    if reduction == 'mean':
        return value.mean()
    elif reduction == 'sum':
        return value.sum()
    elif reduction == 'batchmean':
        return value.sum() / value.size(0)
    return value


def maxmin(x: torch.Tensor, w: torch.Tensor, dim=-2):
    return torch.max(torch.min(x.unsqueeze(-1), w[None]), dim=dim)[0]


def minmax(x: torch.Tensor, w: torch.Tensor, dim=-2):
    return torch.min(torch.max(x.unsqueeze(-1), w[None]), dim=dim)[0]


def maxprod(x: torch.Tensor, w: torch.Tensor, dim=-2):
    return torch.max(x.unsqueeze(-1) * w[None], dim=dim)[0]
