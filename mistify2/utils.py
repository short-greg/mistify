import torch


def reduce(value: torch.Tensor, reduction: str):

    if reduction == 'mean':
        return value.mean()
    elif reduction == 'sum':
        return value.sum()
    return value
    