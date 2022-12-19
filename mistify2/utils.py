import torch


def get_comp_weight_size(in_features: int, out_features: int, in_variables: int=None):
    if in_variables is None:
        return torch.Size([in_features, out_features])
    return torch.Size([in_variables, in_features, out_features])


def reduce(value: torch.Tensor, reduction: str):

    if reduction == 'mean':
        return value.mean()
    elif reduction == 'sum':
        return value.sum()
    return value
    