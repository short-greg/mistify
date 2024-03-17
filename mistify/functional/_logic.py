# 3rd party
import torch


from ._join import (
    union_on, inter, ada_inter, ada_inter_on,
    ada_union, ada_union_on, union, inter_on
)


def or_(x: torch.Tensor, w: torch.Tensor, dim=-2, g: bool=False) -> torch.Tensor:
    """Take max min between two tensors to compute the relation

    Args:
        x (torch.Tensor): Input tensor
        w (torch.Tensor): Weight tensor to calculate relation of
        dim (int, optional): Dimension to aggregate. Defaults to -2.

    Returns:
        torch.Tensor: The relation between two tensors
    """
    return union_on(inter(x.unsqueeze(-1), w[None], g), dim=dim, g=g)


def ada_or(x: torch.Tensor, w: torch.Tensor, dim=-2) -> torch.Tensor:
    """Take max min between two tensors to compute the relation

    Args:
        x (torch.Tensor): Input tensor
        w (torch.Tensor): Weight tensor to calculate relation of
        dim (int, optional): Dimension to aggregate. Defaults to -2.

    Returns:
        torch.Tensor: The relation between two tensors
    """
    return ada_union_on(ada_inter(x.unsqueeze(-1), w[None]), dim=dim)


def ada_and(x: torch.Tensor, w: torch.Tensor, dim=-2) -> torch.Tensor:
    """Take max min between two tensors to compute the relation

    Args:
        x (torch.Tensor): Input tensor
        w (torch.Tensor): Weight tensor to calculate relation of
        dim (int, optional): Dimension to aggregate. Defaults to -2.

    Returns:
        torch.Tensor: The relation between two tensors
    """
    return ada_inter_on(ada_union(x.unsqueeze(-1), w[None]), dim=dim)


def and_(x: torch.Tensor, w: torch.Tensor, dim=-2, g: bool=False) -> torch.Tensor:
    """Take min max between two tensors to compute the relation

    Args:
        x (torch.Tensor): Input tensor
        w (torch.Tensor): Weight tensor to calculate relation of
        dim (int, optional): Dimension to aggregate. Defaults to -2.

    Returns:
        torch.Tensor: The relation between two tensors
    """
    return inter_on(union(x.unsqueeze(-1), w[None], g), dim=dim, g=g)


def or_prod(x: torch.Tensor, w: torch.Tensor, dim=-2, g: bool=False) -> torch.Tensor:
    """Take max prod between two tensors to compute the relation

    Args:
        x (torch.Tensor): Input tensor
        w (torch.Tensor): Weight tensor to calculate relation of
        dim (int, optional): Dimension to aggregate. Defaults to -2.

    Returns:
        torch.Tensor: The relation between two tensors
    """
    return union_on(x.unsqueeze(-1) * w[None], dim=dim, g=g)
