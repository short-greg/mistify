# 3rd party
import torch


def swap(m: torch.Tensor) -> torch.Tensor:
    """swap the term dimension and variable dimension

    Args:
        m (torch.Tensor): the fuzzy set

    Returns:
        torch.Tensor: a fuzzy set with terms and variable dimensions swapped
    """
    return m.transpose(-1, -2)


def expand_term(m: torch.Tensor, n_terms: int=None) -> torch.Tensor:
    """expand out the term dimension

    Args:
        m (torch.Tensor): the fuzzy set
        n_terms (int, optional): the number of terms to expand out. Defaults to None.

    Returns:
        torch.Tensor: the expanded term
    """
    if n_terms is None:
        return m.unsqueeze(m.dim())
    shape = list(m.shape) + [n_terms]
    shape[-2] = -1
    return m.reshape(shape)


def collapse_term(m: torch.Tensor) -> torch.Tensor:
    """collapse the term dimension

    Args:
        m (torch.Tensor): the fuzzy set

    Returns:
        torch.Tensor: the expanded term
    """
    shape = list(m.shape)
    shape.pop(-1)
    shape[-1] = -1
    return m.reshape(shape)
