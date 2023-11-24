# 3rd party
import torch


def swap(m: torch.Tensor, is_batch: bool=True) -> torch.Tensor:
    """swap the term dimension and variable dimension

    Args:
        m (torch.Tensor): the fuzzy set
        is_batch (bool, optional): whether the fuzzy set is a batch set. Defaults to True.

    Returns:
        torch.Tensor: a fuzzy set with terms and variable dimensions swapped
    """
    if not is_batch:
        return m.transpose(0, 1)
    return m.transpose(1, 2)


def expand_term(m: torch.Tensor, n_terms: int=None, is_batch: bool=True) -> torch.Tensor:
    """expand out the term dimension

    Args:
        m (torch.Tensor): the fuzzy set
        n_terms (int, optional): the number of terms to expand out. Defaults to None.
        is_batch (bool, optional): whether the fuzzy set is a batch set. Defaults to False.

    Returns:
        torch.Tensor: the expanded term
    """
    if n_terms is None:
        return m.unsqueeze(m.dim())
    if is_batch:
        return m.reshape(m.shape[0], -1, n_terms)
    return m.reshape(-1, n_terms)


def collapse_term(m: torch.Tensor, is_batch: bool=True) -> torch.Tensor:
    """collapse the term dimension

    Args:
        m (torch.Tensor): the fuzzy set
        is_batch (bool, optional): whether the fuzzy set is a batch set. Defaults to False.

    Returns:
        torch.Tensor: the expanded term
    """
    if is_batch:
        return m.reshape(m.shape[0], -1)
    return m.reshape(-1)
