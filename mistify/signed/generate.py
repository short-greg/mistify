import torch


def negatives(*size: int, dtype=torch.float32, device='cpu') -> torch.Tensor:
    """
    Args:
        dtype (, optional): The type to set the ternary set to. Defaults to torch.float32.
        device (str, optional): _description_. Defaults to 'cpu'.

    Returns:
        torch.Tensor: Ternary set with all negative values
    """
    return torch.full(*size, -1, dtype=dtype, device=device)


def positives(*size: int, dtype=torch.float32, device='cpu'):
    """
    Args:
        dtype (, optional): The type to set the ternary set to. Defaults to torch.float32.
        device (str, optional): _description_. Defaults to 'cpu'.

    Returns:
        torch.Tensor: Ternary set with all positive values
    """

    return torch.ones(*size, dtype=dtype, device=device)


def unknowns(*size: int, dtype=torch.float32, device='cpu') -> torch.Tensor:
    """
    Args:
        dtype (, optional): The type to set the ternary set to. Defaults to torch.float32.
        device (str, optional): _description_. Defaults to 'cpu'.

    Returns:
        torch.Tensor: Ternary set with all unknown values
    """

    return torch.zeros(*size, dtype=dtype, device=device)


def rand(*size: int, dtype=torch.float32, device='cpu') -> torch.Tensor:
    """
    Args:
        dtype (_type_, optional): . Defaults to torch.float32.
        device (str, optional): . Defaults to 'cpu'.

    Returns:
        torch.Tensor: Random value
    """

    return torch.randn(*size, device=device, dtype=dtype).sign()


def forget(m: torch.Tensor, p: float) -> torch.Tensor:
    """Randomly forget values (this will make them unknown)

    Args:
        m (torch.Tensor): the membership matrix
        p (float): the probability of forgetting

    Returns:
        torch.Tensor: the tensor with randomly forgotten values
    """
    return m * (torch.rand_like(m) < p).type_as(m)
