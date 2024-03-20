import torch


def rand(*size: int, dtype=torch.float32, device='cpu'):
    """Generate random boolean set

    Args:
        dtype (optional): The dtype of the data . Defaults to torch.float32.
        device (str, optional): The device of the data. Defaults to 'cpu'.

    Returns:
        torch.Tensor: The result
    """
    return (torch.rand(*size, device=device, dtype=dtype)).round()


def negatives(*size: int, dtype=torch.float32, device='cpu'):
    """
    Generate a negative boolean set

    Args:
        dtype (optional): The dtype of the data. Defaults to torch.float32.
        device (str, optional): The device of the data. Defaults to 'cpu'.

    Returns:
        torch.Tensor: Negative boolean set
    """
    return torch.zeros(*size, dtype=dtype, device=device)


def positives(*size: int, dtype=torch.float32, device='cpu'):
    """
    Generate a positive boolean set

    Args:
        dtype (optional): The dtype of the data. Defaults to torch.float32.
        device (str, optional): The device of the data. Defaults to 'cpu'.

    Returns:
        torch.Tensor: Positive boolean set
    """
    return torch.ones(*size, dtype=dtype, device=device)


def unknowns(*size: int, dtype=torch.float32, device='cpu'):
    """
    Generate a set of unknowns. Unknowns are represented with the value 0.5

    Args:
        dtype (optional): The dtype of the data. Defaults to torch.float32.
        device (str, optional): The device of the data. Defaults to 'cpu'.

    Returns:
        torch.Tensor: Set of unknowns
    """
    return torch.full(size, 0.5, dtype=dtype, device=device)

def forget(m: torch.Tensor, p: float) -> torch.Tensor:
    """Randomly forget values (this will make them unknown)

    Args:
        m (torch.Tensor): the membership matrix
        p (float): the probability of forgetting

    Returns:
        torch.Tensor: the tensor with randomly forgotten values
    """
    return m * (torch.rand_like(m) < p).type_as(m) + 0.5
