import torch


def rand(*size: int,  dtype=torch.float32, device='cpu') -> torch.Tensor:
    """Generate random fuzzy set

    Args:
        dtype (optional): The dtype of the data. Defaults to torch.float32.
        device (str, optional): The device of the data. Defaults to 'cpu'.

    Returns:
        torch.Tensor: The result
    """
    return (torch.rand(*size, device=device, dtype=dtype))


def positives(*size: int, dtype=torch.float32, device='cpu') -> torch.Tensor:
    """
    Generate a positive fuzzy set

    Args:
        dtype (optional): The dtype of the data. Defaults to torch.float32.
        device (str, optional): The device of the data. Defaults to 'cpu'.

    Returns:
        torch.Tensor: Positive fuzzy set
    """
    return torch.ones(*size, dtype=dtype, device=device)


def negatives(*size: int, dtype: torch.dtype=torch.float32, device='cpu') -> torch.Tensor:
    """
    Generate a negative fuzzy set

    Args:
        dtype (optional): The dtype of the data. Defaults to torch.float32.
        device (str, optional): The device of the data. Defaults to 'cpu'.

    Returns:
        torch.Tensor: Negative fuzzy set
    """
    return torch.zeros(*size, dtype=dtype, device=device)
