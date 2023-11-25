
# 3rd party
import torch


def negatives(*size: int, dtype=torch.float32, device='cpu') -> torch.Tensor:
    """
    Args:
        dtype (, optional): The type to set the ternary set to. Defaults to torch.float32.
        device (str, optional): _description_. Defaults to 'cpu'.

    Returns:
        torch.Tensor: Ternary set with all negative values
    """
    return torch.full(size, -1, dtype=dtype, device=device)


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


def rand(*size: int, positive_p: float=0.5, unknown_p: float=0.0, dtype=torch.float32, device='cpu') -> torch.Tensor:
    """
    Args:
        dtype (_type_, optional): . Defaults to torch.float32.
        device (str, optional): . Defaults to 'cpu'.

    Returns:
        torch.Tensor: Random value
    """

    y = torch.rand(*size, device=device, dtype=dtype)
    if positive_p != 0.5:
        offset = positive_p - 0.5
        y = torch.sign(y - offset)
    else:
        y = torch.sign(y)
    if unknown_p != 0.0:
        return y * (torch.rand(*size, device, dtype=dtype) <= unknown_p).type_as(y)
    return y

