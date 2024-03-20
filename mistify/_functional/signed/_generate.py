
# 3rd party
import torch


def negatives(*size: int, dtype=torch.float32, device='cpu') -> torch.Tensor:
    """
    Generate a negative signed set

    Args:
        dtype (optional): The dtype of the data. Defaults to torch.float32.
        device (str, optional): The device of the data. Defaults to 'cpu'.

    Returns:
        torch.Tensor: Negative signed set
    """
    return torch.full(size, -1, dtype=dtype, device=device)


def positives(*size: int, dtype=torch.float32, device='cpu'):
    """
    Generate a positive signed set

    Args:
        dtype (optional): The dtype of the data. Defaults to torch.float32.
        device (str, optional): The device of the data. Defaults to 'cpu'.

    Returns:
        torch.Tensor: Positive signed set
    """

    return torch.ones(*size, dtype=dtype, device=device)


def unknowns(*size: int, dtype=torch.float32, device='cpu') -> torch.Tensor:
    """
    Generate a signed set of unknowns. Unknowns are represented by 0

    Args:
        dtype (optional): The dtype of the data. Defaults to torch.float32.
        device (str, optional): The device of the data. Defaults to 'cpu'.

    Returns:
        torch.Tensor: Unknown signed set
    """

    return torch.zeros(*size, dtype=dtype, device=device)


def rand(*size: int, positive_p: float=0.5, dtype=torch.float32, device='cpu') -> torch.Tensor:
    """Generate random signed set

    Args:
        positive_p (float, optional): The probability of a positive. Defaults to 0.5.
        dtype (optional): The dtype of the data. Defaults to torch.float32.
        device (str, optional): The device of the data. Defaults to 'cpu'.

    Returns:
        torch.Tensor: The random signed set
    """

    y = torch.rand(*size, device=device, dtype=dtype)
    if positive_p != 0.5:
        offset = positive_p - 0.5
        y = torch.sign(y - offset)
    else:
        y = torch.sign(y)
    # if unknown_p != 0.0:
    #     return y * (torch.rand(*size, device, dtype=dtype) <= unknown_p).type_as(y)
    return y


def forget(m: torch.Tensor, p: float) -> torch.Tensor:
    """Randomly forget values (this will make them unknown)

    Args:
        m (torch.Tensor): the membership matrix
        p (float): the probability of forgetting

    Returns:
        torch.Tensor: the tensor with randomly forgotten values
    """
    return m * (torch.rand_like(m) < p).type_as(m)
