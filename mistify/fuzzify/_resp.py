import torch


def resp(x: torch.Tensor, dim: int=-1, eps: float=1e-7, soft: bool=False) -> torch.Tensor:
    """Calculate the responsibility. The base calculate assumes all values are greater than zero

    Args:
        x (torch.Tensor): The value to take the responsibility of
        dim (int, optional): The dimension to take the responsibility on. Defaults to -1.
        eps (float, optional): The offset to avoid divisions by zero. Defaults to 1e-7.
        soft (bool): False. Whether to use softmax
        
    Returns:
        torch.Tensor: The responsibility
    """
    if soft is False:
        x = x.relu()
        return x / (x.sum(dim=dim, keepdim=True) + eps)
    return torch.softmax(x, dim=dim)


def resp_average(x: torch.Tensor, resp: torch.Tensor) -> torch.Tensor:
    """Weight all values by the responsibility and calculate the mean

    Args:
        x (torch.Tensor): The value to take the average of
        resp (torch.Tensor): The responsibility

    Returns:
        torch.Tensor: The weghted average
    """
    return (x * resp).sum(dim=0) / resp.sum(dim=0)


def resp_median(x: torch.Tensor, resp: torch.Tensor) -> torch.Tensor:
    """Calculate the weighted median using the responsibilities

    Args:
        x (torch.Tensor): The value to take the average of
        resp (torch.Tensor): The responsibility

    Returns:
        torch.Tensor: The weighted median
    """
    sorted_x, sorted_ind = x.sort(dim=0)
    sorted_resp = resp.gather(0, sorted_ind)
    cum_resp = sorted_resp.cumsum(dim=0)

    # get the value closest to 0.5
    _, closest_idx = (cum_resp - 0.5).abs().min(dim=0, keepdim=True)
    return sorted_x.gather(0, closest_idx)
