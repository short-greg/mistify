import torch

def var_normalize(m: torch.Tensor, keepdim: bool=True, eps: float=1e-7) -> torch.Tensor:
    """Normalize the input based on the vairance

    Args:
        m (torch.Tensor): The input
        keepdim (bool, optional): Whether to keep the dimension. Defaults to True.
        eps (float, optional): Epsilon to add to prevent divide by zero. Defaults to 1e-7.

    Returns:
        torch.Tensor: Normalized input
    """
    # calculate sum over the dimension
    m_sum = m.sum(dim=-2, keepdim=True)
    normalized = m_sum / (m_sum.sum(dim=-1, keepdim=True) + eps)
    if keepdim:
        return normalized
    return normalized.squeeze(-2)
