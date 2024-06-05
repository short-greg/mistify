import torch

def var_normalize(m: torch.Tensor, keepdim: bool=True, eps: float=1e-7) -> torch.Tensor:

    # calculate sum over the dimension
    m_sum = m.sum(dim=-2, keepdim=True)
    normalized = m_sum / (m_sum.sum(dim=-1, keepdim=True) + eps)
    if keepdim:
        return normalized
    return normalized.squeeze(-2)
