# 3rd party
import torch

import torch

from ._grad import (
    MaxOnG, ClampG,
    MinOnG, G
)

def equalize_size(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:

    r1 = [1] * x1.dim()
    r2 = [1] * x2.dim()

    for i, (s1, s2) in enumerate(zip(x1.shape, x2.shape)):
        if s1 == 1 and s2 != 1:
            r1[i] = s2
        elif s1 != 1 and s2 == 1:
            r2[i] = s1
        elif s2 == s1:
            pass
        else:
            raise ValueError(
                f'x1 and x2 have incompatibles sizes {x1.shape} {x2.shape}' 
                f' dim: {i} s1: {s1} s2: {s2}'
            )
        
    return x1.repeat(r1), x2.repeat(r2)


def union(x1: torch.Tensor, x2: torch.Tensor, g: G=None) -> torch.Tensor:
    """Convenience function to use the straight through estimator for max

    Args:
        x1 (torch.Tensor): First tensor
        x2 (torch.Tensor): Second tensor
        g (G): Function to use on grads for straight through estimation

    Returns:
        torch.Tensor: The max tensor
    """
    if g is None:
        return torch.max(x1, x2)
    
    # TODO: reconsider this
    x1, x2 = equalize_size(x1, x2)
    x_combined = torch.cat(
        [x1[...,None], x2[...,None]], dim=-1
    )
    return MaxOnG.apply(x_combined, -1, False, g)[0]


def inter(x1: torch.Tensor, x2: torch.Tensor, g: G=None) -> torch.Tensor:
    """Convenience function to use the straight through estimator for min

    Args:
        x1 (torch.Tensor): First tensor
        x2 (torch.Tensor): Second tensor
        g (G): Function to use on grads for straight through estimation

    Returns:
        torch.Tensor: The min tensor
    """
    if g is None:
        return torch.min(x1, x2)
    x1, x2 = equalize_size(x1, x2)
    x_combined = torch.cat(
        [x1[...,None], x2[...,None]], dim=-1
    )
    return MaxOnG.apply(x_combined, -1, False, g)[0]
    # return MinG.apply(x1, x2, g)


def smooth_union(
    x: torch.Tensor, x2: torch.Tensor, 
    a: float=None, eps: float=1e-7
) -> torch.Tensor:
    """Smooth approximation to the max function of two tensors

    Args:
        x (torch.Tensor): Tensor to take max of
        x2 (torch.Tensor): Other tensor to take max of
        a (float): Smoothing value

    Returns:
        torch.Tensor: Tensor containing the maximum of x1 and x2
    """
    if a is None:
        return torch.max(x, x2)
    
    # combined = torch.cat([x[None], x2[None]], dim=0)
    z1 = torch.exp(x * a)
    z2 = torch.exp(x2 * a)
    return (x * z1 + x2 * z2) / (z1 + z2 + eps)
    # z2 = z1 / denominator

    # return (x * z1)
    
    # return (z * combined).sum(dim=0)

    # z1 = ((x + 1) ** a).detach()
    # z2 = ((x2 + 1) ** a).detach()
    # return (x * z1 + x2 * z2) / (z1 + z2)


def smooth_union_on(x: torch.Tensor, dim: int, a: float=None, keepdim: bool=False) -> torch.Tensor:
    """Take smooth max over specified dimension

    Args:
        x (torch.Tensor): 
        dim (int): Dimension to take max over
        a (float): Smoothing value. The larger the value the smoother

    Returns:
        torch.Tensor: Result of the smooth max
    """
    if a is None:
        return torch.max(x, dim=dim, keepdim=keepdim)[0]
    # z = ((x + 1) ** a).detach()
    # return (x * z).sum(dim=dim, keepdim=keepdim) / z.sum(dim=dim, keepdim=keepdim)
    z = torch.softmax(x * a, dim=dim)
    return (x * z).sum(dim=dim, keepdim=keepdim)


def smooth_inter(x: torch.Tensor, x2: torch.Tensor, a: float=None) -> torch.Tensor:
    """Take smooth m over specified dimension

    Args:
        x (torch.Tensor): 
        dim (int): Dimension to take max over
        a (float): Smoothing value. The larger the value the smoother

    Returns:
        torch.Tensor: Result of the smooth max
    """
    if a is None:
        return torch.min(x, x2)
    return smooth_union(x, x2, -a)


def smooth_inter_on(
        x: torch.Tensor, dim: int, a: float=None, keepdim: bool=False
    ) -> torch.Tensor:
    """Take smooth min over specified dimension

    Args:
        x (torch.Tensor): 
        dim (int): Dimension to take max over
        a (float): Smoothing value. The larger the value the smoother
        keepdim (bool): Whether to keep the dimension or not

    Returns:
        torch.Tensor: Result of the smooth max
    """
    if a is None:
        return torch.min(x, dim=dim, keepdim=keepdim)[0]
    return smooth_union_on(x, dim, -a, keepdim=keepdim)


def prob_inter_on(
        x: torch.Tensor, dim: int, keepdim: bool=False
    ) -> torch.Tensor:
    """Take product over the specified dimension

    Args:
        x (torch.Tensor): 
        dim (int): Dimension to take max over
        keepdim (bool): Whether to keep the dimension or not

    Returns:
        torch.Tensor: Result of the smooth max
    """
    return torch.prod(x, dim, keepdim)


def ada_union(x: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Smooth approximation to the max function of two tensors

    Args:
        x (torch.Tensor): Tensor to take max of
        x2 (torch.Tensor): Other tensor to take max of
    
    Returns:
        torch.Tensor: Tensor containing the maximum of x1 and x2
    """
    # TODO: Reevaluate the reduction from 690 to 69
    q = torch.clamp(-69 / torch.log(torch.min(x, x2)), max=1000, min=-1000).detach()  
    return ((x ** q + x2 ** q) / 2) ** (1 / q)


def ada_inter(x: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Smooth approximation to the min function of two tensors

    Args:
        x (torch.Tensor): Tensor to take max of
        x2 (torch.Tensor): Other tensor to take max of
    
    Returns:
        torch.Tensor: Tensor containing the maximum of x1 and x2
    """
    q = torch.clamp(69 / torch.log(torch.min(x, x2)).detach(), max=1000, min=-1000)
    result = ((x ** q + x2 ** q) / 2) ** (1 / q)
    return result


def ada_union_on(x: torch.Tensor, dim: int, keepdim: bool=False) -> torch.Tensor:
    """Take smooth max over specified dimension

    Args:
        x (torch.Tensor): 
        dim (int): Dimension to take max over
        a (float): Smoothing value. The larger the value the smoother

    Returns:
        torch.Tensor: Result of the smooth max
    """
    q = torch.clamp(-69 / torch.log(torch.min(x, dim=dim)[0]).detach(), max=1000, min=-1000)
    return (torch.sum(x ** q.unsqueeze(dim), dim=dim, keepdim=keepdim) / x.size(dim)) ** (1 / q)


def ada_inter_on(x: torch.Tensor, dim: int, keepdim: bool=False) -> torch.Tensor:
    """Take smooth min over specified dimension

    Args:
        x (torch.Tensor): 
        dim (int): Dimension to take max over
        keepdim (bool): Whether to keep the dimension or not

    Returns:
        torch.Tensor: Result of the smooth max
    """
    q = torch.clamp(69 / torch.log(torch.min(x, dim=dim)[0]).detach(), max=1000, min=-1000)
    return (torch.sum(x ** q.unsqueeze(dim), dim=dim, keepdim=keepdim) / x.size(dim)) ** (1 / q)


def inter_on(x: torch.Tensor, dim: int=-1, keepdim: bool=False, g: G=None, idx: bool=False) -> torch.Tensor:
    """Convenience function to use the straight through estimator for min

    Args:
        x (torch.Tensor): The input
        dim (int, optional): The dimension. Defaults to -1.
        keepdim (bool, optional): Whether to keep the dimension. Defaults to False.
        g (G): Function to use on grads for straight through estimation
        idx: Whether to keep the index that is output for calculating Min

    Returns:
        torch.Tensor: The min
    """
    if g is None:
        y = torch.min(x, dim, keepdim)
    else:
        y = MinOnG.apply(x, dim, keepdim, g)
    if idx:
        return y
    return y[0]


def union_on(x: torch.Tensor, dim: int=-1, keepdim: bool=False, g: G=None, idx: bool=False) -> torch.Tensor:
    """Convenience function to use the straight through estimator for max

    Args:
        x (torch.Tensor): The input
        dim (int, optional): The dimension. Defaults to -1.
        keepdim (bool, optional): Whether to keep the dimension. Defaults to False.
        g (G): Function to use on grads for straight through estimation
        idx: Whether to keep the index that is output for calculating Max

    Returns:
        torch.Tensor: The max
    """
    if g is None:
        y = torch.max(x, dim, keepdim)
    else: y = MaxOnG.apply(x, dim, keepdim, g)
    if idx:
        return y
    return y[0]


def prob_inter_on(x: torch.Tensor, dim: int=-1, keepdim: bool=False) -> torch.Tensor:
    """Take the product on a given dimension

    Args:
        x (torch.Tensor): Tensor to take the product of
        dim (int, optional): The dimension to take the product on. Defaults to -1.
        keepdim (bool, optional): Whether to keep the dim. Defaults to False.

    Returns:
        torch.Tensor: The product
    """
    return torch.prod(x, dim=dim, keepdim=keepdim)


def prob_inter(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Take the prod of two tensors 

    Args:
        m1 (torch.Tensor): Tensor 1 to take the prod of
        m2 (torch.Tensor): Tensor 2 to take the prod of
    Returns:
        torch.Tensor: The prod
    """
    return x1 * x2


def prob_union(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Take the prob sum on a given dimension

    Args:
        m1 (torch.Tensor): Tensor 1 to take the prob sum of
        m2 (torch.Tensor): Tensor 2 to take the prob sum of
    Returns:
        torch.Tensor: The prob sum
    """
    return x1 + x2 - x1 * x2


def prob_union_on(x: torch.Tensor, dim: int=-1, keepdim: bool=False) -> torch.Tensor:
    """_summary_

    Args:
        x (torch.Tensor): The tensor to take the probabilistic max on
        dim (int, optional): The dimension to take it on. Defaults to -1.
        keepdim (bool, optional): Whether to keep the dimension. Defaults to False.

    Returns:
        torch.Tensor: The probabilistic union
    """
    return 1 - torch.prod(1 - x, dim=dim, keepdim=keepdim)


def bounded_inter(x1: torch.Tensor, x2: torch.Tensor, g: G=None) -> torch.Tensor:
    """Convenience function to use the straight through estimator for bounded min

    Args:
        x1 (torch.Tensor): First tensor
        x2 (torch.Tensor): Second tensor
        g (G): Function to use on grads for straight through estimation

    Returns:
        torch.Tensor: The max tensor
    """
    y = x1 + x2 - 1
    if g is None:
        return torch.relu(y)
    return ClampG.apply(
        y, 0.0, 1.0, g
    )


def bounded_union(x1: torch.Tensor, x2: torch.Tensor, g: G=None) -> torch.Tensor:
    """Convenience function to use the straight through estimator for bounded max

    Args:
        x1 (torch.Tensor): First tensor
        x2 (torch.Tensor): Second tensor
        g (G): Function to use on grads for straight through estimation

    Returns:
        torch.Tensor: The max tensor
    """
    y = x1 + x2
    # max_val = torch.tensor(1.0, dtype=x1.dtype, device=x1.device)
    if g is None:
        return torch.clamp(y, max=1.0)
    return ClampG.apply(
        y, 0.0, 1.0, g
    )


def bounded_union_on(m: torch.Tensor, dim=-1, keepdim: bool=False, g: G=None) -> torch.Tensor:
    """Take the bounded max on a given dimension

    Args:
        x (torch.Tensor): Tensor to take the bounded max of
        dim (int, optional): The dimension to take the bounded max on. Defaults to -1.
        keepdim (bool, optional): Whether to keep the dim. Defaults to False.
        g (G): Function to use on grads for straight through estimation

    Returns:
        torch.Tensor: The bounded max
    """
    y = m.sum(dim=dim, keepdim=keepdim)
    if g is None:
        return torch.clamp(y, max=1.0)
    return ClampG.apply(
        y, 0.0, 1.0, g
    )


def bounded_inter_on(x: torch.Tensor, dim=-1, keepdim: bool=False, g: G=None) -> torch.Tensor:
    """Take the bounded min on a given dimension

    Args:
        x (torch.Tensor): Tensor to take the bounded min of
        dim (int, optional): The dimension to take the bounded min on. Defaults to -1.
        keepdim (bool, optional): Whether to keep the dim. Defaults to False.
        g (G): Function to use on grads for straight through estimation

    Returns:
        torch.Tensor: The bounded min
    """
    y = x.sum(dim=dim, keepdim=keepdim) - x.size(dim) + 1
    if g is False:
        return torch.relu(y)
    return ClampG.apply(
        y, 0.0, 1.0, g
    )
