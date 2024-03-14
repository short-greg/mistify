import typing
import torch
from abc import ABC, abstractmethod


def maxmin(x: torch.Tensor, w: torch.Tensor, dim=-2) -> torch.Tensor:
    """Take max min between two tensors to compute the relation

    Args:
        x (torch.Tensor): Input tensor
        w (torch.Tensor): Weight tensor to calculate relation of
        dim (int, optional): Dimension to aggregate. Defaults to -2.

    Returns:
        torch.Tensor: The relation between two tensors
    """
    return torch.max(torch.min(x.unsqueeze(-1), w[None]), dim=dim)[0]


def ada_maxmin(x: torch.Tensor, w: torch.Tensor, dim=-2) -> torch.Tensor:
    """Take max min between two tensors to compute the relation

    Args:
        x (torch.Tensor): Input tensor
        w (torch.Tensor): Weight tensor to calculate relation of
        dim (int, optional): Dimension to aggregate. Defaults to -2.

    Returns:
        torch.Tensor: The relation between two tensors
    """
    return adamax_on(adamin(x.unsqueeze(-1), w[None]), dim=dim)


def ada_minmax(x: torch.Tensor, w: torch.Tensor, dim=-2) -> torch.Tensor:
    """Take max min between two tensors to compute the relation

    Args:
        x (torch.Tensor): Input tensor
        w (torch.Tensor): Weight tensor to calculate relation of
        dim (int, optional): Dimension to aggregate. Defaults to -2.

    Returns:
        torch.Tensor: The relation between two tensors
    """
    return adamin_on(adamax(x.unsqueeze(-1), w[None]), dim=dim)


def minmax(x: torch.Tensor, w: torch.Tensor, dim=-2) -> torch.Tensor:
    """Take min max between two tensors to compute the relation

    Args:
        x (torch.Tensor): Input tensor
        w (torch.Tensor): Weight tensor to calculate relation of
        dim (int, optional): Dimension to aggregate. Defaults to -2.

    Returns:
        torch.Tensor: The relation between two tensors
    """
    return torch.min(torch.max(x.unsqueeze(-1), w[None]), dim=dim)[0]


def maxprod(x: torch.Tensor, w: torch.Tensor, dim=-2) -> torch.Tensor:
    """Take max prod between two tensors to compute the relation

    Args:
        x (torch.Tensor): Input tensor
        w (torch.Tensor): Weight tensor to calculate relation of
        dim (int, optional): Dimension to aggregate. Defaults to -2.

    Returns:
        torch.Tensor: The relation between two tensors
    """
    return torch.max(x.unsqueeze(-1) * w[None], dim=dim)[0]


def smooth_max(x: torch.Tensor, x2: torch.Tensor, a: float=None) -> torch.Tensor:
    """Smooth approximation to the max function of two tensors

    Args:
        x (torch.Tensor): Tensor to take max of
        x2 (torch.Tensor): Other tensor to take max of
        a (float): Value to 

    Returns:
        torch.Tensor: Tensor containing the maximum of x1 and x2
    """
    if a is None:
        return torch.max(x, x2)
    z1 = ((x + 1) ** a).detach()
    z2 = ((x2 + 1) ** a).detach()
    return (x * z1 + x2 * z2) / (z1 + z2)


def smooth_max_on(x: torch.Tensor, dim: int, a: float=None, keepdim: bool=False) -> torch.Tensor:
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
    z = ((x + 1) ** a).detach()
    return (x * z).sum(dim=dim, keepdim=keepdim) / z.sum(dim=dim, keepdim=keepdim)


def smooth_min(x: torch.Tensor, x2: torch.Tensor, a: float=None) -> torch.Tensor:
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
    return smooth_max(x, x2, -a)


def smooth_min_on(
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
    return smooth_max_on(x, dim, -a, keepdim=keepdim)


def prod_on(
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


def adamax(x: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
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


def adamin(x: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
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


def adamax_on(x: torch.Tensor, dim: int, keepdim: bool=False) -> torch.Tensor:
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


def adamin_on(x: torch.Tensor, dim: int, keepdim: bool=False) -> torch.Tensor:
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


def max_on(x: torch.Tensor, dim: int=-1, keepdim: bool=False) -> torch.Tensor:
    """Take the max on a given dimension

    Args:
        x (torch.Tensor): Tensor to take the max of
        dim (int, optional): The dimension to take the max on. Defaults to -1.
        keepdim (bool, optional): Whether to keep the dim. Defaults to False.

    Returns:
        torch.Tensor: The max
    """
    return torch.max(x, dim=dim, keepdim=keepdim)[0]


def min_on(x: torch.Tensor, dim: int=-1, keepdim: bool=False) -> torch.Tensor:
    """Take the min on a given dimension

    Args:
        x (torch.Tensor): Tensor to take the max of
        dim (int, optional): The dimension to take the min on. Defaults to -1.
        keepdim (bool, optional): Whether to keep the dim. Defaults to False.

    Returns:
        torch.Tensor: The min
    """
    return torch.min(x, dim=dim, keepdim=keepdim)[0]


def prod_on(x: torch.Tensor, dim: int=-1, keepdim: bool=False) -> torch.Tensor:
    """Take the product on a given dimension

    Args:
        x (torch.Tensor): Tensor to take the product of
        dim (int, optional): The dimension to take the product on. Defaults to -1.
        keepdim (bool, optional): Whether to keep the dim. Defaults to False.

    Returns:
        torch.Tensor: The product
    """
    return torch.prod(x, dim=dim, keepdim=keepdim)


def prod(m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
    """Take the prod of two tensors 

    Args:
        m1 (torch.Tensor): Tensor 1 to take the prod of
        m2 (torch.Tensor): Tensor 2 to take the prod of
    Returns:
        torch.Tensor: The prod
    """
    return m1 * m2


def prob_sum(m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
    """Take the prob sum on a given dimension

    Args:
        m1 (torch.Tensor): Tensor 1 to take the prob sum of
        m2 (torch.Tensor): Tensor 2 to take the prob sum of
    Returns:
        torch.Tensor: The prob sum
    """
    return m1 + m2 - m1 * m2


def bounded_max(m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
    """Sum up two tensors and output with a max of 1

    Args:
        m1 (torch.Tensor): Tensor 1 to take the bounded max of
        m2 (torch.Tensor): Tensor 2 to take the bounded max of

    Returns:
        torch.Tensor: The bounded max
    """
    return torch.min(m1 + m2, torch.tensor(1.0, dtype=m1.dtype, device=m1.device))


def bounded_min(m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
    """Sum up two tensors and subtract number of tensors with a min of 0

    Args:
        m1 (torch.Tensor): Tensor 1 to take the bounded min of
        m2 (torch.Tensor): Tensor 2 to take the bounded min of

    Returns:
        torch.Tensor: The bounded min
    """
    return torch.max(m1 + m2 - 1, torch.tensor(0.0, dtype=m1.dtype, device=m1.device))


def bounded_max_on(m: torch.Tensor, dim=-1, keepdim: bool=False) -> torch.Tensor:
    """Take the bounded max on a given dimension

    Args:
        x (torch.Tensor): Tensor to take the bounded max of
        dim (int, optional): The dimension to take the bounded max on. Defaults to -1.
        keepdim (bool, optional): Whether to keep the dim. Defaults to False.

    Returns:
        torch.Tensor: The bounded max
    """
    return torch.min(
        m.sum(dim=dim, keepdim=keepdim),
        torch.tensor(1.0, device=m.device, dtype=m.dtype)
    )


def bounded_min_on(m: torch.Tensor, dim=-1, keepdim: bool=False) -> torch.Tensor:
    """Take the bounded min on a given dimension

    Args:
        x (torch.Tensor): Tensor to take the bounded min of
        dim (int, optional): The dimension to take the bounded min on. Defaults to -1.
        keepdim (bool, optional): Whether to keep the dim. Defaults to False.

    Returns:
        torch.Tensor: The bounded min
    """
    return torch.max(
        m.sum(dim=dim, keepdim=keepdim) 
        - m.size(dim) + 1, 
        torch.tensor(0.0, device=m.device, dtype=m.dtype)
    )


def to_signed(binary: torch.Tensor) -> torch.Tensor:
    """Convert a binary (zeros/ones) tensor to a signed one

    Args:
        binary (torch.Tensor): The binary tensor

    Returns:
        torch.Tensor: The signed tensor
    """
    return (binary * 2) - 1


def to_binary(signed: torch.Tensor) -> torch.Tensor:
    """Convert a signed (neg ones/ones) tensor to a binary one (zeros/ones)

    Args:
        signed (torch.Tensor): The signed tensor

    Returns:
        torch.Tensor: The binary tensor
    """
    return (signed + 1) / 2


ON_F = typing.Callable[[torch.Tensor, int, bool], torch.Tensor]
BETWEEN_F = typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class LogicalF(ABC):
    """A function for executing 
    """

    @abstractmethod
    def __call__(self, x: torch.Tensor, w: torch.Tensor, dim=-2) -> torch.Tensor:
        pass


class AndF(LogicalF):

    def __init__(self, union_between: BETWEEN_F, inter_on: ON_F):
        """Create a Functor for performing an And operation between
        a tensor and a weight

        Args:
            union_between (BETWEEN_F): The inner operation
            inter_on (ON_F): The aggregate operation
        """
        self.union_between = union_between
        self.inter_on = inter_on

    def __call__(self, x: torch.Tensor, w: torch.Tensor, dim=-2) -> torch.Tensor:

        return self.inter_on(
            self.union_between(x.unsqueeze(-1), w[None]), dim=dim
        )


class OrF(LogicalF):

    def __init__(self, inter_between: BETWEEN_F, union_on: ON_F):
        """Create a Functor for performing an Or operation between
        a tensor and a weight

        Args:
            inter_between (BETWEEN_F): The inner operation
            union_on (ON_F): The aggregate operation
        """
        self.union_on = union_on
        self.inter_between = inter_between

    def __call__(self, x: torch.Tensor, w: torch.Tensor, dim=-2) -> torch.Tensor:

        return self.union_on(
            self.inter_between(x.unsqueeze(-1), w[None]), dim=dim
        )
