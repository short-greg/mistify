"""
Core modules for Mistify

MistifyLoss: 

Functions: Standard functions used by mistify
CompositionBase: 
"""
import torch
import torch.nn as nn
from abc import abstractmethod
from enum import Enum


class ToOptim(Enum):
    """
    Specify whehther to optimize x, theta or both
    """

    X = 'x'
    THETA = 'theta'
    BOTH = 'both'

    def x(self) -> bool:
        return self in (ToOptim.X, ToOptim.BOTH)

    def theta(self) -> bool:
        return self in (ToOptim.THETA, ToOptim.BOTH)


class MistifyLoss(nn.Module):
    """Loss to use in modules for Mistify
    """

    def __init__(self, module: nn.Module, reduction: str='mean'):
        super().__init__()
        self.reduction = reduction
        self._module = module
        if reduction not in ('mean', 'sum', 'batchmean', 'none'):
            raise ValueError(f"Reduction {reduction} is not a valid reduction")

    @property
    def module(self) -> nn.Module:
        return self._module 

    def reduce(self, y: torch.Tensor):

        if self.reduction == 'mean':
            return y.mean()
        elif self.reduction == 'sum':
            return y.sum()
        elif self.reduction == 'batchmean':
            return y.sum() / len(y)
        elif self.reduction == 'none':
            return y
        
    @abstractmethod
    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def get_comp_weight_size(in_features: int, out_features: int, in_variables: int=None):

    if in_variables is None or in_variables == 0:
        return torch.Size([in_features, out_features])
    return torch.Size([in_variables, in_features, out_features])


def smooth_max(x: torch.Tensor, x2: torch.Tensor, a: float) -> torch.Tensor:
    """Smooth approximation to the max function of two tensors

    Args:
        x (torch.Tensor): Tensor to take max of
        x2 (torch.Tensor): Other tensor to take max of
        a (float): Value to 

    Returns:
        torch.Tensor: Tensor containing the maximum of x1 and x2
    """
    z1 = ((x + 1) ** a).detach()
    z2 = ((x2 + 1) ** a).detach()
    return (x * z1 + x2 * z2) / (z1 + z2)

def smooth_max_on(x: torch.Tensor, dim: int, a: float, keepdim: bool=False) -> torch.Tensor:
    """Take smooth max over specified dimension

    Args:
        x (torch.Tensor): 
        dim (int): Dimension to take max over
        a (float): Smoothing value. The larger the value the smoother

    Returns:
        torch.Tensor: Result of the smooth max
    """
    z = ((x + 1) ** a).detach()
    return (x * z).sum(dim=dim, keepdim=keepdim) / z.sum(dim=dim, keepdim=keepdim)


def smooth_min(x: torch.Tensor, x2: torch.Tensor, a: float) -> torch.Tensor:
    """Take smooth m over specified dimension

    Args:
        x (torch.Tensor): 
        dim (int): Dimension to take max over
        a (float): Smoothing value. The larger the value the smoother

    Returns:
        torch.Tensor: Result of the smooth max
    """
    return smooth_max(x, x2, -a)


def smooth_min_on(x: torch.Tensor, dim: int, a: float, keepdim: bool=False) -> torch.Tensor:
    """Take smooth min over specified dimension

    Args:
        x (torch.Tensor): 
        dim (int): Dimension to take max over
        a (float): Smoothing value. The larger the value the smoother
        keepdim (bool): Whether to keep the dimension or not

    Returns:
        torch.Tensor: Result of the smooth max
    """
    return smooth_max_on(x, dim, -a, keepdim=keepdim)


def adamax(x: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Smooth approximation to the max function of two tensors

    Args:
        x (torch.Tensor): Tensor to take max of
        x2 (torch.Tensor): Other tensor to take max of
    
    Returns:
        torch.Tensor: Tensor containing the maximum of x1 and x2
    """
    q = torch.clamp(-69 / torch.log(torch.max(x, x2)), max=1000, min=-1000).detach()  
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
    q = torch.clamp(-69 / torch.log(torch.max(x, dim=dim)[0]).detach(), max=1000, min=-1000)
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


class ComplementBase(nn.Module):
    """Base complement class for calculating complement of a set
    """

    def __init__(self, concatenate_dim: int=None):
        """initializer

        Args:
            concatenate_dim (int, optional): 
              Dim to concatenate the complement with. If None, it does not concatenate.
              Defaults to None.
        """
        super().__init__()
        self.concatenate_dim = concatenate_dim

    def postprocess(self, m: torch.Tensor, m_complement: torch.Tensor) -> torch.Tensor:
        """Postprocess the complement

        Args:
            m (torch.Tensor): The input tensor
            m_complement (torch.Tensor): The complemented tensor

        Returns:
            torch.Tensor: The postprocessed tensor
        """
        if self.concatenate_dim is None:
            return m_complement
        
        return torch.cat(
            [m, m_complement], dim=self.concatenate_dim
        )
    
    @abstractmethod
    def complement(self, m: torch.Tensor) -> torch.Tensor:
        """Take complemento f tensor

        Args:
            m (torch.Tensor): Tensor to take complement of

        Returns:
            torch.Tensor: Complemented tensor
        """
        raise NotImplementedError

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """Take complement of tesor

        Args:
            m (torch.Tensor): 

        Returns:
            torch.Tensor: 
        """
        return self.postprocess(m, self.complement(m))


class CompositionBase(nn.Module):

    def __init__(
        self, in_features: int, out_features: int, in_variables: int=None
    ):
        """Base class for taking relations between two tensor

        Args:
            in_features (int): Number of input features (i.e. terms)
            out_features (int): Number of outputs features (i.e. terms)
            in_variables (int, optional): Number of linguistic variables in. Defaults to None.
        """
        super().__init__()
        self._in_features = in_features
        self._out_features = out_features
        self._multiple_variables = in_variables is not None
        self.weight = torch.nn.parameter.Parameter(
            self.init_weight(in_features, out_features, in_variables)
        )
    
    @abstractmethod
    def init_weight(self, in_features: int, out_features: int, in_variables: int=None) -> torch.Tensor:
        pass
