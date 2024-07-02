# 1st party
from abc import abstractmethod, ABC

# 3rd party
import torch.nn as nn
import torch

# local
from .._functional import clamp, G


class MembershipAct(nn.Module, ABC):
    """Abstract base class for the membership
    """

    def __init__(self, n_terms: int=None):
        """Create the membership activation

        Args:
            n_terms (int, optional): The number of terms for the activation. Defaults to None.
        """
        super().__init__()
        self._n_terms = n_terms
    
    @abstractmethod
    def forward(self, m: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Descale(MembershipAct):
    """Activation scales the membership function to disregard values below a certain value and 
    remove all other values
    """

    def __init__(self, lower_bound: float):
        """Create a descaling activation

        Args:
            lower_bound (float): The threshold 

        """
        super().__init__(None)
        if not (0 < lower_bound < 1):
            raise ValueError(f'Argument lower_bound must be in range (0, 1) not {lower_bound}')
        self._lower_bound = lower_bound
        self._scale = 1 / (1 - self._lower_bound)
    
    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """Descale the membership

        Args:
            m (torch.Tensor): the membership

        Returns:
            torch.Tensor: The updated membership
        """
        return (torch.clamp(m, self._lower_bound) - self._lower_bound) / self._scale


# TODO: check if it works with current
class Sigmoidal(MembershipAct):
    """Inverse sigmoid followed by parameterized forward sigmoid"""

    def __init__(self, n_terms: int, positive_scale: bool=False, n_vars: int=False, device='cpu'):
        """Create an activation with a parameterized sigmoid

        Args:
            n_terms (int): The number of terms
            positive_scale (bool, optional): Whether it should be scaled positively or not. Defaults to False.
            n_vars (int, optional): The number of vars. Use None if not defined, Use False if no var dimension. Defaults to False.
            device (str, optional): Device for the parameter. Defaults to 'cpu'.
        """
        super().__init__(n_terms)
        self._positive_scale = positive_scale
        if n_vars is False:
            self.b = torch.nn.Parameter(torch.empty((n_terms,), device=device))
            self.s = torch.nn.Parameter(torch.empty((n_terms,), device=device))
        else:
            n_vars = 1 if n_vars is None else n_vars
            self.b = torch.nn.Parameter(torch.empty((n_vars, n_terms,), device=device))
            self.s = torch.nn.Parameter(torch.empty((n_vars, n_terms,), device=device))

        self._batch_dim = 2 if n_vars is False else 3
        torch.nn.init.normal_(self.b, -0.05, 0.05)
        torch.nn.init.uniform_(self.s, 0.5, 1.5)
    
    def forward(self, m: torch.Tensor):

        if m.dim() == self._batch_dim:
            s = self.s[None]
            b = self.b[None]
        else:
            s, b = self.s, self.b
        
        if self._positive_scale:
            s = torch.nn.functional.softplus(s)

        # inverts sigmoid and then calculates sigmoid again through parameterization
        result = 1 / (
            1 + ((1 / m - 1) ** s) * torch.exp(b)
        )
        # ensure the edges (0, 1) are not nans
        condition1 = ((m==0) & (s >0)) | ((m==1) & (s < 0))
        condition2 = ((m==1) & (s >0)) | ((m==0) & (s < 0))
        result = torch.where(condition1, result, torch.tensor(0.0, dtype=result.dtype, device=result.device))
        result = torch.where(condition2, result, torch.tensor(1.0, dtype=result.dtype, device=result.device))
        
        return result


class Triangular(MembershipAct):
    """Warps the membership by a triangular function then warps back"""

    def __init__(self, n_terms: int, n_vars: int=False, device='cpu'):
        super().__init__(n_terms)
        if n_vars is False:
            self.b_base = torch.nn.Parameter(torch.empty((n_terms,), device=device))
            self.offset_base = torch.nn.Parameter(torch.empty((n_terms,), device=device))
        else:

            n_vars = 1 if n_vars is None else n_vars
            self.b_base = torch.nn.Parameter(torch.empty((n_vars, n_terms,), device=device))
            self.offset_base = torch.nn.Parameter(torch.empty((n_vars, n_terms,), device=device))

        self._batch_dim = 2 if n_vars is False else 3
        torch.nn.init.normal_(self.b_base, -0.05, 0.05)
        torch.nn.init.normal_(self.offset_base, -0.05, 0.05)
    
    def forward(self, m):
        if m.dim() == self._batch_dim:
            b_base = self.b_base[None]
            offset_base = self.offset_base[None]
        else:
            b_base, offset_base = self.b_base, self.offset_base

        b = nn.functional.softplus(b_base)
        offset = torch.sigmoid(offset_base)
        x = m / 2 - offset
        return torch.min((x + b) / b, (b - x) / b)


class Hedge(nn.Module):
    """Update the linguistic term with an exponential
    """

    def __init__(self, n_terms: int, n_vars: int=False, lower_bound: float=0.0, upper_bound: float=None, g: G=None):

        super().__init__()
        if lower_bound < 0.0:
            raise ValueError(f'Lower bound must be greater than or equal to 0 not {lower_bound}')
        self._n_terms = n_terms
        if lower_bound is not None and lower_bound < 0:
            raise ValueError(f'Argument lower bound must be greater than 0 not {lower_bound}')
        if upper_bound is not None and upper_bound < lower_bound:
            raise ValueError(f'Argument upper bound must be greater than or equal to lower bound not {upper_bound}')
        
        if n_vars is False:
            self._w = nn.parameter.Parameter(torch.ones(n_terms))
        else:
            n_vars = 1 if n_vars is None else n_vars
            self._w = nn.parameter.Parameter(torch.ones(n_vars, n_terms))

        self._upper_bound = upper_bound
        self._lower_bound = lower_bound
        self._batch_dim = 2 if n_vars is False else 3
        self.g = g

    # TODO: Determine if this is necessary
    def align(self, w: torch.Tensor, m: torch.Tensor, dim: int=1) -> torch.Tensor:
        """Align the membership

        Args:
            w (torch.Tensor): The value to align to
            m (torch.Tensor): The membership
            dim (int, optional): The dimension. Defaults to 1.

        Returns:
            torch.Tensor: The 
        """
        shape = [1] * m.dim()
        shape[dim] = 0

        for i, s in enumerate(shape):
            if s == 1:
                w = w.unsqueeze(i)
        return dim

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """Use the hedge on the membership

        Args:
            m (torch.Tensor): The input

        Returns:
            torch.Tensor: The hedged membership
        """
        if m.dim() == self._batch_dim:
            w = self._w[None]
        else:
            w = self._w

        w = clamp(w, self._lower_bound, self._upper_bound, self.g)
        return m ** w
