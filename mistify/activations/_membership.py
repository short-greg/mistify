import torch.nn as nn
import torch
from abc import abstractmethod


class MembershipActivation(nn.Module):

    def __init__(self, n_features: int):
        super().__init__()
        self._n_features = n_features
    
    @abstractmethod
    def forward(self, m: torch.Tensor):
        raise NotImplementedError


class Descale(nn.Module):

    def __init__(self, lower_bound: torch.Tensor):
        super().__init__()
        if not (0 < lower_bound < 1):
            raise ValueError(f'Argument lower_bound must be in range (0, 1) not {lower_bound}')
        self._lower_bound = lower_bound
        self._scale = 1 / (1 - self._lower_bound)
    
    def forward(self, m: torch.Tensor) -> torch.Tensor:
        
        return (torch.clamp(m, self._lower_bound) - self._lower_bound) / self._scale


class SigmoidActivation(MembershipActivation):
    """Inverse sigmoid followed by parameterized forward sigmoid"""

    def __init__(self, n_features: int, positive_scale: bool=False, device='cpu'):
        super().__init__(n_features)
        self._positive_scale = positive_scale
        self.b = torch.nn.Parameter(torch.empty((n_features,), device=device))
        self.s = torch.nn.Parameter(torch.empty((n_features,), device=device))
        torch.nn.init.normal_(self.b, -0.05, 0.05)
        torch.nn.init.uniform_(self.s, 0.5, 1.5)
    
    def forward(self, m: torch.Tensor):

        if m.dim() > 1:
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


class TriangularActivation(MembershipActivation):
    """Warps the membership by a triangular function"""

    def __init__(self, n_features: int, device='cpu'):
        super().__init__(n_features)
        self.b_base = torch.nn.Parameter(torch.empty((n_features,), device=device))
        self.offset_base = torch.nn.Parameter(torch.empty((n_features,), device=device))
        torch.nn.init.normal_(self.b_base, -0.05, 0.05)
        torch.nn.init.normal_(self.offset_base, -0.05, 0.05)
    
    def forward(self, m):

        b = nn.functional.softplus(self.b_base)
        offset = torch.sigmoid(self.offset_base)
        x = m / 2 - offset
        return torch.min((x + b) / b, (b - x) / b)


class Hedge(nn.Module):

    def __init__(self, n_terms: int, dim: int=-1, lower_bound: float=None, upper_bound: float=None):

        super().__init__()
        self._n_terms = n_terms
        if lower_bound is not None and lower_bound < 0:
            raise ValueError(f'Argument lower bound must be greater than 0 not {lower_bound}')
        if upper_bound is not None and upper_bound < lower_bound:
            raise ValueError(f'Argument upper bound must be greater than or equal to lower bound not {upper_bound}')
        
        self._w = nn.parameter.Parameter(torch.ones(n_terms))
        self._upper_bound = upper_bound
        self._lower_bound = lower_bound
        self._dim = dim

    def align(self, w: torch.Tensor, m: torch.Tensor, dim: int=1) -> torch.Tensor:

        shape = [1] * m.dim()
        shape[dim] = 0

        for i, s in enumerate(shape):
            if s == 1:
                w = w.unsqueeze(i)
        return dim

    def forward(self, m: torch.Tensor) -> torch.Tensor:

        w = self.align(torch.clamp(self._w, self._lower_bound, self._upper_bound))
        return m ** w
