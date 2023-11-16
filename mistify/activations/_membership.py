import torch.nn as nn
import torch


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

        for s in shape:
            if s == 1:
                w.unsqueeze(s)
        return dim

    def forward(self, m: torch.Tensor) -> torch.Tensor:

        w = self.align(torch.clamp(self._w, self._lower_bound, self._upper_bound))
        return m ** w


