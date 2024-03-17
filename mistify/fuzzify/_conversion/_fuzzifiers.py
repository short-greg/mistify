from abc import abstractmethod
import torch
from torch import Tensor
import torch.nn as nn
from torch import clamp
import torch.nn.functional

from ._conclude import HypoM
from ._utils import generate_repeat_params, generate_spaced_params


class Fuzzifier(nn.Module):

    def __init__(self, n_terms: int):
        super().__init__()
        self._n_terms = n_terms

    @property
    def n_terms(self) -> int:
        return self._n_terms

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class Defuzzifier(nn.Module):
    """Defuzzify the input
    """

    def __init__(self, n_terms: int):
        super().__init__()
        self._n_terms = n_terms

    @property
    def n_terms(self) -> int:
        return self._n_terms

    @abstractmethod
    def hypo(self, m: torch.Tensor) -> HypoM:
        pass

    @abstractmethod
    def conclude(self, value_weight: HypoM) -> torch.Tensor:
        pass

    @abstractmethod
    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return self.conclude(self.hypo(m))


class EmbeddingFuzzifier(Fuzzifier):

    def __init__(
        self, n_terms: int, out_variables: int, f=clamp
    ):
        """Convert labels to fuzzy embeddings

        Args:
            terms (int): The number of terms to output for
            out_variables (int): The number of variables to output for each term
            f (function, optional): A function that maps the output between 0 and 1. Defaults to clamp.
        """
        super().__init__(n_terms)
        self._embedding = nn.Embedding(
            n_terms, out_variables
        )
        self.f = f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError('Embedding crispifier only works for two dimensional tensors')
        return self.f(self._embedding(x))


class GaussianFuzzifier(Fuzzifier):

    def __init__(self, n_terms: int, in_features: int=None, tunable: bool=True, width: float=None):
        super().__init__(n_terms)
        
        width = width if width is not None else 1.0 / (2 * (n_terms + 1))

        self._loc = torch.nn.parameter.Parameter(
            generate_spaced_params(n_terms + 2, in_features=in_features)[:,:,1:-1]
        )
        self._scale = torch.nn.parameter.Parameter(
            torch.exp(generate_repeat_params(n_terms, width, in_features=in_features)) - 1
        )

        self.tunable(tunable)
        self._fit_loss = nn.MSELoss()

    def forward(self, x: Tensor) -> Tensor:

        scale = torch.nn.functional.softplus(self._scale)
        return torch.exp(
            -0.5 * ((x.unsqueeze(-1) - self._loc) / scale) ** 2
        )

    # updated the formula so need to update here
    def integral(self, x: torch.Tensor):

        return self._scale * torch.tensor(-torch.pi / 2.0, device=x.device) * (
            torch.erf((self._loc - x) / (self._scale * torch.sqrt(torch.tensor(2.0))) 
            ))

    def hypo(self, m: torch.Tensor) -> HypoM:
        
        # get the lower bound
        inv = torch.sqrt(-torch.log(m) * (2 * self._scale ** 2))
        lhs = -inv + self._loc
        rhs = inv + self._loc
        sum_left = self.integral(lhs)

        x = -torch.sqrt(-torch.log(m))

        sum_left = (torch.sqrt(torch.tensor(torch.pi)) /  2 ) * (
            (1 + torch.erf(x))
        )
        sum_rec = (rhs - lhs) * m
        return HypoM(
            sum_left * 2 + sum_rec, m
        )

    def defuzzify(self, x, m):
        """
        Defuzzify the membership tensor using the Center of Sums method.
        :param x: Input tensor that was fuzzified.
        :param m: Membership tensor resulting from fuzzification.
        :return: Defuzzified value using the Center of Sums method.
        """
        # Calculate the weighted sum of the input values, using membership values as weights
        weighted_sum = torch.sum(x * m)
        # Sum of the membership values
        sum_of_memberships = torch.sum(m)
        # Compute the weighted average (center of sums)
        if sum_of_memberships > 0:
            cos_value = weighted_sum / sum_of_memberships
        else:
            cos_value = torch.tensor(0.0)  # Fallback in case of zero division
        return cos_value
    
    def tunable(self, tunable: bool=True):

        for p in self.parameters():
            p.requires_grad_(tunable)

    def resp_loss(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        
        # assume that each of the components has some degree of
        # responsibility
        t = t.clamp(min=0.0) + torch.rand_like(t) * 1e-6
        r = t / t.sum(dim=-1, keepdim=True)
        Nk = r.sum(dim=0, keepdim=True)
        target_loc = (r * x[:,:,None]).sum(dim=0, keepdim=True) / Nk

        target_scale = (r * (x[:,:,None] - target_loc) ** 2).sum(dim=0, keepdim=True) / Nk

        cur_scale = torch.nn.functional.softplus(self._scale)
        
        scale_loss = self._fit_loss(cur_scale, target_scale.detach())
        loc_loss = self._fit_loss(self._loc, target_loc.detach())
        return scale_loss + loc_loss
