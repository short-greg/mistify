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

        self._z = None

        self._loc = generate_spaced_params(n_terms + 2, in_features=in_features)[:,:,1:-1]
        self._scale = torch.log(
                torch.exp(generate_repeat_params(n_terms, width, in_features=in_features)) - 1)
        if tunable:
            self._loc = torch.nn.parameter.Parameter(self._loc)
            self._scale = torch.nn.parameter.Parameter(self._scale)

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

    def partial_fit(self, x: torch.Tensor, t: torch.Tensor=None, accumulate: bool=True):
        
        # TODO: Test this code
        # Train it as a GMM
        # Can set targets here

        if t is None:
            t = self(x)
        else:
            t = t.clamp(0, 1)
        if self._z is None:
            self._z = 1 / self._n_terms
        t = self._z * t
        r = t / t.sum(dim=-1, keepdim=True)
        Nk = r.sum(dim=0, keepdim=True)

        new_loc = (t * x[:,:,None]).sum(dim=0, keepdim=True) / Nk
        new_scale = (t * (self.x[:,:,None] - self._loc) ** 2).sum(dim=0, keepdim=True) / Nk
        cur_scale = torch.nn.functional.softplus(self._scale)

        target_scale = torch.exp(0.98 * cur_scale + 0.02 * new_scale -1)
        target_loc = 0.98 * self._loc + 0.02 * new_loc
        self._z = 0.98 * self._z + 0.02 * Nk / Nk.sum(dim=-1, keepdim=True)
        
        # update using an "optimizer" or simply update
        # probably add a utility for this
        if accumulate:
            dscale = self._scale - target_scale
            dloc = self._loc - target_loc
            if self._scale.grad is None:
                self._scale.grad = dscale
                self._loc.grad = dloc
            else:
                self._scale.grad.data = self._scale.grad.data + dscale
                self._loc.grad.data = self._loc.grad.data + dscale
        else:
            self._scale.data = target_scale
            self._loc.data = target_loc
