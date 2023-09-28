"""
Functionality for crisp binary sets where 1 is True and 0 is False

"""

import torch

from .._base import CompositionBase, maxmin, ComplementBase, get_comp_weight_size
from torch import nn
from .utils import positives


# TODO:
# I want to simplify all of this
# I have OrNeuron and AndNeuron

class BinaryComposition(CompositionBase):

    def init_weight(self, in_features: int, out_features: int, in_variables: int = None) -> torch.Tensor:
        return positives(get_comp_weight_size(in_features, out_features, in_variables))

    def forward(self, m: torch.Tensor):
        return maxmin(m, self.weight).round()

    def clamp_weights(self):
        self.weight.data = self.weight.data.clamp(0, 1)


class BinaryComplement(ComplementBase):

    def complement(self, m: torch.Tensor):
        return 1 - m


class BinaryElse(nn.Module):

    def __init__(self, dim: int=-1):

        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        y = x.max(dim=self.dim)[0]
        return (1 - y)
