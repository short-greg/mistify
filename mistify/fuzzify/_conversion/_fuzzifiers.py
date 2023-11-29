from abc import abstractmethod


import torch
import torch.nn as nn
from torch import clamp

from ._conclude import HypoWeight


class Fuzzifier(nn.Module):

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class Defuzzifier(nn.Module):
    """Defuzzify the input
    """

    @abstractmethod
    def hypo(self, m: torch.Tensor) -> HypoWeight:
        pass

    @abstractmethod
    def conclude(self, value_weight: HypoWeight) -> torch.Tensor:
        pass

    @abstractmethod
    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return self.conclude(self.hypo(m))


class EmbeddingFuzzifier(Fuzzifier):

    def __init__(
        self, terms: int, out_variables: int, f=clamp
    ):
        """Convert labels to fuzzy embeddings

        Args:
            terms (int): The number of terms to output for
            out_variables (int): The number of variables to output for each term
            f (function, optional): A function that maps the output between 0 and 1. Defaults to clamp.
        """
        super().__init__()
        self._terms = terms
        self._embedding = nn.Embedding(
            terms, out_variables
        )
        self.f = f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError('Embedding crispifier only works for two dimensional tensors')
        return self.f(self._embedding(x))
