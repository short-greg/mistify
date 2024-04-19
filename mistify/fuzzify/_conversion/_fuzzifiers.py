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
