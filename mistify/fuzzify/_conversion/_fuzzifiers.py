from abc import abstractmethod, ABC
import torch
import torch.nn as nn
from torch import clamp
import torch.nn.functional

from ._conclude import HypoWeight


class Fuzzifier(nn.Module, ABC):

    def __init__(self, n_terms: int, n_vars: int=None):
        super().__init__()
        self._n_terms = n_terms
        self._n_vars = n_vars

    @property
    def n_terms(self) -> int:
        return self._n_terms
    
    @property
    def n_vars(self) -> int:
        return self._n_vars

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class Defuzzifier(nn.Module):
    """Defuzzify the input
    """

    def __init__(self, n_terms: int, n_vars: int=None):
        super().__init__()
        self._n_terms = n_terms
        self._n_vars = n_vars

    @property
    def n_terms(self) -> int:
        return self._n_terms

    @abstractmethod
    def hypo(self, m: torch.Tensor) -> HypoWeight:
        pass

    @abstractmethod
    def conclude(self, value_weight: HypoWeight) -> torch.Tensor:
        pass

    def forward(self, m: torch.Tensor, weight_override: torch.Tensor=None) -> torch.Tensor:
        
        hypothesis = self.hypo(m)
        hypothesis.weight = weight_override or hypothesis.weight
        return self.conclude(hypothesis)


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
        super().__init__(n_terms, out_variables)
        self._embedding = nn.Embedding(
            n_terms, out_variables
        )
        self.f = f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError('Embedding crispifier only works for two dimensional tensors')
        return self.f(self._embedding(x))
