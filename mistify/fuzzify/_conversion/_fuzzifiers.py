from abc import abstractmethod


import torch
import torch.nn as nn
from torch import clamp

from ._conclude import HypoWeight
from ._converters import FuzzyConverter


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


class ConverterDefuzzifier(Defuzzifier):

    def __init__(self, converter: FuzzyConverter):
        """Wrap a FuzzyConverter to create a defuzzifier

        Args:
            converter (FuzzyConverter): The fuzzy converter to wrap
        """
        super().__init__()
        self.converter = converter

    def hypo(self, m: torch.Tensor) -> HypoWeight:
        """Calculate the hypothesis

        Args:
            m (torch.Tensor): The fuzzy set input

        Returns:
            HypoWeight: The hypothesis and weight
        """
        return self.converter.hypo(m)

    def conclude(self, hypo_weight: HypoWeight) -> torch.Tensor:
        """

        Args:
            hypo_weight (HypoWeight): _description_

        Returns:
            torch.Tensor: The defuzzified value
        """
        return self.converter.conclude(hypo_weight)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return self.converter.defuzzify(m)


class ConverterFuzzifier(Fuzzifier):

    def __init__(self, converter: FuzzyConverter):
        """Wrap a FuzzyConverter to create a fuzzifier

        Args:
            converter (FuzzyConverter): The fuzzy converter to wrap
        """
        super().__init__()
        self.converter = converter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The input

        Args:
            x (torch.Tensor): the input

        Returns:
            torch.Tensor: the fuzzified input
        """
        return self.converter.fuzzify(x)


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
