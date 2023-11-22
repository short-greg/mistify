from abc import abstractmethod


import torch
import torch.nn as nn
from torch import clamp

from ._conclude import ValueWeight, Conclusion
from ._converters import FuzzyConverter # SigmoidFuzzyConverter, RangeFuzzyConverter


class Fuzzifier(nn.Module):

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class Defuzzifier(nn.Module):

    @abstractmethod
    def hypo(self, m: torch.Tensor) -> ValueWeight:
        pass

    @abstractmethod
    def accumulate(self, value_weight: ValueWeight) -> torch.Tensor:
        pass

    @abstractmethod
    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return self.accumulate(self.hypo(m))


class ConverterDecrispifier(Defuzzifier):

    def __init__(self, crisp_converter: FuzzyConverter):

        super().__init__()
        self.crisp_converter = crisp_converter

    def hypo(self, m: torch.Tensor) -> ValueWeight:
        return self.crisp_converter.hypo(m)

    def accumulate(self, value_weight: ValueWeight) -> torch.Tensor:
        return self.crisp_converter.accumulate(value_weight)


class ConverterCrispifier(Fuzzifier):

    def __init__(self, crisp_converter: FuzzyConverter):

        super().__init__()
        self.crisp_converter = crisp_converter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.crisp_converter.crispify(x)


# class SigmoidDefuzzifier(Defuzzifier):

#     def __init__(self, converter: SigmoidFuzzyConverter):

#         super().__init__()
#         self.converter = converter

#     def forward(self, m: torch.Tensor):
#         return self.converter.defuzzify(m)
    
#     @classmethod
#     def build(cls, out_variables: int, out_terms: int, eps: float=1e-7, conclusion: Conclusion=None):
#         return SigmoidDefuzzifier(
#             SigmoidFuzzyConverter(out_variables, out_terms, eps, conclusion)
#         )


# class RangeDefuzzifier(Defuzzifier):

#     def __init__(self, converter: RangeFuzzyConverter):

#         super().__init__()
#         self.converter = converter

#     def forward(self, m: torch.Tensor):
#         return self.converter.defuzzify(m)
    
#     @classmethod
#     def build(cls, out_variables: int, out_terms: int, conclusion: Conclusion=None):
#         return RangeDefuzzifier(
#             RangeFuzzyConverter(out_variables, out_terms, conclusion)
#         )


class ConverterDefuzzifier(Defuzzifier):

    def __init__(self, converter: FuzzyConverter):
        super().__init__()
        self.converter = converter

    def hypo(self, m: torch.Tensor) -> ValueWeight:
        return self.converter.hypo(m)

    def accumulate(self, value_weight: ValueWeight) -> torch.Tensor:
        return self.converter.accumulate(value_weight)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return self.converter.defuzzify(m)


class ConverterFuzzifier(Fuzzifier):

    def __init__(self, converter: FuzzyConverter):
        super().__init__()
        self.converter = converter

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return self.converter.fuzzify(m)


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


# class SignedCrispifier(Crispifier):

#     def __init__(self, boolean_crispifier: Crispifier):
#         super().__init__()
#         self._crispifier = boolean_crispifier

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
        
#         return functional.to_signed(self._crispifier(x))
