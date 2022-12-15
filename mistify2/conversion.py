import torch
import torch.nn.functional as nn_func
import torch.nn as nn
from .fuzzy import FuzzySet
from .crisp import CrispSet
from abc import abstractmethod


class FuzzyConverter(nn.Module):

    @abstractmethod
    def fuzzify(self, x: torch.Tensor) -> FuzzySet:
        pass

    @abstractmethod
    def defuzzify(self, m: FuzzySet) -> torch.Tensor:
        pass

    def forward(self, x: torch.Tensor) -> FuzzySet:
        return self.fuzzify(x)


class CrispConverter(nn.Module):

    @abstractmethod
    def crispify(self, x: torch.Tensor) -> CrispSet:
        pass

    @abstractmethod
    def decrispify(self, x: torch.Tensor) -> CrispSet:
        pass

    def forward(self, x: torch.Tensor) -> FuzzySet:
        return self.crispify(x)


class Crispifier(nn.Module):

    @abstractmethod
    def forward(self, x: torch.Tensor) -> CrispSet:
        pass


class Fuzzifier(nn.Module):

    @abstractmethod
    def forward(self, x: torch.Tensor) -> FuzzySet:
        pass


class Defuzzifier(nn.Module):

    @abstractmethod
    def forward(self, m: FuzzySet) -> torch.Tensor:
        pass


class Decrispifier(nn.Module):

    @abstractmethod
    def fowrard(self, m: CrispSet) -> torch.Tensor:
        pass


class SigmoidFuzzyConverter(FuzzyConverter):

    def __init__(self, out_variables: int, out_features: int, eps: float=1e-7):

        super().__init__()
        self.weight = nn.parameter.Parameter(
            torch.randn(out_variables, out_features)
        )
        self.bias = nn.parameter.Parameter(
            torch.randn(out_variables, out_features)
        )
        self.eps = eps

    def fuzzify(self, x: torch.Tensor) -> FuzzySet:
        return nn_func.sigmoid(
            -(x[:,:,None] - self.bias[None]) * self.weight[None]
        )

    def defuzzify(self, m: FuzzySet) -> torch.Tensor:
        # x = ln(y/(1-y))

        return (-torch.log(
            1 / (m.data + self.eps) - 1
        ) / self.weight[None] + self.bias[None]).mean(dim=2)


class SigmoidDefuzzifier(Defuzzifier):

    def __init__(self, converter: SigmoidFuzzyConverter):

        super().__init__()
        self._converter = converter

    def forward(self, m: FuzzySet):
        return self._converter.defuzzify(m)
    
    @classmethod
    def build(cls, out_variables: int, out_features: int, eps: float=1e-7):
        return SigmoidDefuzzifier(
            SigmoidFuzzyConverter(out_variables, out_features, eps)
        )
