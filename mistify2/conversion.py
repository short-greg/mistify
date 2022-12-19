import torch
import torch.nn.functional as nn_func
import torch.nn as nn
from .fuzzy import FuzzySet
from .crisp import CrispSet
from abc import abstractmethod
from dataclasses import dataclass
import typing


@dataclass
class ValueWeight:

    weight: torch.Tensor
    value: torch.Tensor

    def __iter__(self) -> typing.Iterator[torch.Tensor]:

        yield self.weight
        yield self.value


class FuzzyConverter(nn.Module):

    @abstractmethod
    def fuzzify(self, x: torch.Tensor) -> FuzzySet:
        pass

    @abstractmethod
    def imply(self, m: FuzzySet) -> ValueWeight:
        pass

    @abstractmethod
    def accumulate(self, value_weight: ValueWeight) -> torch.Tensor:
        pass

    def defuzzify(self, m: FuzzySet) -> torch.Tensor:
        return self.accumulate(self.imply(m))

    def forward(self, x: torch.Tensor) -> FuzzySet:
        return self.fuzzify(x)


class CrispConverter(nn.Module):

    @abstractmethod
    def crispify(self, x: torch.Tensor) -> CrispSet:
        pass

    @abstractmethod
    def imply(self, m: FuzzySet) -> ValueWeight:
        pass

    @abstractmethod
    def accumulate(self, value_weight: ValueWeight) -> torch.Tensor:
        pass

    def decrispify(self, m: CrispSet) -> torch.Tensor:
        return self.accumulate(self.imply(m))

    def forward(self, x: torch.Tensor) -> CrispSet:
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
    def imply(self, m: FuzzySet) -> ValueWeight:
        pass

    @abstractmethod
    def accumulate(self, value_weight: ValueWeight) -> torch.Tensor:
        pass

    @abstractmethod
    def forward(self, m: FuzzySet) -> torch.Tensor:
        return self.accumulate(self.imply(m))


class Decrispifier(nn.Module):

    @abstractmethod
    def imply(self, m: FuzzySet) -> ValueWeight:
        pass

    @abstractmethod
    def accumulate(self, value_weight: ValueWeight) -> torch.Tensor:
        pass

    def fowrard(self, m: CrispSet) -> torch.Tensor:
        return self.accumulate(self.imply(m))


class Accumulator(nn.Module):

    @abstractmethod
    def forward(self, value_weight: ValueWeight) -> torch.Tensor:
        pass


class MaxValueAcc(Accumulator):

    def forward(self, value_weight: ValueWeight) -> torch.Tensor:

        return torch.max(value_weight.value, dim=-1)[0]


class MaxAcc(Accumulator):

    def forward(self, value_weight: ValueWeight) -> torch.Tensor:

        indices = torch.max(value_weight.weight, dim=-1, keepdim=True)[1]
        return torch.gather(value_weight.value, -1, indices).squeeze(dim=-1)


class WeightedAverageAcc(Accumulator):

    def forward(self, value_weight: ValueWeight) -> torch.Tensor:

        return (
            torch.sum(value_weight.value * value_weight.weight, dim=-1) 
            / torch.sum(value_weight.weight, dim=-1)
        )


class SigmoidFuzzyConverter(FuzzyConverter):

    def __init__(self, out_variables: int, out_features: int, eps: float=1e-7, accumulator: Accumulator=None):

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

    def imply(self, m: FuzzySet) -> ValueWeight:
        return ValueWeight(m, (-torch.log(
            1 / (m.data + self.eps) - 1
        ) / self.weight[None] + self.bias[None]))

    # def defuzzify(self, m: FuzzySet) -> torch.Tensor:
    #     # x = ln(y/(1-y))
    #     return self. # mean(dim=2)


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
