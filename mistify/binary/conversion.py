# 1st party
from abc import abstractmethod
from dataclasses import dataclass
import typing

# 3rd party
import torch
import torch.nn as nn
import torch.nn.functional

# local
from ..base import ValueWeight, Accumulator, MaxValueAcc


class CrispConverter(nn.Module):
    """Convert tensor to crisp set
    """

    @abstractmethod
    def crispify(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def imply(self, m: torch.Tensor) -> ValueWeight:
        pass

    @abstractmethod
    def accumulate(self, value_weight: ValueWeight) -> torch.Tensor:
        pass

    def decrispify(self, m: torch.Tensor) -> torch.Tensor:
        return self.accumulate(self.imply(m))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.crispify(x)


class Crispifier(nn.Module):

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class Decrispifier(nn.Module):

    @abstractmethod
    def imply(self, m: torch.Tensor) -> ValueWeight:
        pass

    @abstractmethod
    def accumulate(self, value_weight: ValueWeight) -> torch.Tensor:
        pass

    def fowrard(self, m: torch.Tensor) -> torch.Tensor:
        return self.accumulate(self.imply(m))


class StepCrispConverter(CrispConverter):

    def __init__(
        self, out_variables: int, out_terms: int, 
        accumulator: Accumulator=None
    ):
        super().__init__()

        self.threshold = nn.parameter.Parameter(
            torch.randn(out_variables, out_terms)
        )

        self._accumulator = accumulator or MaxValueAcc()

    def crispify(self, x: torch.Tensor) -> torch.Tensor:
        return (x[:,:,None] >= self.threshold[None]).type_as(x)

    def imply(self, m: torch.Tensor) -> ValueWeight:
        
        return ValueWeight(
            m * self.threshold[None], m
        )

    def accumulate(self, value_weight: ValueWeight) -> torch.Tensor:
        return self._accumulator.forward(value_weight)




# Not sure why i have strides
# def get_strided_indices(n_points: int, stride: int, step: int=1):
#     initial_indices = torch.arange(0, n_points).as_strided((n_points - stride + 1, stride), (1, 1))
#     return initial_indices[torch.arange(0, len(initial_indices), step)]


# def stride_coordinates(coordinates: torch.Tensor, stride: int, step: int=1):

#     dim2_index = get_strided_indices(coordinates.size(2), stride, step)
#     return coordinates[:, :, dim2_index]

