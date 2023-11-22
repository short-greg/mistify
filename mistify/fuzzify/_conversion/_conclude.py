from dataclasses import dataclass
from abc import abstractmethod
from enum import Enum

import typing

import torch
import torch.nn as nn


@dataclass
class ValueWeight:

    value: torch.Tensor
    weight: torch.Tensor

    def __iter__(self) -> typing.Iterator[torch.Tensor]:

        yield self.value
        yield self.weight


class Conclusion(nn.Module):

    @abstractmethod
    def forward(self, value_weight: ValueWeight) -> torch.Tensor:
        pass


class MaxValueAcc(Conclusion):

    def forward(self, value_weight: ValueWeight) -> torch.Tensor:

        return torch.max(value_weight.value, dim=-1)[0]


class MaxAcc(Conclusion):

    def forward(self, value_weight: ValueWeight) -> torch.Tensor:

        indices = torch.max(value_weight.weight, dim=-1, keepdim=True)[1]
        return torch.gather(value_weight.value, -1, indices).squeeze(dim=-1)


class WeightedAverageAcc(Conclusion):

    def forward(self, value_weight: ValueWeight) -> torch.Tensor:

        return (
            torch.sum(value_weight.value * value_weight.weight, dim=-1) 
            / torch.sum(value_weight.weight, dim=-1)
        )

class AccEnum(Enum):

    max = MaxAcc
    max_value = MaxValueAcc
    weighted_average = WeightedAverageAcc

    @classmethod
    def get(cls, acc: typing.Union[Conclusion, str]) -> Conclusion:

        if isinstance(acc, str):
            return AccEnum[acc].value()
        return acc
