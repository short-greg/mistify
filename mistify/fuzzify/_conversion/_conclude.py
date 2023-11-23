"""


"""
from dataclasses import dataclass
from abc import abstractmethod
from enum import Enum

import typing

import torch
import torch.nn as nn


@dataclass
class HypoWeight:
    """Structure that defines a hypothesis and its weight
    """

    hypo: torch.Tensor
    weight: torch.Tensor

    def __iter__(self) -> typing.Iterator[torch.Tensor]:

        yield self.hypo
        yield self.weight


class Conclusion(nn.Module):
    """Class that defines several hypotheses 
    """
    @abstractmethod
    def forward(self, value_weight: HypoWeight) -> torch.Tensor:
        pass


class MaxValueConc(Conclusion):
    """Choose the hypothesis with the maximum value
    """

    def forward(self, hypo_weight: HypoWeight) -> torch.Tensor:
        """
        Args:
            hypo_weight (HypoWeight): The hypotheses and their weights

        Returns:
            torch.Tensor: The conclusion
        """
        return torch.max(hypo_weight.hypo, dim=-1)[0]


class MaxConc(Conclusion):
    """Choose the hypothesis with the maximum weight
    """

    def forward(self, hypo_weight: HypoWeight) -> torch.Tensor:
        """
        Args:
            hypo_weight (HypoWeight): The hypotheses and weights

        Returns:
            torch.Tensor: the hypothesis with the maximum weight
        """
        indices = torch.max(hypo_weight.weight, dim=-1, keepdim=True)[1]
        return torch.gather(hypo_weight.hypo, -1, indices).squeeze(dim=-1)


class WeightedAverageConc(Conclusion):
    """Take the weighted average of all the hypotheses
    """

    def forward(self, hypo_weight: HypoWeight) -> torch.Tensor:
        """
        Args:
            hypo_weight (HypoWeight): The hypotheses and weights

        Returns:
            torch.Tensor: the weighted average of the hypotheses
        """
        return (
            torch.sum(hypo_weight.hypo * hypo_weight.weight, dim=-1) 
            / torch.sum(hypo_weight.weight, dim=-1)
        )

class ConcEnum(Enum):

    max = MaxConc
    max_value = MaxValueConc
    weighted_average = WeightedAverageConc

    @classmethod
    def get(cls, conc: typing.Union[Conclusion, str]) -> Conclusion:

        if isinstance(conc, str):
            return ConcEnum[conc].value()
        return conc
