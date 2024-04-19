"""


"""
from abc import abstractmethod
from enum import Enum

import typing

from ._hypo import HypoM
import torch
import torch.nn as nn


class Conclusion(nn.Module):
    """Class that defines several hypotheses 
    """
    @abstractmethod
    def forward(self, value_weight: HypoM) -> torch.Tensor:
        pass


class MaxValueConc(Conclusion):
    """Choose the hypothesis with the maximum value
    """

    def forward(self, hypo_weight: HypoM) -> torch.Tensor:
        """
        Args:
            hypo_weight (HypoM): The hypotheses and their weights

        Returns:
            torch.Tensor: The conclusion
        """
        return torch.max(hypo_weight.hypo, dim=-1)[0]


class MaxConc(Conclusion):
    """Choose the hypothesis with the maximum weight
    """

    def forward(self, hypo_weight: HypoM) -> torch.Tensor:
        """
        Args:
            hypo_weight (HypoM): The hypotheses and weights

        Returns:
            torch.Tensor: the hypothesis with the maximum weight
        """
        indices = torch.max(hypo_weight.m, dim=-1, keepdim=True)[1]
        return torch.gather(hypo_weight.hypo, -1, indices).squeeze(dim=-1)


class WeightedAverageConc(Conclusion):
    """Take the weighted average of all the hypotheses
    """

    def forward(self, hypo_weight: HypoM) -> torch.Tensor:
        """
        Args:
            hypo_weight (HypoM): The hypotheses and weights

        Returns:
            torch.Tensor: the weighted average of the hypotheses
        """
        return (
            torch.sum(hypo_weight.hypo * hypo_weight.m, dim=-1) 
            / torch.sum(hypo_weight.m, dim=-1)
        )


class AverageConc(Conclusion):
    """Take the weighted average of all the hypotheses
    """

    def forward(self, hypo_weight: HypoM) -> torch.Tensor:
        """
        Args:
            hypo_weight (HypoM): The hypotheses and weights

        Returns:
            torch.Tensor: the weighted average of the hypotheses
        """
        return (
            torch.mean(hypo_weight.hypo, dim=-1)
        )


class ConcEnum(Enum):

    max = MaxConc
    max_value = MaxValueConc
    weighted_average = WeightedAverageConc
    average = AverageConc

    @classmethod
    def get(cls, conc: typing.Union[Conclusion, str]) -> Conclusion:

        if isinstance(conc, str):
            return ConcEnum[conc].value()
        return conc
