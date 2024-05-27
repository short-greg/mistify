"""


"""
from abc import abstractmethod
from enum import Enum

import typing

from ._hypo import HypoWeight
import torch
import torch.nn as nn


class Conclusion(nn.Module):
    """Class that defines several hypotheses 
    """
    def __init__(self, n_terms: int, n_vars: int=None) -> None:
        super().__init__()
        self._n_terms = n_terms
        self._n_vars = n_vars

    @abstractmethod
    def forward(self, hypo_m: HypoWeight) -> torch.Tensor:
        pass


class MaxValueConc(Conclusion):
    """Choose the hypothesis with the maximum value
    """

    def forward(self, hypo_m: HypoWeight) -> torch.Tensor:
        """
        Args:
            hypo_w (HypoW): The hypotheses and their weights

        Returns:
            torch.Tensor: The conclusion
        """
        return torch.max(hypo_m.hypo, dim=-1)[0]


class MaxConc(Conclusion):
    """Choose the hypothesis with the maximum weight
    """

    def forward(self, hypo_m: HypoWeight) -> torch.Tensor:
        """
        Args:
            hypo_weight (HypoM): The hypotheses and weights

        Returns:
            torch.Tensor: the hypothesis with the maximum weight
        """
        indices = torch.max(hypo_m.weight, dim=-1, keepdim=True)[1]
        return torch.gather(hypo_m.hypo, -1, indices).squeeze(dim=-1)


class WeightedMAverageConc(Conclusion):
    """Take the weighted average of all the hypotheses
    """

    def forward(self, hypo_m: HypoWeight) -> torch.Tensor:
        """
        Args:
            hypo_weight (HypoM): The hypotheses and weights

        Returns:
            torch.Tensor: the weighted average of the hypotheses
        """
        return (
            torch.sum(hypo_m.hypo * hypo_m.weight, dim=-1) 
            / torch.sum(hypo_m.weight, dim=-1)
        )


class AverageConc(Conclusion):
    """Take the weighted average of all the hypotheses
    """

    def forward(self, hypo_m: HypoWeight) -> torch.Tensor:
        """
        Args:
            hypo_weight (HypoM): The hypotheses and weights

        Returns:
            torch.Tensor: the weighted average of the hypotheses
        """
        return (
            torch.mean(hypo_m.hypo, dim=-1)
        )


class WeightedPAverageConc(Conclusion):
    """Take the weighted average of all the hypotheses using learned
    parameters
    """
    
    def __init__(self, n_terms: int, n_vars: int=None) -> None:
        n_vars = n_vars or 1
        super().__init__(n_terms, n_vars)
        shape = [n_vars, n_terms]
        self.layer_weight = nn.parameter.Parameter(
            torch.randn(shape) * 0.025
        )
        self.layer_weightf = nn.Softmax(dim=-1)

    def forward(self, hypo_m: HypoWeight) -> torch.Tensor:
        """
        Args:
            hypo_weight (HypoM): The hypotheses and weights

        Returns:
            torch.Tensor: the weighted average of the hypotheses
        """
        w = self.layer_weightf(self.layer_weight)[None]
        return (
            torch.sum(hypo_m.hypo * w, dim=-1)
        )


class ConcEnum(Enum):

    max = MaxConc
    max_value = MaxValueConc
    weighted_m_average = WeightedMAverageConc
    average = AverageConc
    weighted_p_average = WeightedPAverageConc

    @classmethod
    def get(cls, conc: typing.Union[Conclusion, str], n_terms: int=None, n_vars: int=None) -> Conclusion:

        if isinstance(conc, str):
            return ConcEnum[conc].value(n_terms, n_vars)
        return conc
