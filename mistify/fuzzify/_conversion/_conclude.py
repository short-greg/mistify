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
    def forward(self, hypo_weight: HypoWeight) -> torch.Tensor:
        pass

    @property
    def n_terms(self) -> int:
        return self._n_terms
    
    @property
    def n_vars(self) -> int:
        return self._n_vars


class FlattenConc(Conclusion):
    """Class that defines several hypotheses 
    """
    def __init__(self, conclusion: 'Conclusion') -> None:
        if conclusion.n_vars is None:
            n_vars = None
            n_terms = conclusion.n_terms
        else:
            n_terms = conclusion.n_terms * conclusion.n_vars
            n_vars = 1
        super().__init__(n_terms, n_vars)
        self._conclusion = conclusion

    def forward(self, hypo_weight: HypoWeight) -> torch.Tensor:
        
        shape = list(hypo_weight.hypo.shape)
        shape[-2] = 1
        shape[-1] = -1
        hypo_weight = HypoWeight(
            hypo_weight.hypo.view(shape),
            hypo_weight.weight.view(shape)
        )
        return self._conclusion.forward(
            hypo_weight
        )


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
    def get(
        cls, conc: typing.Union[Conclusion, str], 
        n_terms: int=None, n_vars: int=None, 
        flatten: bool=False
    ) -> Conclusion:

        if isinstance(conc, str):
            conc = ConcEnum[conc].value(n_terms, n_vars)
        if flatten:
            conc = FlattenConc(conc)
        return conc
