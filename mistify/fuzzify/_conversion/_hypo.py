# 1st party
from abc import abstractmethod
from enum import Enum
import typing

# 3rd party
import torch.nn as nn
import torch

# local
from .._shapes import Shape, Nonmonotonic, Monotonic


class ShapeHypothesis(nn.Module):
    """A hypothesizer generates candidates for defuzzification.
    """

    def __init__(self, truncate: bool=False) -> None:
        super().__init__()
        self.truncate = truncate

    @abstractmethod
    def forward(self, *shapes: Shape) -> torch.Tensor:
        pass


class AreaHypothesis(ShapeHypothesis):
    """Use the area under the fuzzy set
    """

    def forward(self, *shapes: Nonmonotonic, m: torch.Tensor) -> torch.Tensor:
        
        return torch.cat(
            [shape.areas(m, self.truncate) for shape in shapes], dim=2
        )


class MeanCoreHypothesis(ShapeHypothesis):
    """Use the mean value of the 'core' of the fuzzy set for the hypothesis
    """

    def forward(self, *shapes: Nonmonotonic, m: torch.Tensor):

        cores = []
        for shape in shapes:
            # if shape.mean_cores is None:
            #     raise ValueError('Cannot calculate mean core if None')
            cores.append(shape.mean_cores(m, self.truncate))
        return torch.cat(
            cores, dim=2
        )


class MinCoreHypothesis(ShapeHypothesis):
    """Use the min value of the 'core' of the fuzzy set for the hypothesis. Use for 'Monotonic'
    """

    def forward(self, *shapes: Monotonic, m: torch.Tensor):

        cores = []
        for shape in shapes:
            # if shape.min_cores is None:
            #     raise ValueError('Cannot calculate mean core if None')
            cores.append(shape.min_cores(m))
        return torch.cat(
            cores, dim=2
        )


class CentroidHypothesis(ShapeHypothesis):
    """Use the centroid of the fuzzy set for the hypothesis
    """

    def forward(self, *shapes: Nonmonotonic, m: torch.Tensor):
        return torch.cat(
            [shape.centroids(m, self.truncate) for shape in shapes], dim=2
        )


class HypothesisEnum(Enum):

    area = AreaHypothesis
    mean_core = MeanCoreHypothesis
    centroid = CentroidHypothesis
    min_core = MinCoreHypothesis

    @classmethod
    def get(cls, hypothesis: typing.Union['ShapeHypothesis', str]) -> ShapeHypothesis:
        if isinstance(hypothesis, ShapeHypothesis):
            return hypothesis
        return cls[hypothesis].value()
