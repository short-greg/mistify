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

    @abstractmethod
    def forward(self, *shapes: Shape) -> torch.Tensor:
        pass


class AreaHypothesis(ShapeHypothesis):
    """Use the area under the fuzzy set
    """

    def forward(self, *shapes: typing.List[Shape]) -> torch.Tensor:
        
        return torch.cat(
            [shape.areas for shape in shapes], dim=2
        )


class MeanCoreHypothesis(ShapeHypothesis):
    """Use the mean value of the 'core' of the fuzzy set for the hypothesis
    """

    def forward(self, *shapes: typing.List[Nonmonotonic]):

        cores = []
        for shape in shapes:
            if shape.mean_cores is None:
                raise ValueError('Cannot calculate mean core if None')
            cores.append(shape.mean_cores)
        return torch.cat(
            cores, dim=2
        )


class MinCoreHypothesis(ShapeHypothesis):
    """Use the min value of the 'core' of the fuzzy set for the hypothesis. Use for 'Monotonic'
    """

    def forward(self, *shapes: typing.List[Monotonic]):

        cores = []
        for shape in shapes:
            if shape.min_cores is None:
                raise ValueError('Cannot calculate mean core if None')
            cores.append(shape.min_cores)
        return torch.cat(
            cores, dim=2
        )


class CentroidHypothesis(ShapeHypothesis):
    """Use the centroid of the fuzzy set for the hypothesis
    """

    def forward(self, *shapes: typing.List[Nonmonotonic]):
        return torch.cat(
            [shape.centroids for shape in shapes], dim=2
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
