from abc import abstractmethod
from enum import Enum
import typing

import torch.nn as nn
import torch

from .._shapes import Shape, Nonmonotonic, Monotonic


class ShapeHypothesis(nn.Module):

    @abstractmethod
    def forward(self, *shapes: Shape):
        pass


class AreaHypothesis(ShapeHypothesis):

    def forward(self, *shapes: typing.List[Shape]):
        return torch.cat(
            [shape.areas for shape in shapes], dim=2
        )


class MeanCoreHypothesis(ShapeHypothesis):

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
