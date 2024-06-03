# 1st party
from abc import abstractmethod
from enum import Enum
import typing
from dataclasses import dataclass

# 3rd party
import torch.nn as nn
import torch

# local
from .._shapes import Shape, Nonmonotonic, Monotonic


@dataclass
class HypoWeight:
    """Structure that defines a hypothesis and its weight
    """

    hypo: torch.Tensor
    weight: torch.Tensor

    def __iter__(self) -> typing.Iterator[torch.Tensor]:

        yield self.hypo
        yield self.weight
    

class ShapeHypothesis(nn.Module):
    """A hypothesizer generates candidates for defuzzification.
    """

    def __init__(self, truncate: bool=False) -> None:
        super().__init__()
        self.truncate = truncate

    @abstractmethod
    def forward(self, *shapes: Shape) -> HypoWeight:
        pass


class AreaHypothesis(ShapeHypothesis):
    """Use the area under the fuzzy set
    """

    def forward(self, shapes: typing.List[Nonmonotonic], m: torch.Tensor) -> HypoWeight:
        
        i = 0
        result = []
        centroids = []
        for shape in shapes:
            result.append(
                shape.areas(m[:,:,i:shape.n_terms + i], self.truncate)
            )
            centroids.append(
                shape.centroids(m[:,:,i:shape.n_terms + i], self.truncate)
            )

            i += shape.n_terms

        weight = torch.cat(
            result, dim=2
        )
        hypo = torch.cat(
            centroids, dim=2
        )
        return HypoWeight(hypo, weight)


class MeanCoreHypothesis(ShapeHypothesis):
    """Use the mean value of the 'core' of the fuzzy set for the hypothesis
    """

    def forward(self, shapes: typing.List[Nonmonotonic], m: torch.Tensor) -> HypoWeight:

        i = 0
        result = []
        for shape in shapes:
            result.append(
                shape.mean_cores(m[:,:,i:shape.n_terms + i], self.truncate)
            )

            i += shape.n_terms
        return HypoWeight(torch.cat(
            result, dim=2
        ), m)


class MinCoreHypothesis(ShapeHypothesis):
    """Use the min value of the 'core' of the fuzzy set for the hypothesis. Use for 'Monotonic'
    """

    def forward(self, shapes: typing.List[Monotonic], m: torch.Tensor) -> HypoWeight:

        i = 0
        result = []
        for shape in shapes:
            result.append(
                shape.min_cores(m[:,:,i:shape.n_terms + i])
            )

            i += shape.n_terms
        
        return HypoWeight(torch.cat(
            result, dim=2
        ), m)


class CentroidHypothesis(ShapeHypothesis):
    """Use the centroid of the fuzzy set for the hypothesis
    """

    def forward(self, shapes: typing.List[Nonmonotonic], m: torch.Tensor) -> HypoWeight:
        
        i = 0
        result = []
        for shape in shapes:
            result.append(
                shape.centroids(m[:,:,i:shape.n_terms + i], self.truncate)
            )

            i += shape.n_terms
        return HypoWeight(torch.cat(
            result, dim=2
        ), m)


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
