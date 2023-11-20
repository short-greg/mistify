from abc import abstractmethod
from enum import Enum
import typing

import torch.nn as nn
import torch

from .._shapes import Shape


class ShapeImplication(nn.Module):

    @abstractmethod
    def forward(self, shapes: Shape):
        pass


class AreaImplication(ShapeImplication):

    def forward(self, shapes: Shape):
        return torch.cat(
            [shape.areas for shape in shapes], dim=2
        )


class MeanCoreImplication(ShapeImplication):

    def forward(self, shapes: Shape):

        cores = []
        for shape in shapes:
            if shape.mean_cores is None:
                raise ValueError('Cannot calculate mean core if None')
            cores.append(shape.mean_cores)
        return torch.cat(
            cores, dim=2
        )


class CentroidImplication(ShapeImplication):

    def forward(self, shapes: Shape):
        return torch.cat(
            [shape.centroids for shape in shapes], dim=2
        )


class ImplicationEnum(Enum):

    area = AreaImplication
    mean_core = MeanCoreImplication
    centroid = CentroidImplication

    @classmethod
    def get(cls, implication: typing.Union['ShapeImplication', str]) -> ShapeImplication:
        if isinstance(implication, ShapeImplication):
            return implication
        return cls[implication].value()
