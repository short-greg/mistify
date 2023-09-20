# 1st party
from abc import abstractmethod
from dataclasses import dataclass
import typing

# 3rd party
import torch
import torch.nn as nn
import torch.nn.functional

# local
from . import membership as memb
from .membership import Shape


# 1st party
from abc import abstractmethod
from dataclasses import dataclass
import typing

# 3rd party
import torch
import torch.nn as nn
import torch.nn.functional

# local
from . import membership as memb
from .membership import Shape


@dataclass
class ValueWeight:

    value: torch.Tensor
    weight: torch.Tensor

    def __iter__(self) -> typing.Iterator[torch.Tensor]:

        yield self.value
        yield self.weight


def get_implication(implication: typing.Union['ShapeImplication', str]):

    if isinstance(implication, ShapeImplication):
        return implication
    if implication == 'area':
        return AreaImplication()
    if implication == 'mean_core':
        return MeanCoreImplication()
    if implication == 'centroid':
        return CentroidImplication()
    raise ValueError(f"Name {implication} cannot be created")


class ShapeImplication(nn.Module):

    @abstractmethod
    def forward(self, *shapes: memb.Shape):
        pass


class AreaImplication(nn.Module):

    def forward(self, *shapes: memb.Shape):
        return torch.cat(
            [shape.areas for shape in shapes], dim=2
        )


class MeanCoreImplication(nn.Module):

    def forward(self, *shapes: memb.Shape):
        return torch.cat(
            [shape.mean_cores for shape in shapes], dim=2
        )


class CentroidImplication(nn.Module):

    def forward(self, *shapes: memb.Shape):
        return torch.cat(
            [shape.centroids for shape in shapes], dim=2
        )


def get_strided_indices(n_points: int, stride: int, step: int=1):
    initial_indices = torch.arange(0, n_points).as_strided((n_points - stride + 1, stride), (1, 1))
    return initial_indices[torch.arange(0, len(initial_indices), step)]


def stride_coordinates(coordinates: torch.Tensor, stride: int, step: int=1):

    dim2_index = get_strided_indices(coordinates.size(1), stride, step)
    return coordinates[:, dim2_index]


class Accumulator(nn.Module):

    @abstractmethod
    def forward(self, value_weight: ValueWeight) -> torch.Tensor:
        pass


class MaxValueAcc(Accumulator):

    def forward(self, value_weight: ValueWeight) -> torch.Tensor:

        return torch.max(value_weight.value, dim=-1)[0]


class MaxAcc(Accumulator):

    def forward(self, value_weight: ValueWeight) -> torch.Tensor:

        indices = torch.max(value_weight.weight, dim=-1, keepdim=True)[1]
        return torch.gather(value_weight.value, -1, indices).squeeze(dim=-1)


class WeightedAverageAcc(Accumulator):

    def forward(self, value_weight: ValueWeight) -> torch.Tensor:

        return (
            torch.sum(value_weight.value * value_weight.weight, dim=-1) 
            / torch.sum(value_weight.weight, dim=-1)
        )


