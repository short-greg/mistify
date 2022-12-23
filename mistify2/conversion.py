import torch
import torch.nn.functional as nn_func
import torch.nn as nn
from .fuzzy import FuzzySet
from .crisp import CrispSet
from abc import abstractmethod
from dataclasses import dataclass
import typing
import membership as memb
from .membership import Shape
import typing


@dataclass
class ValueWeight:

    weight: torch.Tensor
    value: torch.Tensor

    def __iter__(self) -> typing.Iterator[torch.Tensor]:

        yield self.weight
        yield self.value


class FuzzyConverter(nn.Module):

    @abstractmethod
    def fuzzify(self, x: torch.Tensor) -> FuzzySet:
        pass

    @abstractmethod
    def imply(self, m: FuzzySet) -> ValueWeight:
        pass

    @abstractmethod
    def accumulate(self, value_weight: ValueWeight) -> torch.Tensor:
        pass

    def defuzzify(self, m: FuzzySet) -> torch.Tensor:
        return self.accumulate(self.imply(m))

    def forward(self, x: torch.Tensor) -> FuzzySet:
        return self.fuzzify(x)


class CrispConverter(nn.Module):

    @abstractmethod
    def crispify(self, x: torch.Tensor) -> CrispSet:
        pass

    @abstractmethod
    def imply(self, m: FuzzySet) -> ValueWeight:
        pass

    @abstractmethod
    def accumulate(self, value_weight: ValueWeight) -> torch.Tensor:
        pass

    def decrispify(self, m: CrispSet) -> torch.Tensor:
        return self.accumulate(self.imply(m))

    def forward(self, x: torch.Tensor) -> CrispSet:
        return self.crispify(x)


class Crispifier(nn.Module):

    @abstractmethod
    def forward(self, x: torch.Tensor) -> CrispSet:
        pass


class Fuzzifier(nn.Module):

    @abstractmethod
    def forward(self, x: torch.Tensor) -> FuzzySet:
        pass


class Defuzzifier(nn.Module):

    @abstractmethod
    def imply(self, m: FuzzySet) -> ValueWeight:
        pass

    @abstractmethod
    def accumulate(self, value_weight: ValueWeight) -> torch.Tensor:
        pass

    @abstractmethod
    def forward(self, m: FuzzySet) -> torch.Tensor:
        return self.accumulate(self.imply(m))


class Decrispifier(nn.Module):

    @abstractmethod
    def imply(self, m: FuzzySet) -> ValueWeight:
        pass

    @abstractmethod
    def accumulate(self, value_weight: ValueWeight) -> torch.Tensor:
        pass

    def fowrard(self, m: CrispSet) -> torch.Tensor:
        return self.accumulate(self.imply(m))


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

# TODO: Add ternary

class StepCrispConverter(CrispConverter):

    def __init__(self, out_variables: int, out_features: int, accumulator: Accumulator=None):
        super().__init__()
        self.weight = nn.parameter.Parameter(
            torch.randn(out_variables, out_features)
        )
        self.bias = nn.parameter.Parameter(
            torch.randn(out_variables, out_features)
        )
        self._accumulator = accumulator

    def crispify(self, x: torch.Tensor) -> CrispSet:
        return (x[:,:,None] >= self.weight[None]).type_as(x)

    def imply(self, m: CrispSet) -> ValueWeight:
        
        return ValueWeight(
            m, m * self.weight[None]
        )

    def accumulate(self, value_weight: ValueWeight) -> torch.Tensor:
        return self._accumulator.forward(value_weight)


class SigmoidFuzzyConverter(FuzzyConverter):

    def __init__(self, out_variables: int, out_features: int, eps: float=1e-7, accumulator: Accumulator=None):

        super().__init__()
        self.weight = nn.parameter.Parameter(
            torch.randn(out_variables, out_features)
        )
        self.bias = nn.parameter.Parameter(
            torch.randn(out_variables, out_features)
        )
        self.eps = eps
        self._accumulator = accumulator

    def fuzzify(self, x: torch.Tensor) -> FuzzySet:
        return nn_func.sigmoid(
            -(x[:,:,None] - self.bias[None]) * self.weight[None]
        )

    def imply(self, m: FuzzySet) -> ValueWeight:

        #     # x = ln(y/(1-y))
        return ValueWeight(m, (-torch.log(
            1 / (m.data + self.eps) - 1
        ) / self.weight[None] + self.bias[None]))

    def accumulate(self, value_weight: ValueWeight) -> torch.Tensor:
        return self._accumulator.forward(value_weight)


class SigmoidDefuzzifier(Defuzzifier):

    def __init__(self, converter: SigmoidFuzzyConverter):

        super().__init__()
        self._converter = converter

    def forward(self, m: FuzzySet):
        return self._converter.defuzzify(m)
    
    @classmethod
    def build(cls, out_variables: int, out_features: int, eps: float=1e-7):
        return SigmoidDefuzzifier(
            SigmoidFuzzyConverter(out_variables, out_features, eps)
        )


class ShapeConverter(FuzzyConverter):

    def __init__(self, left_edge: Shape, middle: Shape, right_edge: Shape):

        super().__init__([left_edge, middle, right_edge])
        self._left_edge = left_edge
        self._middle = middle
        self._right_edge = right_edge
    
    @property
    def left_edge(self) -> Shape:
        return self._left_edge

    @property
    def middle(self) -> Shape:
        return self._middle
    
    @property
    def right_edge(self) -> Shape:
        return self._right_edge

    def truncate(self, m: torch.Tensor) -> 'ShapeConverter':
        return self.__class__(
            self._left_edge.truncate(m[:,:,0:1]),
            self._middle.truncate(m[:,:,1:-1]),
            self._right_edge.truncate(m[:,:,-1:]),
        )

    def scale(self, m: torch.Tensor) -> 'ShapeConverter':
        return self.__class__(
            self._left_edge.scale(m[:,:,0:1]),
            self._middle.scale(m[:,:,1:-1]),
            self._right_edge.scale(m[:,:,-1:]),
        )

    def join(self, x: torch.Tensor) -> torch.Tensor:

        return torch.cat(
            [self._left_edge.join(x), self._middle.join(x), self._right_edge.join(x)],
            dim=2
        )

class ShapeAggregator(nn.Module):

    @abstractmethod
    def forward(self, *shapes: memb.Shape):
        pass


class AreaAggregator(nn.Module):

    def forward(self, *shapes: memb.Shape):
        return torch.cat(
            [shape.areas for shape in shapes], dim=2
        )


class MeanCoreAggregator(nn.Module):

    def forward(self, *shapes: memb.Shape):
        return torch.cat(
            [shape.mean_cores for shape in shapes], dim=2
        )


class CentroidAggregator(nn.Module):

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


class IsoscelesFuzzyConverter(FuzzyConverter):

    def __init__(self, n_variables: int, n_values: int, accumulator: Accumulator=None, flat_edges: bool=False, truncate: bool=True, fixed: bool=False):

        super().__init__()
        self._aggregator = AreaAggregator()

        self._n_variables = n_variables
        self._n_values = n_values
        
        self._step = 1
        if flat_edges:
            self._left_cls = memb.DecreasingRightTrapezoid
            self._right_cls = memb.IncreasingRightTrapezoid
            self._n_side_points = 3
            self._n_points = n_values + 2
            self._side_step = 1
        else:
            self._side_step = 1
            self._side_points = 2
            self._n_points = n_values
            self._left_cls = memb.DecreasingRightTriangle
            self._right_cls = memb.IncreasingRightTriangle

        self._middle_cls = memb.IsoscelesTriangle
        self._accumulator = accumulator
        self._truncate = truncate
        self._side_points = 3
        self._middle_points = n_values

        self._params = torch.linspace(0.0, 1.0, self._n_points)
        self._params = self._params[None]
        self._params = self._params.repeat(n_variables, 1)

        if not fixed:
            self._params = nn.parameter.Parameter(self._params)

    def generate_params(self):
        positive_params = torch.nn.functional.softplus(self._params)
        cumulated_params = torch.cumsum(positive_params, dim=1)
        min_val = cumulated_params[:,:1]
        max_val = cumulated_params[:,-1:]
        scaled_params  = (cumulated_params - min_val) / (max_val - min_val)
        return scaled_params

    def create_shapes(self) -> typing.Tuple[Shape, Shape, Shape]:
        left = self._params[:,:self._params].view(self._n_variables, 1, -1)
        right = self._params[:,-self._side_points:].view(self._n_values, 1, -1)
        middle = stride_coordinates(
            self._params[:,self._side_step:self._side_step + self._middle_points],
            self._n_points, self._step
        )
        return self._left_cls(left), self._middle_cls(middle), self._right_cls(right)

    def _join(self, x: torch.Tensor):
        return torch.cat(
            [shape.join(x) for shape in self.create_shapes() ],dim=2
        )
    
    def _aggregate(self, m: torch.Tensor):
        xs = []
        for shape in self.create_shapes():
            if self._truncate:
                xs.append(shape.truncate(m) )
            else:
                xs.append(shape.scale(m))
        return self._aggregator(*xs)

    def fuzzify(self, x: torch.Tensor) -> FuzzySet:
        return self._join(x)

    def accumulate(self, value_weight: ValueWeight) -> FuzzySet:
        return self._accumulator.forward(value_weight)

    def imply(self, m: FuzzySet) -> ValueWeight:
        return ValueWeight(m, self._aggregate(m))


# Not sure why i have strides
# def get_strided_indices(n_points: int, stride: int, step: int=1):
#     initial_indices = torch.arange(0, n_points).as_strided((n_points - stride + 1, stride), (1, 1))
#     return initial_indices[torch.arange(0, len(initial_indices), step)]


# def stride_coordinates(coordinates: torch.Tensor, stride: int, step: int=1):

#     dim2_index = get_strided_indices(coordinates.size(2), stride, step)
#     return coordinates[:, :, dim2_index]
