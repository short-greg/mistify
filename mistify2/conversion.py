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

    def __init__(
        self, out_variables: int, out_features: int, 
        accumulator: Accumulator=None, same: bool=False
    ):
        super().__init__()

        if same:
            self.weight = nn.parameter.Parameter(
                torch.randn(out_features)
            )
            self.bias = nn.parameter.Parameter(
                torch.randn(out_features)
            )
        else:
            self.weight = nn.parameter.Parameter(
                torch.randn(out_variables, out_features)
            )
            self.bias = nn.parameter.Parameter(
                torch.randn(out_variables, out_features)
            )

        self._accumulator = accumulator
        self.smae = same

    def crispify(self, x: torch.Tensor) -> CrispSet:
        weight = weight[None, None] if self.same else weight[None]
        return (x[:,:,None] >= weight).type_as(x)

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

    def truncate(self, m: FuzzySet) -> 'ShapeConverter':
        return self.__class__(
            self._left_edge.truncate(m[:,:,0:1]),
            self._middle.truncate(m[:,:,1:-1]),
            self._right_edge.truncate(m[:,:,-1:]),
        )

    def scale(self, m: FuzzySet) -> 'ShapeConverter':
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


@dataclass
class ShapePoints:

    n_side_pts: int
    n_pts: int
    n_middle_pts: int
    side_step: int
    step: int
    n_pts: int


class PolygonFuzzyConverter(FuzzyConverter):

    def __init__(
        self, n_variables: int, n_terms: int, shape_pts: ShapePoints,
        left_cls: memb.Polygon, middle_cls: memb.Polygon, right_cls: memb.Polygon, 
        fixed: bool=False,
        implication: ShapeImplication=None, accumulator: Accumulator=None, truncate: bool=True
    ):
        super().__init__()

        self._shape_pts = shape_pts
        self._left_cls = left_cls
        self._middle_cls = middle_cls
        self._right_cls = right_cls
        self.truncate = truncate
        self.accumulator = accumulator or WeightedAverageAcc()
        self.implication = implication or AreaImplication()
        self._n_variables = n_variables
        self._n_terms = n_terms
        params = torch.linspace(0.0, 1.0, shape_pts.n_pts)
        params = params[None]
        self._params = params.repeat(n_variables, 1)
        if not fixed:
            self._params = nn.parameter.Parameter(self._parameters)
    
    def generate_params(self):
        positive_params = torch.nn.functional.softplus(self._params)
        cumulated_params = torch.cumsum(positive_params, dim=1)
        min_val = cumulated_params[:,:1]
        max_val = cumulated_params[:,-1:]
        scaled_params  = (cumulated_params - min_val) / (max_val - min_val)
        return scaled_params
    
    def _join(self, x: torch.Tensor):
        return torch.cat(
            [shape.join(x) for shape in self.create_shapes() ],dim=2
        )
    
    def create_shapes(self) -> typing.Tuple[Shape, Shape, Shape]:
        left = self._params[:,:self._params].view(self._n_variables, 1, -1)
        right = self._params[:,-self._side_points:].view(self._n_values, 1, -1)
        middle = stride_coordinates(
            self._params[:,self._side_step:self._side_step + self._middle_points],
            self._n_points, self._step
        )
        return self._left_cls(left), self._middle_cls(middle), self._right_cls(right)
    
    def _imply(self, m: torch.Tensor):
        xs = []
        for shape in self.create_shapes():
            if self._truncate:
                xs.append(shape.truncate(m) )
            else:
                xs.append(shape.scale(m))
        return self._implication(*xs)

    def fuzzify(self, x: torch.Tensor) -> FuzzySet:
        return self._join(x)

    def accumulate(self, value_weight: ValueWeight) -> FuzzySet:
        return self._accumulator.forward(value_weight)

    def imply(self, m: FuzzySet) -> ValueWeight:
        return ValueWeight(m, self._imply(m))


class IsoscelesFuzzyConverter(PolygonFuzzyConverter):

    def __init__(
        self, n_variables: int, n_terms: int, implication: ShapeImplication=None, 
        accumulator: Accumulator=None, flat_edges: bool=False, truncate: bool=True, fixed: bool=False
    ):
        if flat_edges:
            left_cls = memb.DecreasingRightTrapezoid
            right_cls = memb.IncreasingRightTrapezoid
            shape_pts = ShapePoints(3, n_terms + 2, n_terms, 1, 1)
        else:
            shape_pts = ShapePoints(2, n_terms, n_terms, 0, 0)
            left_cls = memb.DecreasingRightTriangle
            right_cls = memb.IncreasingRightTriangle

        middle_cls = memb.IsoscelesTriangle
        
        super().__init__(
            n_variables, n_terms, shape_pts, left_cls, middle_cls, right_cls, fixed, implication, accumulator, truncate
        )


class TriangleFuzzyConverter(FuzzyConverter):

    def __init__(
        self, n_variables: int, n_terms: int, implication: ShapeImplication=None, accumulator: Accumulator=None, 
        flat_edges: bool=False, truncate: bool=True, fixed: bool=False
    ):

        if flat_edges:
            left_cls = memb.DecreasingRightTrapezoid
            right_cls = memb.IncreasingRightTrapezoid
            shape_pts = ShapePoints(3, n_terms + 2, n_terms, 1, 1)
        else:
            shape_pts = ShapePoints(3, n_terms + 2, n_terms, 1, 1)
            left_cls = memb.DecreasingRightTriangle
            right_cls = memb.IncreasingRightTriangle
        middle_cls = memb.Triangle
        
        super().__init__(
            n_variables, n_terms, shape_pts, left_cls, middle_cls, right_cls, fixed, implication, accumulator, truncate
        )


class TrapezoidFuzzyConverter(FuzzyConverter):

    def __init__(
        self, n_variables: int, n_terms: int, implication: ShapeImplication=None, accumulator: Accumulator=None, 
        flat_edges: bool=False, truncate: bool=True, fixed: bool=False
    ):
        if flat_edges:
            left_cls = memb.DecreasingRightTrapezoid
            right_cls = memb.IncreasingRightTrapezoid

            shape_pts = ShapePoints(3, n_terms + 2, n_terms, 0, 2)
        else:
            shape_pts = ShapePoints(3, n_terms + 2, n_terms, 1, 2)
            left_cls = memb.DecreasingRightTriangle
            right_cls = memb.IncreasingRightTriangle

        middle_cls = memb.Trapezoid
        
        super().__init__(
            n_variables, n_terms, shape_pts,
            left_cls, middle_cls, right_cls, fixed, 
            implication, accumulator, truncate, 
        )


class LogisticFuzzyConverter(FuzzyConverter):
    pass


# Not sure why i have strides
# def get_strided_indices(n_points: int, stride: int, step: int=1):
#     initial_indices = torch.arange(0, n_points).as_strided((n_points - stride + 1, stride), (1, 1))
#     return initial_indices[torch.arange(0, len(initial_indices), step)]


# def stride_coordinates(coordinates: torch.Tensor, stride: int, step: int=1):

#     dim2_index = get_strided_indices(coordinates.size(2), stride, step)
#     return coordinates[:, :, dim2_index]
