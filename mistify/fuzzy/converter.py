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
from ..base import (
    ValueWeight, Accumulator, MaxAcc, WeightedAverageAcc, ShapeImplication, Shape,
    ShapePoints, get_implication, stride_coordinates
)


class FuzzyConverter(nn.Module):
    """Convert tensor to fuzzy set
    """

    @abstractmethod
    def fuzzify(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def imply(self, m: torch.Tensor) -> ValueWeight:
        pass

    @abstractmethod
    def accumulate(self, value_weight: ValueWeight) -> torch.Tensor:
        pass

    def defuzzify(self, m: torch.Tensor) -> torch.Tensor:
        return self.accumulate(self.imply(m))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuzzify(x)

    def get_accumulator(self, accumulator: typing.Union['Accumulator', str]):

        if isinstance(accumulator, Accumulator):
            return accumulator
        if accumulator == 'max':
            return MaxAcc()
        if accumulator == 'weighted_average':
            return WeightedAverageAcc()
        raise ValueError(f"Name {accumulator} cannot be created")


class Fuzzifier(nn.Module):

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class Defuzzifier(nn.Module):

    @abstractmethod
    def imply(self, m: torch.Tensor) -> ValueWeight:
        pass

    @abstractmethod
    def accumulate(self, value_weight: ValueWeight) -> torch.Tensor:
        pass

    @abstractmethod
    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return self.accumulate(self.imply(m))


class SigmoidFuzzyConverter(FuzzyConverter):

    def __init__(self, out_variables: int, out_terms: int, eps: float=1e-7, accumulator: Accumulator=None):

        super().__init__()
        self.weight = nn.parameter.Parameter(
            torch.randn(out_variables, out_terms)
        )
        self.bias = nn.parameter.Parameter(
            torch.randn(out_variables, out_terms)
        )
        self.eps = eps
        self._accumulator = accumulator or MaxAcc()

    def fuzzify(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(
            -(x[:,:,None] - self.bias[None]) * self.weight[None]
        )

    def imply(self, m: torch.Tensor) -> ValueWeight:

        m = torch.clamp(m, self.eps, 1 - self.eps)
        return ValueWeight(
            torch.logit(m) / (self.weight[None]) + self.bias[None], 
            m
        )

    def accumulate(self, value_weight: ValueWeight) -> torch.Tensor:
        return self._accumulator.forward(value_weight)


class SigmoidDefuzzifier(Defuzzifier):

    def __init__(self, converter: SigmoidFuzzyConverter):

        super().__init__()
        self.converter = converter

    def forward(self, m: torch.Tensor):
        return self.converter.defuzzify(m)
    
    @classmethod
    def build(cls, out_variables: int, out_terms: int, eps: float=1e-7, accumulator: Accumulator=None):
        return SigmoidDefuzzifier(
            SigmoidFuzzyConverter(out_variables, out_terms, eps, accumulator)
        )


class RangeFuzzyConverter(FuzzyConverter):

    def __init__(self, out_variables: int, out_terms: int, accumulator: Accumulator=None):

        super().__init__()
        self.lower = nn.parameter.Parameter(
            torch.randn(out_variables, out_terms) * 0.01
        )
        self.dx = nn.parameter.Parameter(
            torch.randn(out_variables, out_terms) * 0.01
        )
        self._accumulator = accumulator or MaxAcc()

    def fuzzify(self, x: torch.Tensor) -> torch.Tensor:
        lower = self.lower[None]
        upper = torch.nn.functional.softplus(self.dx[None]) + lower
        m = (upper - x[:,:,None]) / (upper - lower)
        return torch.clamp(m, 0, 1)

    def imply(self, m: torch.Tensor) -> ValueWeight:
        lower = self.lower[None]
        upper = torch.nn.functional.softplus(self.dx[None]) + lower
        x = upper - m * (upper - lower)
        return ValueWeight(
            x, m
        )

    def accumulate(self, value_weight: ValueWeight) -> torch.Tensor:
        return self._accumulator.forward(value_weight)


class RangeDefuzzifier(Defuzzifier):

    def __init__(self, converter: RangeFuzzyConverter):

        super().__init__()
        self.converter = converter

    def forward(self, m: torch.Tensor):
        return self.converter.defuzzify(m)
    
    @classmethod
    def build(cls, out_variables: int, out_terms: int, accumulator: Accumulator=None):
        return RangeDefuzzifier(
            RangeFuzzyConverter(out_variables, out_terms, accumulator)
        )


class ConverterDefuzzifier(Defuzzifier):

    def __init__(self, converter: FuzzyConverter):
        super().__init__()
        self.converter = converter

    def imply(self, m: torch.Tensor) -> ValueWeight:
        return self.converter.imply(m)

    def accumulate(self, value_weight: ValueWeight) -> torch.Tensor:
        return self.converter.accumulate(value_weight)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return self.converter.defuzzify(m)


class PolygonFuzzyConverter(FuzzyConverter):
    """Convert 
    """

    def __init__(
        self, n_variables: int, n_terms: int, shape_pts: ShapePoints,
        left_cls: memb.Polygon, middle_cls: memb.Polygon, right_cls: memb.Polygon, 
        fixed: bool=False,
        implication: typing.Union[ShapeImplication, str]="area", 
        accumulator: typing.Union[Accumulator, str]="max", truncate: bool=True
    ):
        super().__init__()

        self._shape_pts = shape_pts
        self._left_cls = left_cls
        self._middle_cls = middle_cls
        self._right_cls = right_cls
        self.truncate = truncate
        self.accumulator = self.get_accumulator(accumulator)
        self.implication = get_implication(implication)
        self._n_variables = n_variables
        self._n_terms = n_terms
        params = torch.linspace(0.0, 1.0, shape_pts.n_pts)
        params = params.unsqueeze(0)
        params = params.repeat(n_variables, 1)
        if not fixed:
            self._params = nn.parameter.Parameter(params)
        else: 
            self._params = params
    
    # how can i make this more flexible (?)
    def generate_params(self):
        positive_params = torch.nn.functional.softplus(self._params)
        cumulated_params = torch.cumsum(positive_params, dim=1)
        min_val = cumulated_params[:,:1]
        max_val = cumulated_params[:,-1:]
        scaled_params  = (cumulated_params - min_val) / (max_val - min_val)
        return scaled_params
    
    def _join(self, x: torch.Tensor):
        return torch.cat(
            [shape.join(x) for shape, _ in self.create_shapes() ],dim=2
        )
    
    def create_shapes(self, m: torch.Tensor=None) -> typing.Iterator[typing.Tuple[Shape, torch.Tensor]]:
        left = memb.ShapeParams(
            self._params[:,:self._shape_pts.n_side_pts].view(self._n_variables, 1, -1))
        yield self._left_cls(left), m[:,:,:1] if m is not None else None

        if self._n_terms > 2:
            middle = memb.ShapeParams(
                stride_coordinates(
                    self._params[
                        :,self._shape_pts.side_step:self._shape_pts.side_step + self._shape_pts.n_middle_pts
                    ],
                    self._shape_pts.n_middle_shape_pts, self._shape_pts.step
            ))
            yield self._middle_cls(middle), m[:,:,:-2] if m is not None else None

        right = memb.ShapeParams(
            self._params[:,-self._shape_pts.n_side_pts:].view(self._n_variables, 1, -1)
        )
        yield self._right_cls(right), m[:,:,-1:] if m is not None else None
    
    def _imply(self, m: torch.Tensor) -> torch.Tensor:
        xs = []
        for shape, m_i in self.create_shapes(m):
            if self.truncate:
                xs.append(shape.truncate(m_i) )
            else:
                xs.append(shape.scale(m_i))
        
        return self.implication(*xs)

    def fuzzify(self, x: torch.Tensor) -> torch.Tensor:
        return self._join(x)

    def accumulate(self, value_weight: ValueWeight) -> torch.Tensor:
        return self.accumulator.forward(value_weight)

    def imply(self, m: torch.Tensor) -> ValueWeight:
        return ValueWeight(self._imply(m), m)


class IsoscelesFuzzyConverter(PolygonFuzzyConverter):

    def __init__(
        self, n_variables: int, n_terms: int, 
        implication: typing.Union[ShapeImplication, str]="area", 
        accumulator: typing.Union[Accumulator, str]="max", flat_edges: bool=False, truncate: bool=True, fixed: bool=False
    ):
        if flat_edges:
            left_cls = memb.DecreasingRightTrapezoid
            right_cls = memb.IncreasingRightTrapezoid
            shape_pts = ShapePoints(3, n_terms + 2, n_terms - 1, 2, 1, 1)
        else:
            shape_pts = ShapePoints(2, n_terms, n_terms - 1, 2, 0, 1)
            left_cls = memb.DecreasingRightTriangle
            right_cls = memb.IncreasingRightTriangle

        middle_cls = memb.IsoscelesTriangle
        
        super().__init__(
            n_variables, n_terms, shape_pts, left_cls, middle_cls, right_cls, fixed, implication, accumulator, truncate
        )


class TriangleFuzzyConverter(PolygonFuzzyConverter):

    def __init__(
        self, n_variables: int, n_terms: int, 
        implication: typing.Union[ShapeImplication, str]="area", 
        accumulator: typing.Union[Accumulator, str]="max", 
        flat_edges: bool=False, truncate: bool=True, fixed: bool=False
    ):

        if flat_edges:
            left_cls = memb.DecreasingRightTrapezoid
            right_cls = memb.IncreasingRightTrapezoid
            shape_pts = ShapePoints(3, n_terms + 2, n_terms, 3, 1, 1)
        else:
            shape_pts = ShapePoints(2, n_terms, n_terms, 3, 0, 1)
            left_cls = memb.DecreasingRightTriangle
            right_cls = memb.IncreasingRightTriangle
        middle_cls = memb.Triangle
        
        super().__init__(
            n_variables, n_terms, shape_pts, left_cls, middle_cls, right_cls, fixed, implication, accumulator, truncate
        )


class TrapezoidFuzzyConverter(PolygonFuzzyConverter):

    def __init__(
        self, n_variables: int, n_terms: int, 
        implication: typing.Union[ShapeImplication, str]="area", 
        accumulator: typing.Union[Accumulator, str]="max", 
        flat_edges: bool=False, truncate: bool=True, fixed: bool=False
    ):
        if flat_edges:
            left_cls = memb.DecreasingRightTrapezoid
            right_cls = memb.IncreasingRightTrapezoid

            shape_pts = ShapePoints(3, n_terms + 4, n_terms + 2, 4, 0, 2)
        else:
            shape_pts = ShapePoints(3, n_terms + 2, n_terms + 2, 4, 1, 2)
            left_cls = memb.DecreasingRightTriangle
            right_cls = memb.IncreasingRightTriangle

        middle_cls = memb.Trapezoid
        
        super().__init__(
            n_variables, n_terms, shape_pts,
            left_cls, middle_cls, right_cls, fixed, 
            implication, accumulator, truncate, 
        )


class LogisticFuzzyConverter(FuzzyConverter):

    def __init__(
        self, n_variables: int, n_terms: int, fixed: bool=False,
        implication: typing.Union[ShapeImplication, str]="area", 
        accumulator: typing.Union[Accumulator, str]="max", truncate: bool=True
    ):
        super().__init__()
        assert n_terms > 1

        self.truncate = truncate
        self.accumulator = self.get_accumulator(accumulator)
        self.implication = get_implication(implication)
        self._n_variables = n_variables
        self._n_terms = n_terms
        biases = torch.linspace(0.0, 1.0, n_terms).unsqueeze(0)
        self._scales = torch.empty(
            n_variables, n_terms).fill_(1.0 / (n_terms - 1)
        ).unsqueeze(2)
        self._biases = biases.repeat(
            n_variables, 1).unsqueeze(2)
        if not fixed:
            self._biases = nn.parameter.Parameter(self._biases)
            self._scales = nn.parameter.Parameter(self._scales)

    def generate_means(self):
        positive_params = torch.nn.functional.softplus(self._biases)
        cumulated_params = torch.cumsum(positive_params, dim=1)
        min_val = cumulated_params[:,:1]
        max_val = cumulated_params[:,-1:]
        scaled_params  = (cumulated_params - min_val) / (max_val - min_val)
        return scaled_params
    
    def _join(self, x: torch.Tensor):
        return torch.cat(
            [shape.join(x) for shape, _ in self.create_shapes() ],dim=2
        )
    
    def create_shapes(self, m: torch.Tensor=None) -> typing.Iterator[typing.Tuple[Shape, torch.Tensor]]:
        left_biases = memb.ShapeParams(
            self._biases[:,:1].view(self._n_variables, 1, 1))
        left_scales = memb.ShapeParams(
            self._scales[:,:1].view(self._n_variables, 1, 1))
        
        yield memb.RightLogistic(left_biases, left_scales, False), m[:,:,:1] if m is not None else None
        
        if self._n_terms > 2:
            mid_biases = memb.ShapeParams(
                self._biases[:,1:-1].view(self._n_variables, self._n_terms - 2, 1))
            mid_scales = memb.ShapeParams(
                self._scales[:,1:-1].view(self._n_variables, self._n_terms - 2, 1))
            
            yield memb.LogisticBell(mid_biases, mid_scales), m[:,:,1:-1] if m is not None else None
            
        right_biases = memb.ShapeParams(
            self._biases[:,-1:].view(self._n_variables, 1, 1))
        right_scales = memb.ShapeParams(
            self._scales[:,-1:].view(self._n_variables, 1, 1))
        
        yield memb.RightLogistic(right_biases, right_scales, True), m[:,:,-1:] if m is not None else None
    
    def _imply(self, m: torch.Tensor) -> torch.Tensor:
        xs = []
        for shape, m_i in self.create_shapes(m):
            if self.truncate:
                xs.append(shape.truncate(m_i) )
            else:
                xs.append(shape.scale(m_i))
        return self.implication(*xs)

    def fuzzify(self, x: torch.Tensor) -> torch.Tensor:
        return self._join(x)

    def accumulate(self, value_weight: ValueWeight) -> torch.Tensor:
        return self.accumulator.forward(value_weight)

    def imply(self, m: torch.Tensor) -> ValueWeight:
        return ValueWeight(self._imply(m), m)


def fuzzy_to_binary(fuzzy: torch.Tensor, threshold: float=0.5):

    return (fuzzy > threshold).type_as(fuzzy)

