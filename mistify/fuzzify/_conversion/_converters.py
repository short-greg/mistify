
# 1st party
from abc import abstractmethod, ABC
import typing


# 3rd party
import torch
import torch.nn as nn
import torch.nn.functional

# local
from .._shapes import Shape
from .. import _shapes as shape
from ._conclude import Conclusion, ValueWeight, MaxAcc, WeightedAverageAcc
from ._hypo import ShapeHypothesis, HypothesisEnum
from ._utils import stride_coordinates
from .._shapes import Shape, ShapeParams, CompositeShape
from ._conclude import ValueWeight, Conclusion, MaxValueAcc, AccEnum
from ... import functional


def generate_spaced_params(n_steps: int, lower: float=0, upper: float=1) -> torch.Tensor:
    return torch.linspace(lower, upper, n_steps)[None, None, :]


def generate_repeat_params(n_steps: int, value: float) -> torch.Tensor:
    return torch.full((1, 1, n_steps), value)


class FuzzyConverter(nn.Module):
    """Convert tensor to fuzzy set
    """

    @abstractmethod
    def fuzzify(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def hypo(self, m: torch.Tensor) -> ValueWeight:
        pass

    @abstractmethod
    def accumulate(self, value_weight: ValueWeight) -> torch.Tensor:
        pass

    def defuzzify(self, m: torch.Tensor) -> torch.Tensor:
        return self.accumulate(self.hypo(m))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuzzify(x)
    
    def reverse(self, m: torch.Tensor) -> torch.Tensor:
        return self.defuzzify(m)

    def get_conclusion(self, conclusion: typing.Union['Conclusion', str]):

        if isinstance(conclusion, Conclusion):
            return conclusion
        if conclusion == 'max':
            return MaxAcc()
        if conclusion == 'weighted_average':
            return WeightedAverageAcc()
        raise ValueError(f"Name {conclusion} cannot be created")


class CompositeFuzzyConverter(FuzzyConverter):

    def __init__(
        self, shapes: typing.List[Shape], 
        hypothesis: typing.Union[ShapeHypothesis, str]="area", 
        conclusion: typing.Union[Conclusion, str]="max",
        truncate: bool=False
    ):
        super().__init__()

        self._composite = CompositeShape(shapes)
        self._hypothesis = HypothesisEnum.get(hypothesis)
        self._conclusion = AccEnum.get(conclusion)
        self._truncate = truncate

    def fuzzify(self, x: torch.Tensor) -> torch.Tensor:
        return self._composite.join(x)

    def accumulate(self, value_weight: ValueWeight) -> torch.Tensor:
        return self._conclusion.forward(value_weight)

    def hypo(self, m: torch.Tensor) -> ValueWeight:
        if self._truncate:
            shapes = self._composite.truncate(m).shapes
        else:
            shapes = self._composite.scale(m).shapes
        return ValueWeight(self._hypothesis(shapes), m)


def polygon(left: shape.Shape, middle: shape.Shape, right: shape.Shape):

    if middle is None:
        return [left, right]
    return [left, right, middle]


class IsoscelesFuzzyConverter(CompositeFuzzyConverter):

    def __init__(
        self, left: typing.Union[shape.DecreasingRightTrapezoid, shape.DecreasingRightTriangle],
        right: typing.Union[shape.IncreasingRightTrapezoid, shape.IncreasingRightTriangle],
        middle: shape.IsoscelesTriangle=None, 
        hypothesis: typing.Union[ShapeHypothesis, str]="area", 
        conclusion: typing.Union[Conclusion, str]="max", 
        truncate: bool=False
    ):
        super().__init__(
            polygon(left, middle, right), hypothesis, conclusion, truncate
        )

    @classmethod
    def from_coords(
        cls, coords: torch.Tensor, n_terms: int,
        hypothesis: typing.Union[ShapeHypothesis, str]="area", 
        conclusion: typing.Union[Conclusion, str]="max", 
        flat_edges: bool=False, 
        truncate: bool=True
    ):
        middle = None
        if flat_edges:
            left = shape.DecreasingRightTrapezoid(ShapeParams(coords[:,:,None,:3]))
            if n_terms > 2:
                middle = shape.IsoscelesTriangle(ShapeParams(stride_coordinates(coords[:,:,1:-1], n_terms - 2, 1, 2)))
            right = shape.IncreasingRightTrapezoid(ShapeParams(coords[:,:,None,-3:]))

        else:
            left = shape.DecreasingRightTriangle(ShapeParams(coords[:,:,None,:2]))
            if n_terms > 2:
                middle = shape.IsoscelesTriangle(ShapeParams(stride_coordinates(coords, n_terms - 2, 1, 2)))
            right = shape.IncreasingRightTriangle(ShapeParams(coords[:,:,None,-2:]))
        return IsoscelesFuzzyConverter(left, right, middle, hypothesis, conclusion, truncate )

    @classmethod
    def from_linspace(
        cls, n_terms: int, hypothesis: typing.Union[ShapeHypothesis, str]="area", 
        conclusion: typing.Union[Conclusion, str]="max", 
        flat_edges: bool=False, truncate: bool=True,
    ):
        if flat_edges:
            coords = generate_spaced_params(n_terms + 2)
        else:
            coords = generate_spaced_params(n_terms)
        return IsoscelesFuzzyConverter.from_coords(
            coords, n_terms, hypothesis, conclusion,
            flat_edges, truncate
        )


class IsoscelesTrapezoidFuzzyConverter(CompositeFuzzyConverter):

    def __init__(
        self, left: typing.Union[shape.DecreasingRightTrapezoid, shape.DecreasingRightTriangle],
        right: typing.Union[shape.IncreasingRightTrapezoid, shape.IncreasingRightTriangle],
        middle: shape.IsoscelesTrapezoid=None, 
        hypothesis: typing.Union[ShapeHypothesis, str]="area", 
        conclusion: typing.Union[Conclusion, str]="max", 
        truncate: bool=False
    ):
        super().__init__(
            polygon(left, middle, right), hypothesis, conclusion, truncate
        )

    @classmethod
    def from_coords(
        cls, coords: torch.Tensor, n_terms: int,
        hypothesis: typing.Union[ShapeHypothesis, str]="area", 
        conclusion: typing.Union[Conclusion, str]="max", 
        flat_edges: bool=False, 
        truncate: bool=True
    ):
        middle = None
        if flat_edges:
            left = shape.DecreasingRightTrapezoid(ShapeParams(coords[:,:,None,:3]))
            if n_terms > 2:
                middle = shape.IsoscelesTrapezoid(ShapeParams(stride_coordinates(coords[:,:,1:-1], n_terms - 2, 2, 3)))
            right = shape.IncreasingRightTrapezoid(ShapeParams(coords[:,:,None,-3:]))
        else:
            left = shape.DecreasingRightTriangle(ShapeParams(coords[:,:,None,:2]))
            if n_terms > 2:
                middle = shape.IsoscelesTrapezoid(ShapeParams(stride_coordinates(coords, n_terms - 2, 2, 3)))
            right = shape.IncreasingRightTriangle(ShapeParams(coords[:,:,None,-2:]))

        return IsoscelesTrapezoidFuzzyConverter(left, right, middle, hypothesis, conclusion, truncate )

    @classmethod
    def from_linspace(
        cls, n_terms: int, hypothesis: typing.Union[ShapeHypothesis, str]="area", 
        conclusion: typing.Union[Conclusion, str]="max", 
        flat_edges: bool=False, truncate: bool=True,
    ):
        if flat_edges:
            coords = generate_spaced_params((n_terms - 2) * 2 + 4)
        else:
            coords = generate_spaced_params((n_terms - 2) * 2 + 2)
        return IsoscelesTrapezoidFuzzyConverter.from_coords(
            coords, n_terms, hypothesis, conclusion,
            flat_edges, truncate
        )


class TrapezoidFuzzyConverter(CompositeFuzzyConverter):

    def __init__(
        self, left: typing.Union[shape.DecreasingRightTrapezoid, shape.DecreasingRightTriangle],
        right: typing.Union[shape.IncreasingRightTrapezoid, shape.IncreasingRightTriangle],
        middle: shape.Trapezoid=None, 
        hypothesis: typing.Union[ShapeHypothesis, str]="area", 
        conclusion: typing.Union[Conclusion, str]="max", 
        truncate: bool=False
    ):
        super().__init__(
            polygon(left, middle, right), hypothesis, conclusion, truncate
        )

    @classmethod
    def from_coords(
        cls, coords: torch.Tensor, n_terms: int,
        hypothesis: typing.Union[ShapeHypothesis, str]="area", 
        conclusion: typing.Union[Conclusion, str]="max", 
        flat_edges: bool=False, 
        truncate: bool=True
    ):
        middle = None
        if flat_edges:
            left = shape.DecreasingRightTrapezoid(ShapeParams(coords[:,:,None,:3]))
            if n_terms > 2:
                middle = shape.Trapezoid(ShapeParams(stride_coordinates(coords[:,:,1:-1], n_terms - 2, 2, 4)))
            right = shape.IncreasingRightTrapezoid(ShapeParams(coords[:,:,None,-3:]))
        else:
            left = shape.DecreasingRightTriangle(ShapeParams(coords[:,:,None,:2]))
            if n_terms > 2:
                middle = shape.Trapezoid(ShapeParams(stride_coordinates(coords, n_terms - 2, 2, 4)))
            right = shape.IncreasingRightTriangle(ShapeParams(coords[:,:,None,-2:]))

        return TrapezoidFuzzyConverter(left, right, middle, hypothesis, conclusion, truncate )

    @classmethod
    def from_linspace(
        cls, n_terms: int, hypothesis: typing.Union[ShapeHypothesis, str]="area", 
        conclusion: typing.Union[Conclusion, str]="max", 
        flat_edges: bool=False, truncate: bool=True,
    ):
        if flat_edges:
            coords = generate_spaced_params((n_terms - 2) * 2 + 4)
        else:
            coords = generate_spaced_params((n_terms - 2) * 2 + 2)
        return TrapezoidFuzzyConverter.from_coords(
            coords, n_terms, hypothesis, conclusion,
            flat_edges, truncate
        )


class TriangleFuzzyConverter(CompositeFuzzyConverter):

    def __init__(
        self, left: typing.Union[shape.DecreasingRightTrapezoid, shape.DecreasingRightTriangle],
        right: typing.Union[shape.IncreasingRightTrapezoid, shape.IncreasingRightTriangle],
        middle: shape.IsoscelesTriangle=None, 
        hypothesis: typing.Union[ShapeHypothesis, str]="area", 
        conclusion: typing.Union[Conclusion, str]="max", 
        truncate: bool=False
    ):
        super().__init__(
            polygon(left, middle, right), hypothesis, conclusion, truncate
        )

    @classmethod
    def from_coords(
        cls, coords: torch.Tensor, n_terms: int,
        hypothesis: typing.Union[ShapeHypothesis, str]="area", 
        conclusion: typing.Union[Conclusion, str]="max", 
        flat_edges: bool=False, 
        truncate: bool=True
    ):
        middle = None
        if flat_edges:
            left = shape.DecreasingRightTrapezoid(ShapeParams(coords[:,:,None,:3]))
            if n_terms > 2:
                middle = shape.Triangle(ShapeParams(stride_coordinates(coords[:,:,1:-1], n_terms - 2, 1, 3)))
            right = shape.IncreasingRightTrapezoid(ShapeParams(coords[:,:,None,-3:]))

        else:
            left = shape.DecreasingRightTriangle(ShapeParams(coords[:,:,None,:2]))
            if n_terms > 2:
                middle = shape.Triangle(ShapeParams(stride_coordinates(coords, n_terms - 2, 1, 3)))
            right = shape.IncreasingRightTriangle(ShapeParams(coords[:,:,None,-2:]))
        return TriangleFuzzyConverter(left, right, middle, hypothesis, conclusion, truncate )

    @classmethod
    def from_linspace(
        cls, n_terms: int, hypothesis: typing.Union[ShapeHypothesis, str]="area", 
        conclusion: typing.Union[Conclusion, str]="max", 
        flat_edges: bool=False, truncate: bool=True,
    ):
        if flat_edges:
            coords = generate_spaced_params(n_terms + 2)
        else:
            coords = generate_spaced_params(n_terms)
        return TriangleFuzzyConverter.from_coords(
            coords, n_terms, hypothesis, conclusion,
            flat_edges, truncate
        )


class SquareFuzzyConverter(CompositeFuzzyConverter):

    def __init__(
        self, 
        square: shape.Square, 
        hypothesis: typing.Union[ShapeHypothesis, str]="area", 
        conclusion: typing.Union[Conclusion, str]="max", 
        truncate: bool=False
    ):
        super().__init__(
            square, hypothesis, conclusion, truncate
        )

    @classmethod
    def from_coords(
        cls, coords: torch.Tensor, n_terms: int,
        hypothesis: typing.Union[ShapeHypothesis, str]="area", 
        conclusion: typing.Union[Conclusion, str]="max", 
        truncate: bool=True
    ):
        square = shape.Square(ShapeParams(stride_coordinates(coords[:,:,:], n_terms, 2, 2)))
        return SquareFuzzyConverter(square, hypothesis, conclusion, truncate )

    @classmethod
    def from_linspace(
        cls, n_terms: int, hypothesis: typing.Union[ShapeHypothesis, str]="area", 
        conclusion: typing.Union[Conclusion, str]="max", 
        truncate: bool=True,
    ):
        coords = generate_spaced_params(n_terms + 1)
        return SquareFuzzyConverter.from_coords(
            coords, n_terms, hypothesis, conclusion,
            truncate
        )


class LogisticFuzzyConverter(CompositeFuzzyConverter):

    def __init__(
        self, left: shape.RightLogistic,
        right: shape.RightLogistic,
        middle: shape.LogisticBell=None, 
        hypothesis: typing.Union[ShapeHypothesis, str]="area", 
        conclusion: typing.Union[Conclusion, str]="max", 
        truncate: bool=False
    ):
        super().__init__(
            polygon(left, middle, right), hypothesis, conclusion, truncate
        )

    @classmethod
    def from_coords(
        cls, bias_coords: torch.Tensor, scale_coords: torch.Tensor, n_terms: int,
        hypothesis: typing.Union[ShapeHypothesis, str]="area", 
        conclusion: typing.Union[Conclusion, str]="max", 
        truncate: bool=True
    ):
        middle = None
        left = shape.RightLogistic(
            ShapeParams(bias_coords[:,:,None,0:1]), 
            ShapeParams(scale_coords[:,:,None,0:1]), 
            False
        )
        if n_terms > 2:
            middle = shape.LogisticBell(
                ShapeParams(bias_coords[:,:,1:-1,None]), ShapeParams(scale_coords[:,:,1:-1,None])
            )
        right = shape.RightLogistic(ShapeParams(bias_coords[:,:,-1:,None]), ShapeParams(scale_coords[:,:,-1:,None]))

        return LogisticFuzzyConverter(left, right, middle, hypothesis, conclusion, truncate)

    @classmethod
    def from_linspace(
        cls, n_terms: int, hypothesis: typing.Union[ShapeHypothesis, str]="area", 
        conclusion: typing.Union[Conclusion, str]="max", 
        truncate: bool=True,
    ):
        bias_coords = generate_spaced_params(n_terms)
        width = 1.0 / 2 * (n_terms - 1.0)
        scale_coords = generate_repeat_params(n_terms, width)
        return LogisticFuzzyConverter.from_coords(
            bias_coords, scale_coords, n_terms, hypothesis, conclusion,
            truncate
        )


class SigmoidFuzzyConverter(CompositeFuzzyConverter):

    def __init__(
        self, sigmoid: shape.Sigmoid=None, 
        hypothesis: typing.Union[ShapeHypothesis, str]="area", 
        conclusion: typing.Union[Conclusion, str]="max", 
        truncate: bool=False
    ):
        super().__init__(
            [sigmoid], hypothesis, conclusion, truncate
        )

    @classmethod
    def from_coords(
        cls, bias_coords: torch.Tensor, scale_coords: torch.Tensor, n_terms: int,
        hypothesis: typing.Union[ShapeHypothesis, str]="area", 
        conclusion: typing.Union[Conclusion, str]="max", 
        truncate: bool=True
    ):
        sigmoid = shape.Sigmoid(
            ShapeParams(bias_coords[:,:,:,None]), ShapeParams(scale_coords[:,:,:,None])
        )
        return SigmoidFuzzyConverter(sigmoid, hypothesis, conclusion, truncate)

    @classmethod
    def from_linspace(
        cls, n_terms: int, hypothesis: typing.Union[ShapeHypothesis, str]="area", 
        conclusion: typing.Union[Conclusion, str]="max", 
        truncate: bool=True,
    ):
        bias_coords = generate_spaced_params(n_terms + 2)[:,:,1:-1]
        width = 1.0 / 2 * (n_terms - 1.0)
        scale_coords = generate_repeat_params(n_terms, width)
        return SigmoidFuzzyConverter.from_coords(
            bias_coords, scale_coords, n_terms, hypothesis, conclusion,
            truncate
        )


class RampFuzzyConverter(CompositeFuzzyConverter):

    def __init__(
        self, ramp: shape.Ramp=None, 
        hypothesis: typing.Union[ShapeHypothesis, str]="area", 
        conclusion: typing.Union[Conclusion, str]="max", 
        truncate: bool=False
    ):
        super().__init__(
            [ramp], hypothesis, conclusion, truncate
        )

    @classmethod
    def from_coords(
        cls, coords: torch.Tensor, n_terms: int,
        hypothesis: typing.Union[ShapeHypothesis, str]="area", 
        conclusion: typing.Union[Conclusion, str]="max", 
        truncate: bool=True
    ):
        
        ramp = shape.Ramp(
            ShapeParams(stride_coordinates(coords, n_terms, 1, 2, 2))
        )
        return RampFuzzyConverter(ramp, hypothesis, conclusion, truncate)

    @classmethod
    def from_linspace(
        cls, n_terms: int, hypothesis: typing.Union[ShapeHypothesis, str]="area", 
        conclusion: typing.Union[Conclusion, str]="max", 
        truncate: bool=True,
    ):
        coords = generate_spaced_params(n_terms + 2)
        return RampFuzzyConverter.from_coords(
            coords, n_terms, hypothesis, conclusion,
            truncate
        )


class StepFuzzyConverter(CompositeFuzzyConverter):

    def __init__(
        self, step: shape.Step=None, 
        hypothesis: typing.Union[ShapeHypothesis, str]="min_core", 
        conclusion: typing.Union[Conclusion, str]="max", 
        truncate: bool=False
    ):
        super().__init__(
            [step], hypothesis, conclusion, truncate
        )

    @classmethod
    def from_coords(
        cls, coords: torch.Tensor, n_terms: int,
        hypothesis: typing.Union[ShapeHypothesis, str]="min_core", 
        conclusion: typing.Union[Conclusion, str]="max", 
        truncate: bool=True
    ):
        step = shape.Step(
            ShapeParams(stride_coordinates(coords, n_terms, 1, 1, 1))
        )
        return StepFuzzyConverter(step, hypothesis, conclusion, truncate)

    @classmethod
    def from_linspace(
        cls, n_terms: int, hypothesis: typing.Union[ShapeHypothesis, str]="min_core", 
        conclusion: typing.Union[Conclusion, str]="max", 
        truncate: bool=True,
    ):
        coords = generate_spaced_params(n_terms + 2)[:,:,1:-1]
        return StepFuzzyConverter.from_coords(
            coords, n_terms, hypothesis, conclusion,
            truncate
        )


class ConverterDecorator(ABC, FuzzyConverter):

    def __init__(self, converter: FuzzyConverter):

        super().__init__()
        self._converter = converter

    @abstractmethod
    def decorate_fuzzify(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def decorate_defuzzify(self, m: torch.Tensor) -> torch.Tensor:
        pass

    def fuzzify(self, x: torch.Tensor) -> torch.Tensor:
        return self._converter.fuzzify(self.decorate_fuzzify(x))

    def accumulate(self, value_weight: ValueWeight) -> torch.Tensor:
        return self._converter.accumulate(value_weight)
    
    def hypo(self, m: torch.Tensor) -> ValueWeight:
        return self.decorate_defuzzify(self._converter.hypo(m))
    

class FuncConverterDecorator(ConverterDecorator):

    def __init__(self, converter: FuzzyConverter, fuzzify: typing.Callable[[torch.Tensor], torch.Tensor], defuzzify: typing.Callable[[torch.Tensor], torch.Tensor]):

        super().__init__(converter)
        self._fuzzify = fuzzify
        self._defuzzify = defuzzify

    def decorate_fuzzify(self, x: torch.Tensor) -> torch.Tensor:
        return self._fuzzify(x)

    def decorate_defuzzify(self, m: torch.Tensor) -> torch.Tensor:
        return self._defuzzify(m)



# class SignedConverter(FuzzyConverter):

#     def __init__(self, base_converter: FuzzyConverter):

#         super().__init__()
#         self._converter = base_converter

#     def crispify(self, x: torch.Tensor) -> torch.Tensor:
#         return functional.to_signed(super().fuzzify(x))

#     def accumulate(self, value_weight: ValueWeight) -> torch.Tensor:
#         return self._converter.accumulate(value_weight)
    
#     def hypo(self, m: torch.Tensor) -> ValueWeight:
#         m = functional.to_binary(m)
#         return self._converter.hypo(m)


# class StepFuzzyConverter(FuzzyConverter):

#     def __init__(
#         self, out_variables: int, out_terms: int, 
#         conclusion: Conclusion=None,
#         threshold_f: typing.Callable[[torch.Tensor, typing.Any], torch.Tensor]=None
#     ):
#         super().__init__()

#         self.threshold = nn.parameter.Parameter(
#             torch.randn(out_variables, out_terms)
#         )
#         self._threshold_f = threshold_f
#         self._conclusion = conclusion or MaxValueAcc()

#     def fuzzify(self, x: torch.Tensor) -> torch.Tensor:
#         if self._threshold_f is not None:
#             return self._threshold_f(x, self.threshold)
#         return (x[:,:,None] >= self.threshold[None]).type_as(x)

#     def hypo(self, m: torch.Tensor) -> ValueWeight:
        
#         return ValueWeight(
#             m * self.threshold[None], m
#         )

#     def accumulate(self, value_weight: ValueWeight) -> torch.Tensor:
#         return self._conclusion.forward(value_weight)


# class ShapeFactory(ABC):

#     @abstractmethod
#     def __call__(self) -> typing.Tuple[shape.Shape, torch.Tensor]:
#         pass


# class SigmoidFuzzyConverter(FuzzyConverter):

#     def __init__(self, out_variables: int, out_terms: int, eps: float=1e-7, conclusion: Conclusion=None):

#         super().__init__()
#         self.weight = nn.parameter.Parameter(
#             torch.randn(out_variables, out_terms)
#         )
#         self.bias = nn.parameter.Parameter(
#             torch.randn(out_variables, out_terms)
#         )
#         self.eps = eps
#         self._conclusion = conclusion or MaxAcc()

#     def fuzzify(self, x: torch.Tensor) -> torch.Tensor:
#         return torch.sigmoid(
#             -(x[:,:,None] - self.bias[None]) * self.weight[None]
#         )

#     def hypo(self, m: torch.Tensor) -> ValueWeight:

#         m = torch.clamp(m, self.eps, 1 - self.eps)
#         return ValueWeight(
#             torch.logit(m) / (self.weight[None]) + self.bias[None], 
#             m
#         )

#     def accumulate(self, value_weight: ValueWeight) -> torch.Tensor:
#         return self._conclusion.forward(value_weight)


# class RangeFuzzyConverter(FuzzyConverter):

#     def __init__(self, out_variables: int, out_terms: int, conclusion: Conclusion=None):

#         super().__init__()
#         self.lower = nn.parameter.Parameter(
#             torch.randn(out_variables, out_terms) * 0.01
#         )
#         self.dx = nn.parameter.Parameter(
#             torch.randn(out_variables, out_terms) * 0.01
#         )
#         self._conclusion = conclusion or MaxAcc()

#     def fuzzify(self, x: torch.Tensor) -> torch.Tensor:
#         lower = self.lower[None]
#         upper = torch.nn.functional.softplus(self.dx[None]) + lower
#         m = (upper - x[:,:,None]) / (upper - lower)
#         return torch.clamp(m, 0, 1)

#     def hypo(self, m: torch.Tensor) -> ValueWeight:
#         lower = self.lower[None]
#         upper = torch.nn.functional.softplus(self.dx[None]) + lower
#         x = upper - m * (upper - lower)
#         return ValueWeight(
#             x, m
#         )

#     def accumulate(self, value_weight: ValueWeight) -> torch.Tensor:
#         return self._conclusion.forward(value_weight)


# class ShapeFuzzyConverter(FuzzyConverter):
#     """Convert 
#     """

#     # def __init__(
#     #     self, n_variables: int, n_terms: int, shape_pts: ShapePoints,
#     #     left_factory: ShapeFactory, middle_factory: ShapeFactory, right_factory: ShapeFactory, 
#     #     hypothesis: typing.Union[ShapeHypothesis, str]="area", 
#     #     conclusion: typing.Union[Conclusion, str]="max", truncate: bool=True
#     # ):
#     #     super().__init__()

#     #     self._shape_pts = shape_pts
#     #     self.truncate = truncate
#     #     self.conclusion = self.get_conclusion(conclusion)
#     #     self.hypothesis = HypothesisEnum.get(hypothesis)
#     #     self._n_variables = n_variables
#     #     self._n_terms = n_terms


#     #     # params = torch.linspace(0.0, 1.0, shape_pts.n_pts)
#     #     # params = params.unsqueeze(0)
#     #     # params = params.repeat(n_variables, 1)

#     #     n_steps = self._get_steps(self._n_terms)
#     #     self._params = self.generate_spaced_params(n_steps, 0, 1)

#     #     self._left, params = left_factory(self._params)
#     #     self._middle, params = middle_factory(params)
#     #     self._right, _ = right_factory(params)
#     # how can i make this more flexible (?)
#     # def generate_params(self):
#     #     positive_params = torch.nn.functional.softplus(self._params)
#     #     cumulated_params = torch.cumsum(positive_params, dim=1)
#     #     min_val = cumulated_params[:,:1]
#     #     max_val = cumulated_params[:,-1:]
#     #     scaled_params  = (cumulated_params - min_val) / (max_val - min_val)
#     #     return scaled_params
    
#     # how to generate params
#     def generate_spaced_params(self, n_steps: int, lower: float=0, upper: float=1) -> torch.Tensor:
#         # n_variables n_terms
#         return torch.linspace(lower, upper, n_steps)[None, None]

#     def generate_repeat_params(self, n_steps: int, value: float) -> torch.Tensor:
#         return torch.full((1, 1, n_steps), value)

#     def _join(self, x: torch.Tensor):
        
#         return torch.cat(
#             [shape.join(x) for shape, _ in self.create_shapes() ],dim=2
#         )
    
#     # ## Put this elsewhere
#     # def create_shapes(self, m: torch.Tensor=None) -> typing.Iterator[typing.Tuple[Shape, torch.Tensor]]:
#     #     left = ShapeParams(
#     #         self._params[:,:self._shape_pts.n_side_pts].view(self._n_variables, 1, -1))
#     #     yield self._left_cls(left), m[:,:,:1] if m is not None else None

#     #     if self._n_terms > 2:
#     #         middle = ShapeParams(
#     #             stride_coordinates(
#     #                 self._params[
#     #                     :,self._shape_pts.side_step:self._shape_pts.side_step + self._shape_pts.n_middle_pts
#     #                 ],
#     #                 self._shape_pts.n_middle_shape_pts, self._shape_pts.step
#     #         ))
#     #         yield self._middle_cls(middle), m[:,:,:-2] if m is not None else None

#     #     right = ShapeParams(
#     #         self._params[:,-self._shape_pts.n_side_pts:].view(self._n_variables, 1, -1)
#     #     )
#     #     yield self._right_cls(right), m[:,:,-1:] if m is not None else None
    
#     def _hypo(self, m: torch.Tensor) -> torch.Tensor:
#         xs = []
#         for shape, m_i in self.create_shapes(m):
#             if self.truncate:
#                 xs.append(shape.truncate(m_i) )
#             else:
#                 xs.append(shape.scale(m_i))
        
#         return self.hypothesis(*xs)

#     def fuzzify(self, x: torch.Tensor) -> torch.Tensor:
#         return self._join(x)

#     def accumulate(self, value_weight: ValueWeight) -> torch.Tensor:
#         return self.conclusion.forward(value_weight)

#     def hypo(self, m: torch.Tensor) -> ValueWeight:
#         return ValueWeight(self._hypo(m), m)


# class LogisticFuzzyConverter(FuzzyConverter):

#     def __init__(
#         self, n_variables: int, n_terms: int, fixed: bool=False,
#         hypothesis: typing.Union[ShapeHypothesis, str]="area", 
#         conclusion: typing.Union[Conclusion, str]="max", truncate: bool=True
#     ):
#         super().__init__()
#         assert n_terms > 1

#         self.truncate = truncate
#         self.conclusion = self.get_conclusion(conclusion)
#         self.hypothesis = HypothesisEnum.get(hypothesis)
#         self._n_variables = n_variables
#         self._n_terms = n_terms
#         biases = torch.linspace(0.0, 1.0, n_terms).unsqueeze(0)
#         self._scales = torch.empty(
#             n_variables, n_terms).fill_(1.0 / (n_terms - 1)
#         ).unsqueeze(2)
#         self._biases = biases.repeat(
#             n_variables, 1).unsqueeze(2)
#         if not fixed:
#             self._biases = nn.parameter.Parameter(self._biases)
#             self._scales = nn.parameter.Parameter(self._scales)

#     def generate_means(self):
#         positive_params = torch.nn.functional.softplus(self._biases)
#         cumulated_params = torch.cumsum(positive_params, dim=1)
#         min_val = cumulated_params[:,:1]
#         max_val = cumulated_params[:,-1:]
#         scaled_params  = (cumulated_params - min_val) / (max_val - min_val)
#         return scaled_params
    
#     def _join(self, x: torch.Tensor):
#         return torch.cat(
#             [shape.join(x) for shape, _ in self.create_shapes() ],dim=2
#         )
    
#     # def create_shapes(self, m: torch.Tensor=None) -> typing.Iterator[typing.Tuple[Shape, torch.Tensor]]:
#     #     left_biases = ShapeParams(
#     #         self._biases[:,:1].view(self._n_variables, 1, 1))
#     #     left_scales = ShapeParams(
#     #         self._scales[:,:1].view(self._n_variables, 1, 1))
        
#     #     yield shape.RightLogistic(left_biases, left_scales, False), m[:,:,:1] if m is not None else None
        
#     #     if self._n_terms > 2:
#     #         mid_biases = ShapeParams(
#     #             self._biases[:,1:-1].view(self._n_variables, self._n_terms - 2, 1))
#     #         mid_scales = ShapeParams(
#     #             self._scales[:,1:-1].view(self._n_variables, self._n_terms - 2, 1))
            
#     #         yield shape.LogisticBell(mid_biases, mid_scales), m[:,:,1:-1] if m is not None else None
            
#     #     right_biases = ShapeParams(
#     #         self._biases[:,-1:].view(self._n_variables, 1, 1))
#     #     right_scales = ShapeParams(
#     #         self._scales[:,-1:].view(self._n_variables, 1, 1))
        
#     #     yield shape.RightLogistic(right_biases, right_scales, True), m[:,:,-1:] if m is not None else None
    
#     def _hypo(self, m: torch.Tensor) -> torch.Tensor:
#         xs = []
#         for shape, m_i in self.create_shapes(m):
#             if self.truncate:
#                 xs.append(shape.truncate(m_i) )
#             else:
#                 xs.append(shape.scale(m_i))
#         return self.hypothesis(*xs)

#     def fuzzify(self, x: torch.Tensor) -> torch.Tensor:
#         return self._join(x)

#     def accumulate(self, value_weight: ValueWeight) -> torch.Tensor:
#         return self.conclusion.forward(value_weight)

#     def hypo(self, m: torch.Tensor) -> ValueWeight:
#         return ValueWeight(self._hypo(m), m)


# class LogisticFuzzyConverter(FuzzyConverter):

#     def __init__(
#         self, n_variables: int, n_terms: int, fixed: bool=False,
#         hypothesis: typing.Union[ShapeHypothesis, str]="area", 
#         conclusion: typing.Union[Conclusion, str]="max", truncate: bool=True
#     ):
#         super().__init__()
#         assert n_terms > 1

#         self.truncate = truncate
#         self.conclusion = self.get_conclusion(conclusion)
#         self.hypothesis = HypothesisEnum.get(hypothesis)
#         self._n_variables = n_variables
#         self._n_terms = n_terms
#         biases = torch.linspace(0.0, 1.0, n_terms).unsqueeze(0)
#         self._scales = torch.empty(
#             n_variables, n_terms).fill_(1.0 / (n_terms - 1)
#         ).unsqueeze(2)
#         self._biases = biases.repeat(
#             n_variables, 1).unsqueeze(2)
#         if not fixed:
#             self._biases = nn.parameter.Parameter(self._biases)
#             self._scales = nn.parameter.Parameter(self._scales)

#     def generate_means(self):
#         positive_params = torch.nn.functional.softplus(self._biases)
#         cumulated_params = torch.cumsum(positive_params, dim=1)
#         min_val = cumulated_params[:,:1]
#         max_val = cumulated_params[:,-1:]
#         scaled_params  = (cumulated_params - min_val) / (max_val - min_val)
#         return scaled_params
    
#     def _join(self, x: torch.Tensor):
#         return torch.cat(
#             [shape.join(x) for shape, _ in self.create_shapes() ],dim=2
#         )
    
#     # def create_shapes(self, m: torch.Tensor=None) -> typing.Iterator[typing.Tuple[Shape, torch.Tensor]]:
#     #     left_biases = ShapeParams(
#     #         self._biases[:,:1].view(self._n_variables, 1, 1))
#     #     left_scales = ShapeParams(
#     #         self._scales[:,:1].view(self._n_variables, 1, 1))
        
#     #     yield shape.RightLogistic(left_biases, left_scales, False), m[:,:,:1] if m is not None else None
        
#     #     if self._n_terms > 2:
#     #         mid_biases = ShapeParams(
#     #             self._biases[:,1:-1].view(self._n_variables, self._n_terms - 2, 1))
#     #         mid_scales = ShapeParams(
#     #             self._scales[:,1:-1].view(self._n_variables, self._n_terms - 2, 1))
            
#     #         yield shape.LogisticBell(mid_biases, mid_scales), m[:,:,1:-1] if m is not None else None
            
#     #     right_biases = ShapeParams(
#     #         self._biases[:,-1:].view(self._n_variables, 1, 1))
#     #     right_scales = ShapeParams(
#     #         self._scales[:,-1:].view(self._n_variables, 1, 1))
        
#     #     yield shape.RightLogistic(right_biases, right_scales, True), m[:,:,-1:] if m is not None else None
    
#     def _hypo(self, m: torch.Tensor) -> torch.Tensor:
#         xs = []
#         for shape, m_i in self.create_shapes(m):
#             if self.truncate:
#                 xs.append(shape.truncate(m_i) )
#             else:
#                 xs.append(shape.scale(m_i))
#         return self.hypothesis(*xs)

#     def fuzzify(self, x: torch.Tensor) -> torch.Tensor:
#         return self._join(x)

#     def accumulate(self, value_weight: ValueWeight) -> torch.Tensor:
#         return self.conclusion.forward(value_weight)

#     def hypo(self, m: torch.Tensor) -> ValueWeight:
#         return ValueWeight(self._hypo(m), m)



# TODO: go through each of these and simplify

# class IsoscelesFuzzyConverter(ShapeFuzzyConverter):

#     def __init__(
#         self, n_variables: int, n_terms: int, 
#         hypothesis: typing.Union[ShapeHypothesis, str]="area", 
#         conclusion: typing.Union[Conclusion, str]="max", flat_edges: bool=False, truncate: bool=True, fixed: bool=False
#     ):
#         if flat_edges:
#             left_cls = shape.DecreasingRightTrapezoid
#             right_cls = shape.IncreasingRightTrapezoid
#             shape_pts = ShapePoints(3, n_terms + 2, n_terms - 1, 2, 1, 1)
#         else:
#             shape_pts = ShapePoints(2, n_terms, n_terms - 1, 2, 0, 1)
#             left_cls = shape.DecreasingRightTriangle
#             right_cls = shape.IncreasingRightTriangle

#         middle_cls = shape.IsoscelesTriangle
        
#         super().__init__(
#             n_variables, n_terms, shape_pts, left_cls, middle_cls, right_cls, fixed, hypothesis, conclusion, truncate
#         )


# I think it is easier to define them one by one
# i don't need this factory stuff


# class TriangleFuzzyConverter(Shape):

#     def __init__(
#         self, n_variables: int, n_terms: int, 
#         hypothesis: typing.Union[ShapeHypothesis, str]="area", 
#         conclusion: typing.Union[Conclusion, str]="max", 
#         flat_edges: bool=False, truncate: bool=True, fixed: bool=False
#     ):

#         if flat_edges:
#             left_cls = shape.DecreasingRightTrapezoid
#             right_cls = shape.IncreasingRightTrapezoid
#             shape_pts = ShapePoints(3, n_terms + 2, n_terms, 3, 1, 1)
#         else:
#             shape_pts = ShapePoints(2, n_terms, n_terms, 3, 0, 1)
#             left_cls = shape.DecreasingRightTriangle
#             right_cls = shape.IncreasingRightTriangle
#         middle_cls = shape.Triangle
        
#         super().__init__(
#             n_variables, n_terms, shape_pts, left_cls, middle_cls, right_cls, fixed, hypothesis, conclusion, truncate
#         )
