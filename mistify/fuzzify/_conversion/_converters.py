
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
from ._conclude import Conclusion, HypoWeight, MaxConc, WeightedAverageConc
from ._hypo import ShapeHypothesis, HypothesisEnum
from ._utils import stride_coordinates
from .._shapes import Shape, ShapeParams, Composite
from ._conclude import HypoWeight, Conclusion, MaxValueConc, ConcEnum
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
    def hypo(self, m: torch.Tensor) -> HypoWeight:
        pass

    @abstractmethod
    def conclude(self, hypo_weight: HypoWeight) -> torch.Tensor:
        pass

    def defuzzify(self, m: torch.Tensor) -> torch.Tensor:
        return self.conclude(self.hypo(m))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuzzify(x)
    
    def reverse(self, m: torch.Tensor) -> torch.Tensor:
        return self.defuzzify(m)

    # def get_conclusion(self, conclusion: typing.Union['Conclusion', str]):
    #     """Get the conclusion to use in conversion

    #     Args:
    #         conclusion (typing.Union[&#39;Conclusion&#39;, str]): _description_

    #     Raises:
    #         ValueError: _description_

    #     Returns:
    #         _type_: _description_
    #     """
    #     if isinstance(conclusion, Conclusion):
    #         return conclusion
    #     if conclusion == 'max':
    #         return MaxConc()
    #     if conclusion == 'weighted_average':
    #         return WeightedAverageConc()
    #     raise ValueError(f"Name {conclusion} cannot be created")


class CompositeFuzzyConverter(FuzzyConverter):
    """Define a conversion of multiple shapes to and from fuzzy sets
    """

    def __init__(
        self, shapes: typing.List[Shape], 
        hypothesis: typing.Union[ShapeHypothesis, str]="area", 
        conclusion: typing.Union[Conclusion, str]="max",
        truncate: bool=False
    ):
        """Combine shapes to use in converting

        Args:
            shapes (typing.List[Shape]): The shapes to use in converting
            hypothesis (typing.Union[ShapeHypothesis, str], optional): The function to use to get the hypotheses. Defaults to "area".
            conclusion (typing.Union[Conclusion, str], optional): The function to draw the conclusion from. Defaults to "max".
            truncate (bool, optional): Whether to truncate (True) or scale (False). Defaults to False.
        """
        super().__init__()

        self._composite = Composite(shapes)
        self._hypothesis = HypothesisEnum.get(hypothesis)
        self._conclusion = ConcEnum.get(conclusion)
        self._truncate = truncate

    def fuzzify(self, x: torch.Tensor) -> torch.Tensor:
        return self._composite.join(x)

    def conclude(self, hypo_weight: HypoWeight) -> torch.Tensor:
        """Draw a conclusion from the input

        Args:
            hypo_weight (HypoWeight): The hypotheses and their weights

        Returns:
            torch.Tensor: The consequent of the fuzzy system
        """
        return self._conclusion.forward(hypo_weight)

    def hypo(self, m: torch.Tensor) -> HypoWeight:
        if self._truncate:
            shapes = self._composite.truncate(m).shapes
        else:
            shapes = self._composite.scale(m).shapes
        return HypoWeight(self._hypothesis(*shapes), m)


def polygon(left: shape.Shape, middle: shape.Shape, right: shape.Shape):

    if middle is None:
        return [left, right]
    return [left, right, middle]


class IsoscelesFuzzyConverter(CompositeFuzzyConverter):
    """Create a FuzzyConverter consisting of isosceles triangles
    """

    def __init__(
        self, left: typing.Union[shape.DecreasingRightTrapezoid, shape.DecreasingRightTriangle],
        right: typing.Union[shape.IncreasingRightTrapezoid, shape.IncreasingRightTriangle],
        middle: shape.IsoscelesTriangle=None, 
        hypothesis: typing.Union[ShapeHypothesis, str]="area", 
        conclusion: typing.Union[Conclusion, str]="max", 
        truncate: bool=False
    ):
        """

        Args:
            left (typing.Union[shape.DecreasingRightTrapezoid, shape.DecreasingRightTriangle]): The shape on the left
            right (typing.Union[shape.IncreasingRightTrapezoid, shape.IncreasingRightTriangle]): The shape on the right
            middle (shape.IsoscelesTriangle, optional): The middle shape. Defaults to None.
            hypothesis (typing.Union[ShapeHypothesis, str], optional): The hypothesis fucntion. Defaults to "area".
            conclusion (typing.Union[Conclusion, str], optional): The conclusion function. Defaults to "max".
            truncate (bool, optional): _description_. Defaults to False.
        """
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
    ) -> 'IsoscelesFuzzyConverter':
        """Create the IsoscelesFuzzy converter from coordinates

        Args:
            coords (torch.Tensor): The coords to use in creation
            n_terms (int): The number of terms for the shape
            hypothesis (typing.Union[ShapeHypothesis, str], optional): The hypothesis function. Defaults to "area".
            conclusion (typing.Union[Conclusion, str], optional): The conclusion function. Defaults to "max".
            flat_edges (bool, optional): Whether to use flat edges or not. Defaults to False.
            truncate (bool, optional): Whether to truncate or not. Defaults to True.

        Returns:
            IsoscelesFuzzyConverter
        """
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
    ) -> 'IsoscelesFuzzyConverter':
        """Create the IsoscelesFuzzy converter from coordinates

        Args:
            n_terms (int): The number of terms for the shape
            hypothesis (typing.Union[ShapeHypothesis, str], optional): The hypothesis function. Defaults to "area".
            conclusion (typing.Union[Conclusion, str], optional): The conclusion function. Defaults to "max".
            flat_edges (bool, optional): Whether to use flat edges or not. Defaults to False.
            truncate (bool, optional): Whether to truncate or not. Defaults to True.

        Returns:
            IsoscelesFuzzyConverter
        """
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
        """

        Args:
            left (typing.Union[shape.DecreasingRightTrapezoid, shape.DecreasingRightTriangle]): The shape on the left
            right (typing.Union[shape.IncreasingRightTrapezoid, shape.IncreasingRightTriangle]): The shape on the right
            middle (shape.IsoscelesTrapezoid, optional): The middle shape. Defaults to None.
            hypothesis (typing.Union[ShapeHypothesis, str], optional): The hypothesis fucntion. Defaults to "area".
            conclusion (typing.Union[Conclusion, str], optional): The conclusion function. Defaults to "max".
            truncate (bool, optional): _description_. Defaults to False.
        """
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
        """Create the IsoscelesTrapezoidFuzzyConverter converter from coordinates

        Args:
            coords (torch.Tensor): The coords to use in creation
            n_terms (int): The number of terms for the shape
            hypothesis (typing.Union[ShapeHypothesis, str], optional): The hypothesis function. Defaults to "area".
            conclusion (typing.Union[Conclusion, str], optional): The conclusion function. Defaults to "max".
            flat_edges (bool, optional): Whether to use flat edges or not. Defaults to False.
            truncate (bool, optional): Whether to truncate or not. Defaults to True.

        Returns:
            IsoscelesTrapezoidFuzzyConverter
        """
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
    ) -> 'IsoscelesTrapezoidFuzzyConverter':
        """Create the IsoscelesTrapezoidFuzzyConverter converter from coordinates

        Args:
            n_terms (int): The number of terms for the shape
            hypothesis (typing.Union[ShapeHypothesis, str], optional): The hypothesis function. Defaults to "area".
            conclusion (typing.Union[Conclusion, str], optional): The conclusion function. Defaults to "max".
            flat_edges (bool, optional): Whether to use flat edges or not. Defaults to False.
            truncate (bool, optional): Whether to truncate or not. Defaults to True.

        Returns:
            IsoscelesTrapezoidFuzzyConverter
        """
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
        """

        Args:
            left (typing.Union[shape.DecreasingRightTrapezoid, shape.DecreasingRightTriangle]): The shape on the left
            right (typing.Union[shape.IncreasingRightTrapezoid, shape.IncreasingRightTriangle]): The shape on the right
            middle (shape.Trapezoid, optional): The middle shape. Defaults to None.
            hypothesis (typing.Union[ShapeHypothesis, str], optional): The hypothesis fucntion. Defaults to "area".
            conclusion (typing.Union[Conclusion, str], optional): The conclusion function. Defaults to "max".
            truncate (bool, optional): _description_. Defaults to False.
        """
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
        """Create the TrapezoidFuzzyConverter converter from coordinates

        Args:
            coords (torch.Tensor): The coords to use in creation
            n_terms (int): The number of terms for the shape
            hypothesis (typing.Union[ShapeHypothesis, str], optional): The hypothesis function. Defaults to "area".
            conclusion (typing.Union[Conclusion, str], optional): The conclusion function. Defaults to "max".
            flat_edges (bool, optional): Whether to use flat edges or not. Defaults to False.
            truncate (bool, optional): Whether to truncate or not. Defaults to True.

        Returns:
            TrapezoidFuzzyConverter
        """
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
        """Create the TrapezoidFuzzyConverter converter from coordinates

        Args:
            n_terms (int): The number of terms for the shape
            hypothesis (typing.Union[ShapeHypothesis, str], optional): The hypothesis function. Defaults to "area".
            conclusion (typing.Union[Conclusion, str], optional): The conclusion function. Defaults to "max".
            flat_edges (bool, optional): Whether to use flat edges or not. Defaults to False.
            truncate (bool, optional): Whether to truncate or not. Defaults to True.

        Returns:
            TrapezoidFuzzyConverter
        """
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
        """

        Args:
            left (typing.Union[shape.DecreasingRightTrapezoid, shape.DecreasingRightTriangle]): The shape on the left
            right (typing.Union[shape.IncreasingRightTrapezoid, shape.IncreasingRightTriangle]): The shape on the right
            middle (shape.Triangle, optional): The middle shape. Defaults to None.
            hypothesis (typing.Union[ShapeHypothesis, str], optional): The hypothesis fucntion. Defaults to "area".
            conclusion (typing.Union[Conclusion, str], optional): The conclusion function. Defaults to "max".
            truncate (bool, optional): _description_. Defaults to False.
        """
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
        """Create the TriangleFuzzyConverter converter from coordinates

        Args:
            coords (torch.Tensor): The coords to use in creation
            n_terms (int): The number of terms for the shape
            hypothesis (typing.Union[ShapeHypothesis, str], optional): The hypothesis function. Defaults to "area".
            conclusion (typing.Union[Conclusion, str], optional): The conclusion function. Defaults to "max".
            flat_edges (bool, optional): Whether to use flat edges or not. Defaults to False.
            truncate (bool, optional): Whether to truncate or not. Defaults to True.

        Returns:
            TriangleFuzzyConverter
        """
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
        """Create the TriangleFuzzyConverter converter from coordinates

        Args:
            n_terms (int): The number of terms for the shape
            hypothesis (typing.Union[ShapeHypothesis, str], optional): The hypothesis function. Defaults to "area".
            conclusion (typing.Union[Conclusion, str], optional): The conclusion function. Defaults to "max".
            flat_edges (bool, optional): Whether to use flat edges or not. Defaults to False.
            truncate (bool, optional): Whether to truncate or not. Defaults to True.

        Returns:
            TriangleFuzzyConverter
        """
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

    def conclude(self, value_weight: HypoWeight) -> torch.Tensor:
        return self._converter.conclude(value_weight)
    
    def hypo(self, m: torch.Tensor) -> HypoWeight:
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
