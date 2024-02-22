
# 1st party
from abc import abstractmethod, ABC
import typing

# 3rd party
import torch
from torch._tensor import Tensor
import torch.nn as nn
import torch.nn.functional

# local
from .._shapes import Shape
from .. import _shapes as shape
from ._conclude import Conclusion, HypoM
from ._hypo import ShapeHypothesis, HypothesisEnum
from ._utils import stride_coordinates, generate_repeat_params, generate_spaced_params
from .._shapes import Shape, ShapeParams, Composite
from ._conclude import HypoM, Conclusion, ConcEnum
from ._fuzzifiers import Fuzzifier, Defuzzifier


class FuzzyConverter(nn.Module):
    """Convert tensor to fuzzy set and vice versa
    """
    def __init__(self, n_terms: int):
        super().__init__()
        self._n_terms = n_terms

    @property
    def n_terms(self) -> int:
        return self._n_terms

    @abstractmethod
    def fuzzify(self, x: torch.Tensor) -> torch.Tensor:
        """Fuzzify the crisp value

        Args:
            x (torch.Tensor): The value to fuzzify

        Returns:
            torch.Tensor: The fuzzy set
        """
        pass

    @abstractmethod
    def hypo(self, m: torch.Tensor) -> HypoM:
        """Convert the fuzzy set to a 'hypothesis' about how to defuzzify

        Args:
            m (torch.Tensor): The fuzzy set

        Returns:
            HypoWeight: The hypothesis and their weights
        """
        pass

    @abstractmethod
    def conclude(self, hypo_weight: HypoM) -> torch.Tensor:
        """Make a conclusion about how to defuzzify from the hypotheses

        Args:
            hypo_weight (HypoWeight): The hypotheses to use in the conclusion

        Returns:
            torch.Tensor: The defuzzified tensor
        """
        pass

    def defuzzify(self, m: torch.Tensor) -> torch.Tensor:
        """Convenience function to form a hypothesis and make a conclusion

        Args:
            m (torch.Tensor): The fuzzy set

        Returns:
            torch.Tensor: The defuzzified value
        """
        return self.conclude(self.hypo(m))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience function to fuzzify the value

        Args:
            x (torch.Tensor): The input

        Returns:
            torch.Tensor: The fuzzy set
        """
        return self.fuzzify(x)
    
    def reverse(self, m: torch.Tensor) -> torch.Tensor:
        """Convenience function to defuzzify the value

        Args:
            x (torch.Tensor): The fuzzy set

        Returns:
            torch.Tensor: The defuzzified value
        """
        return self.defuzzify(m)
    
    def fuzzifier(self) -> 'ConverterFuzzifier':
        """

        Returns:
            ConverterFuzzifier: A fuzzifier built from self
        """
        return ConverterFuzzifier(self)
    
    def defuzzifier(self) -> 'ConverterDefuzzifier':
        """

        Returns:
            ConverterFuzzifier: A defuzzifier built from self
        """
        return ConverterDefuzzifier(self)


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
        super().__init__(sum([shape.n_terms for shape in shapes]))

        self._composite = Composite(shapes)
        self._hypothesis = HypothesisEnum.get(hypothesis)
        self._conclusion = ConcEnum.get(conclusion)
        self._truncate = truncate

    def fuzzify(self, x: torch.Tensor) -> torch.Tensor:
        return self._composite.join(x)

    def conclude(self, hypo_weight: HypoM) -> torch.Tensor:
        """Draw a conclusion from the input

        Args:
            hypo_weight (HypoWeight): The hypotheses and their weights

        Returns:
            torch.Tensor: The consequent of the fuzzy system
        """
        return self._conclusion.forward(hypo_weight)

    def hypo(self, m: torch.Tensor) -> HypoM:
        if self._truncate:
            shapes = self._composite.truncate(m).shapes
        else:
            shapes = self._composite.scale(m).shapes
        return HypoM(self._hypothesis(*shapes), m)


def polygon_set(left: shape.Shape, middle: shape.Shape, right: shape.Shape) -> typing.List[Shape]:
    """Create a set of shapes from left middle and right. If middle is none only middle will be returned

    Args:
        left (shape.Shape): The leftmost polygon
        middle (shape.Shape): The center polygon
        right (shape.Shape): The rightmost polygon

    Returns:
        typing.List[Shape]: The set of shapes created
    """
    if middle is None:
        return [left, right]
    return [left, middle, right]


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
            truncate (bool, optional): Whether to truncate or scale. Defaults to False.
        """
        super().__init__(
            polygon_set(left, middle, right), hypothesis, conclusion, truncate
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
            truncate (bool, optional): Whether to truncate or scale. Defaults to False.
        """
        super().__init__(
            polygon_set(left, middle, right), hypothesis, conclusion, truncate
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
            truncate (bool, optional): Whether to truncate or not. Defaults to True.
        """
        super().__init__(
            polygon_set(left, middle, right), hypothesis, conclusion, truncate
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
            truncate (bool, optional): Whether to truncate or not. Defaults to True.
        """
        super().__init__(
            polygon_set(left, middle, right), hypothesis, conclusion, truncate
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
        """Create a fuzzy converter based on a 'square' function

        Args:
            square (Square), The set of squares to use.
            hypothesis (typing.Union[ShapeHypothesis, str], optional): The hypothesis function to use. Defaults to "area".
            conclusion (typing.Union[Conclusion, str], optional): The conclusion function to use. Defaults to "max".
            truncate (bool, optional): Whether to use truncate or scale. Defaults to True.
        """
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
        truncate: bool=True
    ):
        """Create a fuzzy converter based on the logistic distribution

        Args:
            left (shape.RightLogistic): The left logistic
            right (shape.RightLogistic): The right logistic
            middle (shape.LogisticBell, optional): The middle logistics . Defaults to None.
            hypothesis (typing.Union[ShapeHypothesis, str], optional): The hypothesis function to use. Defaults to "area".
            conclusion (typing.Union[Conclusion, str], optional): The conclusion function to use. Defaults to "max".
            truncate (bool, optional): Whether to use truncate or scale. Defaults to True.
        """
        super().__init__(
            polygon_set(left, middle, right), hypothesis, conclusion, truncate
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
        hypothesis: typing.Union[ShapeHypothesis, str]="min_core", 
        conclusion: typing.Union[Conclusion, str]="max",
    ):
        sigmoid = shape.Sigmoid(
            ShapeParams(bias_coords[:,:,:,None]), ShapeParams(scale_coords[:,:,:,None])
        )
        return SigmoidFuzzyConverter(sigmoid, hypothesis, conclusion, True)

    @classmethod
    def from_linspace(
        cls, n_terms: int, hypothesis: typing.Union[ShapeHypothesis, str]="min_core", 
        conclusion: typing.Union[Conclusion, str]="max", 
    ):
        bias_coords = generate_spaced_params(n_terms + 2)[:,:,1:-1]
        width = 1.0 / (2 * (n_terms + 1))
        scale_coords = generate_repeat_params(n_terms, width)

        return SigmoidFuzzyConverter.from_coords(
            bias_coords, scale_coords, n_terms, hypothesis, conclusion
        )


class RampFuzzyConverter(CompositeFuzzyConverter):
    """A fuzzifier that makes use of a series of ramps
    """

    def __init__(
        self, ramp: shape.Ramp=None, 
        hypothesis: typing.Union[ShapeHypothesis, str]="area", 
        conclusion: typing.Union[Conclusion, str]="max",
    ):
        """Create a fuzzy converter that makes use of the ramp function

        Args:
            ramp (shape.Ramp, optional): The ramp function. Defaults to None.
            hypothesis (typing.Union[ShapeHypothesis, str], optional): The hypothesizer to use. Defaults to "area".
            conclusion (typing.Union[Conclusion, str], optional): The concluder to use. Defaults to "max".
        """
        super().__init__(
            [ramp], hypothesis, conclusion, True
        )

    @classmethod
    def from_coords(
        cls, coords: torch.Tensor, n_terms: int,
        hypothesis: typing.Union[ShapeHypothesis, str]="min_core", 
        conclusion: typing.Union[Conclusion, str]="max"
    ) -> 'RampFuzzyConverter':
        """Create the converter from coordinates. Each ramp function uses two coordinates with a stride of two

        Args:
            coords (torch.Tensor): The coordinates defining the ramp function
            n_terms (int): The number of terms for the fuzzy converter
            hypothesis (typing.Union[ShapeHypothesis, str], optional): The hypothesis function. Defaults to "min_core".
            conclusion (typing.Union[Conclusion, str], optional): The conclusion function for the ramp function. Defaults to "max".

        Returns:
            RampFuzzyConverter: The resulting FuzzyConverter using ramp functions
        """
        
        ramp = shape.Ramp(
            ShapeParams(stride_coordinates(coords, n_terms, 1, 2, 2))
        )
        return RampFuzzyConverter(ramp, hypothesis, conclusion)

    @classmethod
    def from_linspace(
        cls, n_terms: int, hypothesis: typing.Union[ShapeHypothesis, str]="min_core", 
        conclusion: typing.Union[Conclusion, str]="max", 
    ) -> 'RampFuzzyConverter':
        """Create the converter from a linspace with n terms

        Args:
            n_terms (int): The number of terms for the fuzzy converter
            hypothesis (typing.Union[ShapeHypothesis, str], optional): The hypothesis function. Defaults to "min_core".
            conclusion (typing.Union[Conclusion, str], optional): The conclusion function for the ramp function. Defaults to "max".

        Returns:
            RampFuzzyConverter: The resulting FuzzyConverter using ramp functions
        """
        coords = generate_spaced_params(n_terms + 2)
        return RampFuzzyConverter.from_coords(
            coords, n_terms, hypothesis, conclusion,
        )


class StepFuzzyConverter(CompositeFuzzyConverter):
    """A 'fuzzifier' that makes use of a series of step functions
    """

    def __init__(
        self, step: shape.Step=None, 
        hypothesis: typing.Union[ShapeHypothesis, str]="min_core", 
        conclusion: typing.Union[Conclusion, str]="max",
    ):
        """Create a 'fuzzifier' that uses the step function. 

        Args:
            step (shape.Step, optional): The step function to use. Defaults to None.
            hypothesis (typing.Union[ShapeHypothesis, str], optional): The hypothesizer to use. Defaults to "min_core".
            conclusion (typing.Union[Conclusion, str], optional): The conclusion to use. Defaults to "max".
        """
        super().__init__(
            [step], hypothesis, conclusion, True
        )

    @classmethod
    def from_coords(
        cls, coords: torch.Tensor, n_terms: int,
        hypothesis: typing.Union[ShapeHypothesis, str]="min_core", 
        conclusion: typing.Union[Conclusion, str]="max", 
    ) -> 'StepFuzzyConverter':
        """Create the StepFuzzyConverter from coordinates

        Args:
            coords (torch.Tensor): A series of thresholds use for the step functions
            n_terms (int): The number of terms to use
            hypothesis (typing.Union[ShapeHypothesis, str], optional): The hypothesizer to use. Defaults to "min_core".
            conclusion (typing.Union[Conclusion, str], optional): The conclusion to use. Defaults to "max".

        Returns:
            StepFuzzyConverter: the created StepFuzzyConverter
        """
        step = shape.Step(
            ShapeParams(stride_coordinates(coords, n_terms, 1, 1, 1))
        )
        return StepFuzzyConverter(step, hypothesis, conclusion)

    @classmethod
    def from_linspace(
        cls, n_terms: int, hypothesis: typing.Union[ShapeHypothesis, str]="min_core", 
        conclusion: typing.Union[Conclusion, str]="max", 
    ) -> 'StepFuzzyConverter':
        """Create the StepFuzzyConverter from coordinates

        Args:
            n_terms (int): The number of terms
            hypothesis (typing.Union[ShapeHypothesis, str], optional): The hypothesizer to use. Defaults to "min_core".
            conclusion (typing.Union[Conclusion, str], optional): The conclusion to use. Defaults to "max".

        Returns:
            StepFuzzyConverter: the created StepFuzzyConverter
        """
        coords = generate_spaced_params(n_terms + 2)[:,:,1:-1]
        
        return StepFuzzyConverter.from_coords(
            coords, n_terms, hypothesis, conclusion,
        )


class ConverterDecorator(ABC, FuzzyConverter):
    """Define a decorator for the converter
    """

    def __init__(self, converter: FuzzyConverter):
        """

        Args:
            converter (FuzzyConverter): 
        """
        super().__init__(converter.n_terms)
        self._converter = converter

    @abstractmethod
    def decorate_fuzzify(self, x: torch.Tensor) -> torch.Tensor:
        """Decorate the fuzzification function

        Args:
            x (torch.Tensor): The crisp value to fuzzify

        Returns:
            torch.Tensor: The decorated crisp value
        """
        pass

    @abstractmethod
    def decorate_defuzzify(self, m: torch.Tensor) -> torch.Tensor:
        """Decorate the defuzzifier

        Args:
            m (torch.Tensor): The membership

        Returns:
            torch.Tensor: The decorated membership
        """
        pass

    def fuzzify(self, x: torch.Tensor) -> torch.Tensor:
        """Fuzzify the message

        Args:
            x (torch.Tensor): The crisp value to fuzzify

        Returns:
            torch.Tensor: The resulting tensor
        """
        return self._converter.fuzzify(self.decorate_fuzzify(x))

    def conclude(self, hypo_weight: HypoM) -> torch.Tensor:
        """Use the hyptoheses to determine the result

        Args:
            hypo_weight (HypoWeight): The hypotheses and  their weights

        Returns:
            torch.Tensor: The conclusion based on the hypotheses
        """
        return self._converter.conclude(hypo_weight)
    
    def hypo(self, m: torch.Tensor) -> HypoM:
        """

        Args:
            m (torch.Tensor): The membership value

        Returns:
            HypoWeight: The hypothesis and the weight
        """
        return self.decorate_defuzzify(self._converter.hypo(m))
    

class FuncConverterDecorator(ConverterDecorator):

    def __init__(self, converter: FuzzyConverter, fuzzify: typing.Callable[[torch.Tensor], torch.Tensor], defuzzify: typing.Callable[[torch.Tensor], torch.Tensor]):
        """Use functions to decorate the fuzzification and defuzzification functions

        Args:
            converter (FuzzyConverter): The converter to decorate
            fuzzify (typing.Callable[[torch.Tensor], torch.Tensor]): The fuzzification function
            defuzzify (typing.Callable[[torch.Tensor], torch.Tensor]): The defuzzification function
        """
        super().__init__(converter)
        self._fuzzify = fuzzify
        self._defuzzify = defuzzify

    def decorate_fuzzify(self, x: torch.Tensor) -> torch.Tensor:

        """Decorate the defuzzifier

        Args:
            m (torch.Tensor): The membership

        Returns:
            torch.Tensor: The fuzzified value
        """
        return self._fuzzify(x)

    def decorate_defuzzify(self, m: torch.Tensor) -> torch.Tensor:
        """Decorate the defuzzifier

        Args:
            m (torch.Tensor): The membership

        Returns:
            torch.Tensor: The decorated crisp value
        """
        return self._defuzzify(m)


class ConverterDefuzzifier(Defuzzifier):

    def __init__(self, converter: FuzzyConverter):
        """Wrap a FuzzyConverter to create a defuzzifier

        Args:
            converter (FuzzyConverter): The fuzzy converter to wrap
        """
        super().__init__(converter.n_terms)
        self.converter = converter

    def hypo(self, m: torch.Tensor) -> HypoM:
        """Calculate the hypothesis

        Args:
            m (torch.Tensor): The fuzzy set input

        Returns:
            HypoWeight: The hypothesis and weight
        """
        return self.converter.hypo(m)

    def conclude(self, hypo_weight: HypoM) -> torch.Tensor:
        """Use the hyptoheses to determine the result

        Args:
            hypo_weight (HypoWeight): The hypotheses and  their weights

        Returns:
            torch.Tensor: The conclusion based on the hypotheses
        """
        return self.converter.conclude(hypo_weight)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        return self.converter.defuzzify(m)


class ConverterFuzzifier(Fuzzifier):

    def __init__(self, converter: FuzzyConverter):
        """Wrap a FuzzyConverter to create a fuzzifier

        Args:
            converter (FuzzyConverter): The fuzzy converter to wrap
        """
        super().__init__(converter.n_terms)
        self.converter = converter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The input

        Args:
            x (torch.Tensor): the input

        Returns:
            torch.Tensor: the fuzzified input
        """
        return self.converter.fuzzify(x)

