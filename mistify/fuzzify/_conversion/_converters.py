
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
from ._conclude import Conclusion, HypoWeight
from ._hypo import ShapeHypothesis, HypothesisEnum
from ._utils import stride_coordinates, generate_repeat_params, generate_spaced_params
from .._shapes import Shape, Coords, Composite
from ._conclude import HypoWeight, Conclusion, ConcEnum
from ._fuzzifiers import Fuzzifier, Defuzzifier
from ..._base import Constrained
from ..._functional import G


class ShapeFuzzifier(Fuzzifier, Constrained):
    """Define a conversion of multiple shapes to and from fuzzy sets
    """

    def __init__(
        self, shapes: typing.List[Shape], 
        truncate: bool=False, tunable: bool=False
    ):
        """Combine shapes to use in converting

        Args:
            shapes (typing.List[Shape]): The shapes to use in converting
            truncate (bool, optional): Whether to truncate (True) or scale (False). Defaults to False.
        """
        composite = Composite(shapes)
        composite.requires_grad_(tunable)
        super().__init__(
            composite.n_terms, composite.n_vars
        )
        
        self._composite = composite
        self._truncate = truncate

    def defuzzifier(
            self, 
            hypothesis: typing.Union[ShapeHypothesis, str]="area", 
            conclusion: typing.Union[Conclusion, str]="max",
            truncate: bool=False
    ) -> Defuzzifier:
        """Create a defuzzifier from the fuzzifier

        Args:
            hypothesis (typing.Union[ShapeHypothesis, str], optional): The hypothesis to use. Defaults to "area".
            conclusion (typing.Union[Conclusion, str], optional): The conclusion to use. Defaults to "max".
            truncate (bool, optional): Whether to truncate (True) or scale (False). Defaults to False.

        Returns:
            Defuzzifier: The ShapeDefuzzifier that wraps the fuzzifier
        """
        return ShapeDefuzzifier(
            self._composite, hypothesis, conclusion, truncate
        )

    def fuzzify(self, x: torch.Tensor) -> torch.Tensor:
        """Fuzzify the input

        Args:
            x (torch.Tensor): The input

        Returns:
            torch.Tensor: The membership
        """
        return self._composite.join(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fuzzify the input

        Args:
            x (torch.Tensor): The input

        Returns:
            torch.Tensor: The membership
        """
        return self.fuzzify(x)

    def constrain(self):
        """Constrain the fuzzifier
        """
        self._composite.constrain()


class ShapeDefuzzifier(Defuzzifier):
    """ShapeDefuzzifier uses a composite shape in order to defuzzify
    """

    def __init__(
        self, composite: Composite, hypothesis: ShapeHypothesis='area', 
        conclusion: Conclusion='max', truncate: bool=False
    ):
        """Create a ShapeDefuzzifier 

        Args:
            composite (Composite): The shape used for defuzzification
            hypothesis (ShapeHypothesis, optional): The hypothesis to use for defuzzificaiton. Defaults to 'area'.
            conclusion (Conclusion, optional): The conclusion to use for defuzzification. Defaults to 'max'.
            truncate (bool, optional): Whether to truncate (True) or scale (False). Defaults to False.
        """
        super().__init__(composite.n_terms, composite.n_vars)
        self._composite = composite
        self._hypothesis = HypothesisEnum.get(hypothesis)
        self._conclusion = ConcEnum.get(conclusion, self._n_terms, self._n_vars)
        self._truncate = truncate

    def conclude(self, hypo_weight: HypoWeight) -> torch.Tensor:
        """Draw a conclusion from the input

        Args:
            hypo_weight (HypoM): The hypotheses and their weights

        Returns:
            torch.Tensor: The consequent of the fuzzy system
        """
        return self._conclusion(hypo_weight)

    def hypo(self, m: torch.Tensor) -> HypoWeight:
        """Calculate the hypotheses for defuzzifying the membership and their weights

        Args:
            m (torch.Tensor): The membership

        Returns:
            HypoWeight: The hypothesis along with its weight (in general the membership)
        """
        return self._hypothesis(self._composite.shapes, m)
    
    def forward(self, m: torch.Tensor, weight_override: torch.Tensor=None) -> torch.Tensor:
        """Defuzzify the membership

        Args:
            m (torch.Tensor): The membership
            weight_override (torch.Tensor, optional): The tensor to override the weight with. Defaults to None.

        Returns:
            torch.Tensor: The defuzzified membership tensor
        """
        hypothesis = self.hypo(m)
        hypothesis.weight = weight_override or hypothesis.weight
        return self.conclude(hypothesis)
    

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


def validate_terms(*xs: torch.Tensor) -> typing.List[torch.Tensor]:
    """Validate the number of terms

    Returns:
        typing.List[torch.Tensor]: Each membership tensor after being validated
    """
    # Think more about this
    dim = None
    n_terms = None
    result = []
    for x in xs:
        if x.dim() == 1:
            x = x.unsqueeze(0)

        if dim is None:
            dim = x.dim()

            n_terms = None if dim == 2 else x.size(0)
        else:
            if n_terms is not None:
                if n_terms != x.size(0):
                    raise RuntimeError(f'The number of terms is not aligned for the shapes')
            if x.dim() != dim:
                raise RuntimeError(f'The number of dimensions is not aligned')
        result.append(x)
    if (dim != 2) and (dim != 3):
        raise RuntimeError(f'The dimension of the shapes must be 2 or 3 not {dim}')
    
    return x


def validate_n_points(*xs: torch.Tensor, n_points: int=None, ignore_none: bool=True):
    """Validate the number of points for the shape

    Args:
        n_points (int, optional): The number of points necessary for the shape. Defaults to None.
        ignore_none (bool, optional): Whether to ignore if the number of points is none. Defaults to True.

    Raises:
        RuntimeError: If the number of points for the shape is not valid
    """
    for x in xs:
        if ignore_none and x is None:
            continue
        if n_points is not None and x.size(-1) != n_points:
            raise RuntimeError(f'The number of points for the shape must be {n_points}')


def validate_right_trapezoid(*xs: torch.Tensor):
        
    for x in xs:
        if x.size(-1) not in (2, 3):
            raise RuntimeError(f'Number of points for shape must be two (triangle) or three (trapezoid)')


def flat_edges(x: torch.Tensor, base_size: int) -> bool:
    """Get whether the edges for a shape are flat

    Args:
        x (torch.Tensor): The shape tensor
        base_size (int): The base size for the tensor

    Raises:
        RuntimeError: The shape size is invalid

    Returns:
        bool: If the edges are flat
    """

    if x.size(-1) == base_size:
        return False
    if x.size(-1) == (base_size + 1):
        return True
    raise RuntimeError(f'Invalid size for x. Must be either {base_size} or {base_size + 1}')


class IsoscelesFuzzifier(ShapeFuzzifier):
    """Create a IsoscelesFuzzifier consisting of isosceles triangles
    """

    def __init__(
        self, left: typing.Union[shape.RightTrapezoid, shape.RightTriangle],
        right: typing.Union[shape.RightTrapezoid, shape.RightTriangle],
        middle: shape.IsoscelesTriangle=None, tunable: bool=False, 
    ):
        """

        Args:
            left (typing.Union[shape.RightTrapezoid, shape.RightTriangle]): The shape on the left
            right (typing.Union[shape.RightTrapezoid, shape.RightTriangle]): The shape on the right
            middle (shape.IsoscelesTriangle, optional): The middle shape. Defaults to None.
        """
        super().__init__(
            polygon_set(left, middle, right), tunable
        )

    @classmethod
    def create(
        cls, left: torch.Tensor, right: torch.Tensor, middle: torch.Tensor=None, 
        tunable: bool=False, g: G=None
    ) -> 'IsoscelesFuzzifier':
        """Create an IsoscelesFuzzifier from shapes defined by tensors

        Args:
            left (torch.Tensor): The tensor for the left shape
            right (torch.Tensor): The tensor for the right shape
            middle (torch.Tensor, optional): The tensor for the middle shape(s). Defaults to None.
            tunable (bool, optional): WHether the shape parameters can be updated. Defaults to False.
            g (G, optional): The gradient estimator to use. Defaults to None.

        Returns:
            IsoscelesFuzzifier: The new fuzzifier
        """
        left, right, middle = validate_terms(left, right, middle)
        validate_n_points(middle, n_points=2)
        validate_right_trapezoid(left, right)
        left_shape = shape.RightTrapezoid if flat_edges(left, 2) else shape.RightTriangle
        right_shape = shape.RightTrapezoid if flat_edges(left, 2) else shape.RightTriangle
        left_shape = left_shape(Coords(left), False, g=g)
        right_shape = right_shape(Coords(right), True, g=g)
        if middle is not None:
            middle = shape.IsoscelesTriangle(Coords(middle), g)
        return IsoscelesFuzzifier(
            left, right, middle, tunable
        )

    @classmethod
    def from_coords(
        cls, coords: torch.Tensor, n_terms: int,
        flat_edges: bool=False, tunable: bool=False, g: G=None
    ) -> 'IsoscelesFuzzifier':
        """Create the IsoscelesFuzzifier converter from coordinates

        Args:
            coords (torch.Tensor): The coords to use in creation
            n_terms (int): The number of terms for the shape
            flat_edges (bool, optional): Whether to use flat edges or not. Defaults to False.

        Returns:
            IsoscelesFuzzifier
        """
        middle = None
        if flat_edges:
            left = shape.RightTrapezoid(Coords(coords[:,:,None,:3]), False, g=g)
            if n_terms > 2:
                middle = shape.IsoscelesTriangle(Coords(stride_coordinates(coords[:,:,1:-1], n_terms - 2, 1, 2)), g=g)
            right = shape.RightTrapezoid(Coords(coords[:,:,None,-3:]), True, g=g)

        else:
            left = shape.RightTriangle(Coords(coords[:,:,None,:2]), False, g=g)
            if n_terms > 2:
                middle = shape.IsoscelesTriangle(Coords(stride_coordinates(coords, n_terms - 2, 1, 2)), g=g)
            right = shape.RightTriangle(Coords(coords[:,:,None,-2:]), True, g=g)
        return IsoscelesFuzzifier(left, right, middle, tunable)

    @classmethod
    def from_linspace(
        cls, n_terms: int, n_vars: int=None,
        flat_edges: bool=False, tunable: bool=False, g: G=None
    ) -> 'IsoscelesFuzzifier':
        """Create the IsoscelesFuzzy converter from coordinates

        Args:
            n_terms (int): The number of terms for the shape
            flat_edges (bool, optional): Whether to use flat edges or not. Defaults to False.

        Returns:
            IsoscelesFuzzifier
        """
        if flat_edges:
            coords = generate_spaced_params(n_terms + 2, in_features=n_vars)
        else:
            coords = generate_spaced_params(n_terms, in_features=n_vars)
        return IsoscelesFuzzifier.from_coords(
            coords, n_terms, 
            flat_edges, tunable, g
        )


class IsoscelesTrapezoidFuzzifier(ShapeFuzzifier):

    def __init__(
        self, left: typing.Union[shape.RightTrapezoid, shape.RightTriangle],
        right: typing.Union[shape.RightTrapezoid, shape.RightTriangle],
        middle: shape.IsoscelesTrapezoid=None, tunable: bool=False
    ):
        """Create an IsoscelesTrapezoidFuzzifier

        Args:
            left (typing.Union[shape.DecreasingRightTrapezoid, shape.DecreasingRightTriangle]): The shape on the left
            right (typing.Union[shape.IncreasingRightTrapezoid, shape.IncreasingRightTriangle]): The shape on the right
            middle (shape.IsoscelesTrapezoid, optional): The middle shape. Defaults to None.
            hypothesis (typing.Union[ShapeHypothesis, str], optional): The hypothesis fucntion. Defaults to "area".
            conclusion (typing.Union[Conclusion, str], optional): The conclusion function. Defaults to "max".
            truncate (bool, optional): Whether to truncate or scale. Defaults to False.
        """
        super().__init__(
            polygon_set(left, middle, right), tunable
        )

    @classmethod
    def create(
        cls, left: torch.Tensor, right: torch.Tensor, middle: torch.Tensor=None, 
        tunable: bool=False, g: G=None
    ) -> 'IsoscelesTrapezoidFuzzifier':
        """Create an IsoscelesFuzzifier from shapes defined by tensors

        Args:
            left (torch.Tensor): The tensor for the left shape
            right (torch.Tensor): The tensor for the right shape
            middle (torch.Tensor, optional): The tensor for the middle shape(s). Defaults to None.
            tunable (bool, optional): Whether the shape parameters can be updated. Defaults to False.
            g (G, optional): The gradient estimator to use. Defaults to None.

        Returns:
            IsoscelesTrapezoidFuzzifier: The new fuzzifier
        """
        left, right, middle = validate_terms(left, right, middle)
        validate_n_points(middle, n_points=3)
        validate_right_trapezoid(left, right)
        left_shape = shape.RightTrapezoid if flat_edges(left, 2) else shape.RightTriangle
        right_shape = shape.RightTrapezoid if flat_edges(left, 2) else shape.RightTriangle
        left_shape = left_shape(Coords(left), False, g)
        right_shape = right_shape(Coords(right), True, g)
        if middle is not None:
            middle = shape.IsoscelesTrapezoid(Coords(middle), g)
        return IsoscelesTrapezoidFuzzifier(
            left, right, middle, tunable
        )

    @classmethod
    def from_coords(
        cls, coords: torch.Tensor, n_terms: int,
        flat_edges: bool=False, tunable: bool=False, g: G=None
    ):
        """Create the IsoscelesTrapezoidFuzzifier converter from coordinates

        Args:
            coords (torch.Tensor): The coords to use in creation
            n_terms (int): The number of terms for the shape
            flat_edges (bool, optional): Whether to use flat edges or not. Defaults to False.

        Returns:
            IsoscelesTrapezoidFuzzifier
        """
        middle = None
        if flat_edges:
            left = shape.RightTrapezoid(Coords(coords[:,:,None,:3]), False, g)
            if n_terms > 2:
                middle = shape.IsoscelesTrapezoid(Coords(stride_coordinates(coords[:,:,1:-1], n_terms - 2, 2, 3)), g)
            right = shape.RightTrapezoid(Coords(coords[:,:,None,-3:]), True, g)
        else:
            left = shape.RightTriangle(Coords(coords[:,:,None,:2]), False, g)
            if n_terms > 2:
                middle = shape.IsoscelesTrapezoid(Coords(stride_coordinates(coords, n_terms - 2, 2, 3)), g)
            right = shape.RightTriangle(Coords(coords[:,:,None,-2:]), True, g)

        return IsoscelesTrapezoidFuzzifier(left, right, middle, tunable)

    @classmethod
    def from_linspace(
        cls, n_terms: int, n_vars: int=None,
        flat_edges: bool=False, tunable: bool=False, g: G=None
    ) -> 'IsoscelesTrapezoidFuzzifier':
        """Create the IsoscelesTrapezoidFuzzifierconverter from coordinates

        Args:
            n_terms (int): The number of terms for the shape
            flat_edges (bool, optional): Whether to use flat edges or not. Defaults to False.

        Returns:
            IsoscelesTrapezoidFuzzifier
        """
        if flat_edges:
            coords = generate_spaced_params((n_terms - 2) * 2 + 4, in_features=n_vars)
        else:
            coords = generate_spaced_params((n_terms - 2) * 2 + 2, in_features=n_vars)
        return IsoscelesTrapezoidFuzzifier.from_coords(
            coords, n_terms, 
            flat_edges, tunable, g
        )


class TrapezoidFuzzifier(ShapeFuzzifier):

    def __init__(
        self, left: typing.Union[shape.RightTrapezoid, shape.RightTriangle],
        right: typing.Union[shape.RightTrapezoid, shape.RightTriangle],
        middle: shape.Trapezoid=None, tunable: bool=False
    ):
        """Create the TrapezoidFuzzifier

        Args:
            left (typing.Union[shape.RightTrapezoid, shape.RightTriangle]): The shape on the left
            right (typing.Union[shape.RightTrapezoid, shape.RightTriangle]): The shape on the right
            middle (shape.Trapezoid, optional): The middle shape. Defaults to None.
        """
        super().__init__(
            polygon_set(left, middle, right), tunable
        )
    
    @classmethod
    def create(
        cls, left: torch.Tensor, right: torch.Tensor, middle: torch.Tensor=None, 
        tunable: bool=False, g: G=None
    ) -> 'TrapezoidFuzzifier':
        """Create a TrapezoidFuzzifier from shapes defined by tensors

        Args:
            left (torch.Tensor): The tensor for the left shape
            right (torch.Tensor): The tensor for the right shape
            middle (torch.Tensor, optional): The tensor for the middle shape(s). Defaults to None.
            tunable (bool, optional): Whether the shape parameters can be updated. Defaults to False.
            g (G, optional): The gradient estimator to use. Defaults to None.

        Returns:
            TrapezoidFuzzifier: The new fuzzifier
        """
        left, right, middle = validate_terms(left, right, middle)
        validate_n_points(middle, n_points=4)
        validate_right_trapezoid(left, right)
        left_shape = shape.RightTrapezoid if flat_edges(left, 2) else shape.RightTriangle
        right_shape = shape.RightTrapezoid if flat_edges(left, 2) else shape.RightTriangle
        left_shape = left_shape(Coords(left), False, g)
        right_shape = right_shape(Coords(right), True, g)
        if middle is not None:
            middle = shape.Trapezoid(Coords(middle), g)
        return TrapezoidFuzzifier(
            left, right, middle, tunable
        )

    @classmethod
    def from_coords(
        cls, coords: torch.Tensor, n_terms: int,
        flat_edges: bool=False, tunable: bool=False, g: G=None
    ):
        """Create the TrapezoidFuzzifier converter from coordinates

        Args:
            coords (torch.Tensor): The coords to use in creation
            n_terms (int): The number of terms for the shape
            flat_edges (bool, optional): Whether to use flat edges or not. Defaults to False.

        Returns:
            TrapezoidFuzzifier
        """
        middle = None
        if flat_edges:
            left = shape.RightTrapezoid(Coords(coords[:,:,None,:3]), False, g)
            if n_terms > 2:
                middle = shape.Trapezoid(Coords(stride_coordinates(coords[:,:,1:-1], n_terms - 2, 2, 4)), g)
            right = shape.RightTrapezoid(Coords(coords[:,:,None,-3:]), True, g)
        else:
            left = shape.RightTriangle(Coords(coords[:,:,None,:2]), False, g)
            if n_terms > 2:
                middle = shape.Trapezoid(Coords(stride_coordinates(coords, n_terms - 2, 2, 4)), g)
            right = shape.RightTriangle(Coords(coords[:,:,None,-2:]), True, g)

        return TrapezoidFuzzifier(left, right, middle, tunable)

    @classmethod
    def from_linspace(
        cls, n_terms: int, n_vars: int=None,
        flat_edges: bool=False, tunable: bool=False, g: G=None
    ):
        """Create the TrapezoidFuzzifier converter from coordinates

        Args:
            n_terms (int): The number of terms for the shape
            flat_edges (bool, optional): Whether to use flat edges or not. Defaults to False.

        Returns:
            TrapezoidFuzzifier
        """
        if flat_edges:
            coords = generate_spaced_params((n_terms - 2) * 2 + 4, in_features=n_vars)
        else:
            coords = generate_spaced_params((n_terms - 2) * 2 + 2, in_features=n_vars)
        return TrapezoidFuzzifier.from_coords(
            coords, n_terms,
            flat_edges, tunable, g
        )


class TriangleFuzzifier(ShapeFuzzifier):

    def __init__(
        self, left: typing.Union[shape.RightTrapezoid, shape.RightTriangle],
        right: typing.Union[shape.RightTrapezoid, shape.RightTriangle],
        middle: shape.IsoscelesTriangle=None, tunable: bool=False
    ):
        """Create a TriangleFuzzifier

        Args:
            left (typing.Union[shape.RightTrapezoid, shape.RightTriangle]): The shape on the left
            right (typing.Union[shape.RightTrapezoid, shape.RightTriangle]): The shape on the right
            middle (shape.Triangle, optional): The middle shape. Defaults to None.
        """
        super().__init__(
            polygon_set(left, middle, right), tunable
        )
    
    @classmethod
    def create(
        cls, left: torch.Tensor, right: torch.Tensor, middle: torch.Tensor=None, 
        tunable: bool=False
    ) -> 'TriangleFuzzifier':
        """Create the TriangleFuzzifier from shapes defined by tensors

        Args:
            left (torch.Tensor): The tensor for the left shape
            right (torch.Tensor): The tensor for the right shape
            middle (torch.Tensor, optional): The tensor for the middle shape(s). Defaults to None.
            tunable (bool, optional): Whether the shape parameters can be updated. Defaults to False.

        Returns:
            TriangleFuzzifier: The new fuzzifier
        """
        left, right, middle = validate_terms(left, right, middle)
        validate_n_points(middle, n_points=3)
        validate_right_trapezoid(left, right)
        left_shape = shape.RightTrapezoid if flat_edges(left, 2) else shape.RightTriangle
        right_shape = shape.RightTrapezoid if flat_edges(left, 2) else shape.RightTriangle
        left_shape = left_shape(Coords(left), False)
        right_shape = right_shape(Coords(right), True)
        if middle is not None:
            middle = shape.Triangle(Coords(middle))
        return TriangleFuzzifier(
            left, right, middle, tunable
        )

    @classmethod
    def from_coords(
        cls, coords: torch.Tensor, n_terms: int,
        flat_edges: bool=False, tunable: bool=False, g: G=None
    ):
        """Create the TriangleFuzzifier converter from coordinates

        Args:
            coords (torch.Tensor): The coords to use in creation
            n_terms (int): The number of terms for the shape
            flat_edges (bool, optional): Whether to use flat edges or not. Defaults to False.

        Returns:
            TriangleFuzzifier
        """
        middle = None
        if flat_edges:
            left = shape.RightTrapezoid(Coords(coords[:,:,None,:3]), False, g)
            if n_terms > 2:
                middle = shape.Triangle(Coords(stride_coordinates(coords[:,:,1:-1], n_terms - 2, 1, 3)), g)
            right = shape.RightTrapezoid(Coords(coords[:,:,None,-3:]), True, g)

        else:
            left = shape.RightTriangle(Coords(coords[:,:,None,:2]), False, g)
            if n_terms > 2:
                middle = shape.Triangle(Coords(stride_coordinates(coords, n_terms - 2, 1, 3)), g)
            right = shape.RightTriangle(Coords(coords[:,:,None,-2:]), True, g)
        return TriangleFuzzifier(left, right, middle, tunable)

    @classmethod
    def from_linspace(
        cls, n_terms: int, n_vars: int=None,
        flat_edges: bool=False, tunable: bool=False, g: G=None
    ):
        """Create the TriangleFuzzifier converter from coordinates

        Args:
            n_terms (int): The number of terms for the shape
            flat_edges (bool, optional): Whether to use flat edges or not. Defaults to False.

        Returns:
            TriangleFuzzifier
        """
        if flat_edges:
            coords = generate_spaced_params(n_terms + 2, in_features=n_vars)
        else:
            coords = generate_spaced_params(n_terms, in_features=n_vars)
        return TriangleFuzzifier.from_coords(
            coords, n_terms,
            flat_edges, tunable, g
        )


class SquareFuzzifier(ShapeFuzzifier):
    """Create a SquareFuzzifier
    """

    def __init__(
        self, 
        square: shape.Square, tunable: bool=False
    ):
        """Create a fuzzy converter based on a 'square' function

        Args:
            square (Square), The set of squares to use.
        """
        super().__init__(
            square, tunable=tunable
        )

    @classmethod
    def create(
        cls, params: torch.Tensor, tunable: bool=False, g: G=None
    ) -> 'SquareFuzzifier':
        """Create a SquareFuzzifier from parameters defining the shape

        Args:
            params (torch.Tensor): The params for the square fuzzifier
            tunable (bool, optional): Whehter to update the shape parameters. Defaults to False.
            g (G, optional): The gradient estimator if used. Defaults to None.

        Returns:
            SquareFuzzifier: The square fuzzifier
        """
        params = validate_terms(params)
        
        validate_n_points(params, n_points=2)
        params = shape.Square(Coords(params), g=g)
        return SquareFuzzifier(
            params, tunable
        )

    @classmethod
    def from_coords(
        cls, coords: torch.Tensor, n_terms: int, tunable: bool=False, g: G=None
    ) -> 'SquareFuzzifier':
        """Create a SquareFuzzifier from coords

        Args:
            coords (torch.Tensor): The coords to create the fuzzifier from
            n_terms (int): The number of terms
            tunable (bool, optional): Whehter to update the shape parameters. Defaults to False.
            g (G, optional): The gradient estimator. Defaults to None.

        Returns:
            SquareFuzzifier: The SquareFuzzifier
        """
        square = shape.Square(Coords(stride_coordinates(coords[:,:,:], n_terms, 2, 2)), g=g)
        return SquareFuzzifier(square, tunable)

    @classmethod
    def from_linspace(
        cls, n_terms: int, n_vars: int=None, tunable: bool=False, g: G=None
    ) -> 'SquareFuzzifier':
        """Create a SquareFuzzifier from a linspace

        Args:
            n_terms (int): The number of terms
            n_vars (int, optional): The number of variables. Defaults to None.
            tunable (bool, optional): Whehter to update the shape parameters. Defaults to False.
            g (G, optional): The gradient estimator. Defaults to None.

        Returns:
            SquareFuzzifier: The SquareFuzzifier to create
        """
        coords = generate_spaced_params(n_terms + 1, n_features=n_vars)
        return SquareFuzzifier.from_coords(
            coords, n_terms, tunable, g
        )


class LogisticFuzzifier(ShapeFuzzifier):

    def __init__(
        self, left: shape.HalfLogisticBell,
        right: shape.HalfLogisticBell,
        middle: shape.LogisticBell=None, tunable: bool=False
    ):
        """Create a fuzzy converter based on the logistic distribution

        Args:
            left (shape.HalfLogisticBell): The left logistic
            right (shape.HalfLogisticBell): The right logistic
            middle (shape.LogisticBell, optional): The middle logistics . Defaults to None.
        """
        super().__init__(
            polygon_set(left, middle, right), tunable
        )
    
    @classmethod
    def create(
        cls, left_scales: torch.Tensor, left_biases: torch.Tensor, right_scales: torch.Tensor, right_biases: torch.Tensor, 
        middle_scales: torch.Tensor=None, middle_biases: torch.Tensor=None,
        tunable: bool=False
    ) -> 'LogisticFuzzifier':
        """Create a LogisticFuzzifier using shape paramteres

        Args:
            left_scales (torch.Tensor): The scale parameters for the left side
            left_biases (torch.Tensor): The bias parameters for the left side
            right_scales (torch.Tensor): The scale parameters for the right side
            right_biases (torch.Tensor): The bias parameters for the right side
            middle_scales (torch.Tensor, optional): The scale parameters for the middle. Defaults to None.
            middle_biases (torch.Tensor, optional): The bias parameters for the middle. Defaults to None.
            tunable (bool, optional): Whether the shape parameters can be updated. Defaults to False.

        Returns:
            LogisticFuzzifier: The LogisticFuzzifier
        """
        left, right, middle = validate_terms(left_scales, right_scales, left_scales, left_biases, middle_scales, middle_biases)
        validate_n_points(left, middle, right, n_points=1)
        left = shape.HalfLogisticBell(left_biases, left_scales, True)
        right = shape.HalfLogisticBell(right_biases, right_scales, False)

        if middle is not None:
            middle = shape.Logistic(middle_biases, middle_scales)
        return LogisticFuzzifier(
            left, right, middle, tunable
        )

    @classmethod
    def from_coords(
        cls, bias_coords: torch.Tensor, scale_coords: torch.Tensor, n_terms: int,
        tunable: bool=False
    ) -> 'LogisticFuzzifier':
        """Create a LogisticFuzzifier from coords

        Args:
            bias_coords (torch.Tensor): The coordinates for the bias parameters
            scale_coords (torch.Tensor): The coordinates for the scale parameters
            n_terms (int): The number of terms
            tunable (bool, optional): Whether to update the shape parameters. Defaults to False.

        Returns:
            LogisticFuzzifier: the logistic fuzzifier
        """
        middle = None
        left = shape.HalfLogisticBell(
            bias_coords[:,:,None,0], 
            scale_coords[:,:,None,0], 
            False
        )
        if n_terms > 2:
            middle = shape.LogisticBell(
                bias_coords[:,:,1:-1], scale_coords[:,:,1:-1]
            )
        right = shape.HalfLogisticBell(
            bias_coords[:,:,-1:], 
            scale_coords[:,:,-1:], True)

        return LogisticFuzzifier(left, right, middle, tunable)

    @classmethod
    def from_linspace(
        cls, n_terms: int, n_vars: int=None, tunable: bool=False
    ) -> 'LogisticFuzzifier':
        """Create a LogisticFuzzifier from a linspace

        Args:
            n_terms (int): The number of terms
            n_vars (int, optional): The number of vars. Defaults to None.
            tunable (bool, optional): Whether to update the shape parameters. Defaults to False.

        Returns:
            LogisticFuzzifier: the logistic fuzzifier
        """
        bias_coords = generate_spaced_params(n_terms, in_features=n_vars)
        width = 1.0 / 2 * (n_terms - 1.0)
        scale_coords = generate_repeat_params(n_terms, width, in_features=n_vars)
        return LogisticFuzzifier.from_coords(
            bias_coords, scale_coords, n_terms, tunable
        )


class SigmoidFuzzifier(ShapeFuzzifier):
    """SigmoidFuzzifier uses a series of sigmoids for fuzzification. 
    """

    def __init__(
        self, sigmoid: shape.Sigmoid=None, tunable: bool=False
    ):
        """Create a SigmoidFuzzifier

        Args:
            sigmoid (shape.Sigmoid, optional): The sigmoid to use. Defaults to None.
            tunable (bool, optional): Whether to update the shape parameters. Defaults to False.
        """
        super().__init__(
            [sigmoid], tunable
        )

    @classmethod
    def create(
        cls, biases: torch.Tensor, scales: typing.Union[torch.Tensor, float],
        tunable: bool=False
    ) -> 'SigmoidFuzzifier':
        """Create a SigmoidFuzzifier from biases and scales

        Args:
            biases (torch.Tensor): The biases for the sigmoid
            scales (typing.Union[torch.Tensor, float]): The scales for the sigmoid
            tunable (bool, optional): Whether the sigmoid parameters can be tuned. Defaults to False.

        Returns:
            SigmoidFuzzifier: The resulting sigmoid fuzzifier
        """
        biases = validate_terms(biases, scales)
        validate_n_points(biases, scales, n_points=1)
        if isinstance(scales, float):

            scales = generate_repeat_params(biases.n_terms, scales, biases.n_vars)
        sigmoid = shape.Sigmoid(biases, scales)
        return SigmoidFuzzifier(
            sigmoid, tunable
        )

    @classmethod
    def from_coords(
        cls, bias_coords: torch.Tensor, scale_coords: torch.Tensor,
        tunable: bool=False
    ):
        """Create a SigmoidFuzzifier from bias and scale coords

        Args:
            bias_coords (torch.Tensor): The biases for the sigmoid
            scale_coords (torch.Tensor): The scales for the sigmoid
            tunable (bool, optional): Whether the sigmoid parameters can be tuned. Defaults to False.

        Returns:
            SigmoidFuzzifier: The resulign SigmoidFuzzifier
        """
        sigmoid = shape.Sigmoid(
            bias_coords, scale_coords
        )
        return SigmoidFuzzifier(sigmoid, tunable)

    @classmethod
    def from_linspace(
        cls, n_terms: int, n_vars: int=None, tunable: bool=False
    ) -> 'SigmoidFuzzifier':
        """Create a SigmoidFuzzifier from 

        Args:
            n_terms (int): The number of terms
            n_vars (int, optional): The number of variables. Defaults to None.
            tunable (bool, optional): Whether the parameters are tunable. Defaults to False.

        Returns:
            SigmoidFuzzifier: The resulting SigmoidFuzzifier
        """
        bias_coords = generate_spaced_params(n_terms + 2, in_features=n_vars)[:,:,1:-1]
        width = 1.0 / (2 * (n_terms + 1))
        scale_coords = generate_repeat_params(n_terms, width, in_features=n_vars)

        return SigmoidFuzzifier.from_coords(
            bias_coords, scale_coords, tunable
        )

    def defuzzifier(
            self, 
            hypothesis: typing.Union[ShapeHypothesis, str]="min_core", 
            conclusion: typing.Union[Conclusion, str]="average",
            truncate: bool=False
    ) -> Defuzzifier:
        """Create a Defuzzifier for the Sigmoid

        Args:
            hypothesis (typing.Union[ShapeHypothesis, str], optional): The hypothesis to use. Defaults to "min_core".
            conclusion (typing.Union[Conclusion, str], optional): The conclusion to sue. Defaults to "average".
            truncate (bool, optional): Whether to truncate (True) or scale (False). Defaults to False.

        Returns:
            Defuzzifier: The resulting defuzzifier
        """
        
        return ShapeDefuzzifier(
            self._composite, hypothesis, conclusion, truncate
        )


class RampFuzzifier(ShapeFuzzifier):
    """A fuzzifier that makes use of a series of ramps
    """

    def __init__(
        self, ramp: shape.Ramp=None, tunable: bool=False
    ):
        """Create a fuzzy converter that makes use of the ramp function

        Args:
            ramp (shape.Ramp, optional): The ramp function. Defaults to None.
        """
        super().__init__(
            [ramp], True, tunable
        )

    @classmethod
    def create(
        cls, points: torch.Tensor, tunable: bool=False, g: G=None
    ):
        points = validate_terms(points)
        validate_n_points(points, 2)
        point_params = Coords(points)
        ramp = shape.Ramp(point_params, g)
        return RampFuzzifier(
            ramp, tunable
        )

    @classmethod
    def from_coords(
        cls, coords: torch.Tensor, n_terms: int, tunable: bool=False, g: G=None
    ) -> 'RampFuzzifier':
        """Create the converter from coordinates. Each ramp function uses two coordinates with a stride of two

        Args:
            coords (torch.Tensor): The coordinates defining the ramp function
            n_terms (int): The number of terms for the fuzzy converter

        Returns:
            RampFuzzifier: The resulting Fuzzifier using ramp functions
        """
        ramp = shape.Ramp(
            Coords(stride_coordinates(coords, n_terms, 1, 2, 2)), g
        )
        return RampFuzzifier(ramp, tunable)

    @classmethod
    def from_linspace(
        cls, n_terms: int, n_vars: int=None, tunable: bool=False, g: G=None
    ) -> 'RampFuzzifier':
        """Create the converter from a linspace with n terms

        Args:
            n_terms (int): The number of terms for the fuzzy converter

        Returns:
            RampFuzzifier The resulting Fuzzifier using ramp functions
        """
        coords = generate_spaced_params(n_terms + 2, in_features=n_vars)
        return RampFuzzifier.from_coords(
            coords, n_terms, tunable, g
        )

    def defuzzifier(
            self, 
            hypothesis: typing.Union[ShapeHypothesis, str]="min_core", 
            conclusion: typing.Union[Conclusion, str]="average",
            truncate: bool=False
    ) -> Defuzzifier:
        
        return ShapeDefuzzifier(
            self._composite, hypothesis, conclusion, truncate
        )
    
    @property
    def g(self) -> G:
        return self._composite[0].g
    
    @g.setter
    def g(self, g: G) -> G:

        self._composite[0].g = G
        return g


class StepFuzzifier(ShapeFuzzifier):
    """A 'fuzzifier' that makes use of a series of step functions
    """

    def __init__(
        self, step: shape.Step=None, tunable: bool=False
    ):
        """Create a 'fuzzifier' that uses the step function. 

        Args:
            step (shape.Step, optional): The step function to use. Defaults to None.
            tunable (bool): Whether to tune the parameters
        """
        
        super().__init__(
            [step], True, tunable
        )

    @classmethod
    def create(
        cls, points: torch.Tensor, tunable: bool=False
    ):
        # TODO: Add more validation code
        points = validate_terms(points)
        validate_n_points(points, 1)
        # point_params = Coords(points)
        step = shape.Step(points)
        return StepFuzzifier(step, tunable)

    @classmethod
    def from_coords(
        cls, coords: torch.Tensor, n_terms: int, tunable: bool=False
    ) -> 'StepFuzzifier':
        """Create the StepFuzzifier from coordinates

        Args:
            coords (torch.Tensor): A series of thresholds use for the step functions
            n_terms (int): The number of terms to use

        Returns:
            StepFuzzifier: the created StepFuzzifier
        """
        step = shape.Step(
            stride_coordinates(coords, n_terms, 1, 1, 1).squeeze(-1)
        )
        return StepFuzzifier(step, tunable)

    @classmethod
    def from_linspace(
        cls, n_terms: int, n_vars: int=None, tunable: bool=False
    ) -> 'StepFuzzifier':
        """Create the StepFuzzifier from coordinates

        Args:
            n_terms (int): The number of terms

        Returns:
            StepFuzzifier: the created StepFuzzifier
        """
        coords = generate_spaced_params(n_terms + 2, in_features=n_vars)[:,:,1:-1]
        
        return StepFuzzifier.from_coords(
            coords, n_terms, tunable
        )

    def defuzzifier(
            self, 
            hypothesis: typing.Union[ShapeHypothesis, str]="min_core", 
            conclusion: typing.Union[Conclusion, str]="average",
            truncate: bool=False
    ) -> Defuzzifier:
        
        return ShapeDefuzzifier(
            self._composite, hypothesis, conclusion, truncate
        )


class FuzzifierDecorator(Fuzzifier):
    """Define a decorator for the converter
    """

    def __init__(self, fuzzifier: Fuzzifier):
        """

        Args:
            fuzzifier (Fuzzifier): 
        """
        super().__init__(fuzzifier.n_terms, fuzzifier.n_vars)
        self._fuzzifier = fuzzifier

    @abstractmethod
    def decorate_fuzzify(self, x: torch.Tensor) -> torch.Tensor:
        """Decorate the fuzzification function

        Args:
            x (torch.Tensor): The crisp value to fuzzify

        Returns:
            torch.Tensor: The decorated crisp value
        """
        pass

    def fuzzify(self, x: torch.Tensor) -> torch.Tensor:
        """Fuzzify the message

        Args:
            x (torch.Tensor): The crisp value to fuzzify

        Returns:
            torch.Tensor: The resulting tensor
        """
        return self._fuzzifier.fuzzify(self.decorate_fuzzify(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuzzify(x)


class DefuzzifierDecorator(ABC, Defuzzifier):
    """Define a decorator for the converter
    """

    def __init__(self, defuzzifier: Defuzzifier):
        """

        Args:
            converter (Fuzzifier): 
        """
        super().__init__(defuzzifier.n_terms)
        self._defuzzifier = defuzzifier

    @abstractmethod
    def decorate_defuzzify(self, m: torch.Tensor) -> torch.Tensor:
        """Decorate the defuzzifier

        Args:
            m (torch.Tensor): The membership

        Returns:
            torch.Tensor: The decorated membership
        """
        pass

    def conclude(self, hypo_weight: HypoWeight) -> torch.Tensor:
        """Use the hyptoheses to determine the result

        Args:
            hypo_weight (HypoWeight): The hypotheses and  their weights

        Returns:
            torch.Tensor: The conclusion based on the hypotheses
        """
        return self._defuzzifier.conclude(hypo_weight)
    
    def hypo(self, m: torch.Tensor) -> HypoWeight:
        """

        Args:
            m (torch.Tensor): The membership value

        Returns:
            HypoWeight: The hypothesis and the weight
        """
        return self.decorate_defuzzify(self._defuzzifier.hypo(m))


class FuncFuzzifierDecorator(FuzzifierDecorator):

    def __init__(self, fuzzifier: Fuzzifier, fuzzify: typing.Callable[[torch.Tensor], torch.Tensor], defuzzify: typing.Callable[[torch.Tensor], torch.Tensor]):
        """Use functions to decorate the fuzzification and defuzzification functions

        Args:
            fuzzifier (Fuzzifier): The converter to decorate
            fuzzify (typing.Callable[[torch.Tensor], torch.Tensor]): The fuzzification function
        """
        super().__init__(fuzzifier)
        self._fuzzify = fuzzify

    def decorate_fuzzify(self, x: torch.Tensor) -> torch.Tensor:

        """Decorate the defuzzifier

        Args:
            m (torch.Tensor): The membership

        Returns:
            torch.Tensor: The fuzzified value
        """
        return self._fuzzify(x)


class FuncDefuzzifierDecorator(DefuzzifierDecorator):

    def __init__(self, defuzzifier: Defuzzifier, defuzzify: typing.Callable[[torch.Tensor], torch.Tensor]):
        """Use functions to decorate the fuzzification and defuzzification functions

        Args:
            defuzzifier (Defuzzifier): The converter to decorate
            defuzzify (typing.Callable[[torch.Tensor], torch.Tensor]): The defuzzification function
        """
        super().__init__(defuzzifier)
        self._defuzzify = defuzzify

    def decorate_defuzzify(self, m: torch.Tensor) -> torch.Tensor:

        """Decorate the defuzzifier

        Args:
            m (torch.Tensor): The membership

        Returns:
            torch.Tensor: The fuzzified value
        """
        return self._defuzzify(m)
