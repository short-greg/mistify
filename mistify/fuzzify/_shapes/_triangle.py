# 3rd party
import torch

# local
from ._base import Polygon, ShapeParams
from ._utils import calc_m_linear_increasing, calc_m_linear_decreasing, calc_x_linear_decreasing, calc_x_linear_increasing
from ...utils import unsqueeze
from ._trapezoid import trapezoid_centroid, trapezoid_area, trapezoid_mean_core
from ... import _functional as functional


def triangle_area(base1: torch.Tensor, base2: torch.Tensor, height: torch.Tensor) -> torch.Tensor:
    """
    Args:
        base1 (torch.Tensor): The first point at the base of the triangle
        base2 (torch.Tensor): The second point at the base of the triangle
        height (torch.Tensor): The height of the triangle

    Returns:
        torch.Tensor: The area of the triangle
    """
    return (base2 - base1) * height / 2.0


def triangle_centroid(x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x1 (torch.Tensor): The first point of the triangle
        x2 (torch.Tensor): The second point of the triangle
        x3 (torch.Tensor): The third point of the triangle

    Returns:
        torch.Tensor: The centroid of the triangle
    """
    return (x1 + x2 + x3) / 3.0


def triangle_right_centroid(x1: torch.Tensor, x2: torch.Tensor, increasing: bool=True) -> torch.Tensor:
    """
    Args:
        x1 (torch.Tensor): The first point
        x2 (torch.Tensor): The second point
        increasing (bool, optional): whether it is increasing or decreasing. Defaults to True.

    Returns:
        torch.Tensor: The centroid
    """
    if increasing:
        return 1. / 3. * x1 + 2. / 3. * x2
    return 2. / 3. * x1 + 1. / 3. * x2


class RightTriangle(Polygon):
    """A right triangle with an increasing slope
    """

    PT = 2

    def __init__(self, params: ShapeParams, increasing: bool=True):
        super().__init__(params)
        self.increasing = increasing

    def truncate(self, m: torch.Tensor) -> ShapeParams:

        params = self._params()

        if self.increasing:
            new_pt = params.pt(0) * m - params.pt(1) * (1 - m)
        else:
            new_pt = params.pt(0) * (1 - m) - params.pt(1) * m

        params = params.insert(new_pt, 1)
        return params

    def join(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the membership of x with the triangle.

        Args:
            x (torch.Tensor): The value to calculate the membership for

        Returns:
            torch.Tensor: The membership
        """
        params = self._params()
        x = unsqueeze(x)
        return functional.shape.right_triangle(
            x, params.pt(0), params.pt(1), self.increasing
        )

    def truncate(self, m: torch.Tensor) -> ShapeParams:
        
        params = self._params()

        if self.increasing:
            new_pt = params.pt(0) * (1 - m) + params.pt(1) * m
        else:
            new_pt = params.pt(0) * m + params.pt(1) * (1 - m)

        params = params.insert(new_pt, 1, to_unsqueeze=True)
        return params

    def areas(self, m: torch.Tensor, truncate: bool = False) -> torch.Tensor:

        if truncate:
            params = self.truncate(m)
            if self.increasing:
                a = params.pt(2) - params.pt(1)
            else:
                a = params.pt(1) - params.pt(0)
            b = params.pt(2) - params.pt(0)
            return trapezoid_area(a, b, m)
        else:
            params = self._params()
            
        return triangle_area(params.pt(0), params.pt(1), m)
    
    def mean_cores(self, m: torch.Tensor, truncate: bool = False) -> torch.Tensor:
        
        if truncate:
            params = self.truncate(m)
            if self.increasing:
                return self._resize_to_m(
                    trapezoid_mean_core(params.pt(1), params.pt(2)), m
                )
            return self._resize_to_m(
                trapezoid_mean_core(params.pt(0), params.pt(1)), m
            )

        params = self._params()
        if self.increasing:
            return self._resize_to_m(params.pt(1), m)
        return self._resize_to_m(params.pt(0), m)

    def centroids(self, m: torch.Tensor, truncate: bool = False) -> torch.Tensor:
        # x = (b+2a)/(3(a+b))h
        if truncate:
            params = self.truncate(m)
            if self.increasing:
                a = params.pt(2) - params.pt(1)
            else:
                a = params.pt(1) - params.pt(0)
            b = params.pt(2) - params.pt(0)
            return trapezoid_centroid(a, b, m)
        params = self._params()
        
        return self._resize_to_m(
            triangle_right_centroid(params.pt(0), params.pt(1), self.increasing), m
        )


class Triangle(Polygon):

    PT = 3

    def join(self, x: torch.Tensor) -> torch.Tensor:
        """Join calculates the membership value for each section of triangle and uses the maximimum value
        as the value

        Args:
            x (torch.Tensor): The value to calculate the membership for

        Returns:
            torch.Tensor: The membership
        """
        params = self._params()
        x = unsqueeze(x)
        return functional.shape.triangle(
            x, params.pt(0), params.pt(1), params.pt(2)
        )

    def truncate(self, m: torch.Tensor) -> ShapeParams:

        params = self._params()

        new_pt1 = params.pt(0) * (1 - m) - params.pt(1) * m
        new_pt2 = params.pt(1) * (m) - params.pt(2) * (1 - m)
        params = params.insert(new_pt1, 1, to_unsqueeze=True)
        params = params.replace(new_pt2, 2, to_unsqueeze=True)
        return params

    def areas(self, m: torch.Tensor, truncate: bool = False) -> torch.Tensor:

        if truncate:
            params = self.truncate(m)
            a = params.pt(2) - params.pt(1)
            b = params.pt(3) - params.pt(0)
            return self._resize_to_m(
                trapezoid_area(a, b, m), m
            )
        params = self._params()

        return self._resize_to_m(
            triangle_area(params.pt(0), params.pt(2), m), m
        )
    
    def mean_cores(self, m: torch.Tensor, truncate: bool = False) -> torch.Tensor:
        
        if truncate:
            params = self.truncate(m)
            return self._resize_to_m(
                trapezoid_mean_core(params.pt(1), params.pt(2)), m
            )
        return self._resize_to_m(
            self._params().pt(1), m
        )

    def centroids(self, m: torch.Tensor, truncate: bool = False) -> torch.Tensor:
        # x = (b+2a)/(3(a+b))h
        if truncate:
            params = self.truncate(m)
            a = params.pt(2) - params.pt(1)
            b = params.pt(3) - params.pt(0)
            return self._resize_to_m(
                trapezoid_centroid(a, b, m), m
            )
        params = self._params()

        return self._resize_to_m(
            triangle_centroid(params.pt(0), params.pt(1), params.pt(2)), m
        )


class IsoscelesTriangle(Polygon):

    PT = 2

    def join(self, x: torch.Tensor) -> torch.Tensor:
        """Calculates the membership value for each part of the isosceles
        triangle and takes the maximum

        Args:
            x (torch.Tensor): The value to get the membership value for

        Returns:
            torch.Tensor: The membership value of x
        """
        params = self._params()
        x = unsqueeze(x)
        return functional.shape.isosceles(
            x, params.pt(0), params.pt(1)
        )

    def truncate(self, m: torch.Tensor) -> ShapeParams:

        params = self._params()
        new_pt1 = (params.pt(1) * (1 - m) - params.pt(0)) * m
        new_pt2 = 2 * params.pt(1) - new_pt1
        params = params.replace(new_pt1, 1, to_unsqueeze=True)
        params = params.insert(new_pt2, 2, to_unsqueeze=True)
        return params

    def areas(self, m: torch.Tensor, truncate: bool = False) -> torch.Tensor:

        if truncate:
            params = self.truncate(m)
            a = params.pt(2) - params.pt(1)
            b = 2 * params.pt(2) - params.pt(1) - params.pt(0)
            return trapezoid_area(a, b, m)

        params = self._params()
        return triangle_area(
            params.pt(0), params.pt(1), m
        )

    def mean_cores(self, m: torch.Tensor, truncate: bool = False) -> torch.Tensor:
        
        if truncate:
            params = self.truncate(m)
            
            return self._resize_to_m(
                0.5 * (params.pt(2) + params.pt(1)), m)

        params = self._params()
        return self._resize_to_m(
            (params.pt(1) + params.pt(0)) * 0.5, m
        )

    def centroids(self, m: torch.Tensor, truncate: bool = False) -> torch.Tensor:
        # x = (b+2a)/(3(a+b))h

        if truncate:
            params = self.truncate(m)
            
            return self._resize_to_m(
                (params.pt(2) + params.pt(1)) * 0.5
            )
        params = self._params()
        
        return self._resize_to_m(
            params.pt(1), m
        )
