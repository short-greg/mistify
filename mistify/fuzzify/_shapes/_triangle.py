# 3rd party
import torch

# local
from ._base import Polygon
from ._trapezoid import IsoscelesTrapezoid, IncreasingRightTrapezoid, DecreasingRightTrapezoid, Trapezoid
from ._utils import calc_m_linear_increasing, calc_m_linear_decreasing, calc_x_linear_decreasing, calc_x_linear_increasing
from ...utils import unsqueeze

intersect = torch.min
union = torch.max


class IncreasingRightTriangle(Polygon):
    """A right triangle with an increasing slope
    """

    PT = 2

    def join(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the membership of x with the triangle.

        Args:
            x (torch.Tensor): The value to calculate the membership for

        Returns:
            torch.Tensor: The membership
        """
        params = self._params()
        return calc_m_linear_increasing(
            unsqueeze(x), params.pt(0), params.pt(1), self._m
        )

    def _calc_areas(self):
        """
        Returns:
            torch.Tensor: The area of the right triangle
        """
        params = self._params()
        return self._resize_to_m(
            0.5 * (params.pt(1)
            - params.pt(0)) * self._m, self._m
        )

    def _calc_mean_cores(self):
        """
        Returns:
            torch.Tensor: the value of the height of the triangle
        """
        params = self._params()
        return self._resize_to_m(params.pt(1), self._m)

    def _calc_centroids(self):
        """
        Returns:
            torch.Tensor: The center of mass for the right triangle
        """
        p1, p2 = 1 / 3, 2 / 3
        params = self._params()

        return self._resize_to_m(
            p1 * params.pt(0) + p2 * params.pt(1), self._m
        )
    
    def scale(self, m: torch.Tensor) -> 'IncreasingRightTriangle':
        """Update the vertical scale of the right triangle

        Args:
            m (torch.Tensor): The new vertical scale

        Returns:
            IncreasingRightTriangle: The updated vertical scale if the scale is greater
        """
        updated_m = intersect(m, self._m)
        
        params = self._params()
        
        return IncreasingRightTriangle(
            params, updated_m
        )

    def truncate(self, m: torch.Tensor) -> 'IncreasingRightTrapezoid':
        """Truncate the right triangle into a trapezoid. This requires the points be recalculated

        Args:
            m (torch.Tensor): The new maximum value

        Returns:
            IncreasingRightTrapezoid: The triangle truncated into a trapezoid
        """
        updated_m = intersect(self._m, m)
        params = self._params()

        pt = calc_x_linear_increasing(
            updated_m, params.pt(0), params.pt(1), self._m
        )
        params = params.insert(
            pt, 1, to_unsqueeze=True, equalize_to=updated_m
        )
        return IncreasingRightTrapezoid(
            params, updated_m
        )


class DecreasingRightTriangle(Polygon):
    """A right triangle with a decreasing slope
    """

    PT = 2
    
    def join(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the membership of x with the triangle.

        Args:
            x (torch.Tensor): The value to calculate the membership for

        Returns:
            torch.Tensor: The membership
        """
        params = self._params()
        return calc_m_linear_decreasing(
            unsqueeze(x), params.pt(0), params.pt(1), self._m
        )

    def _calc_areas(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: The area of the right triangle
        """
        params = self._params()
        return self._resize_to_m((
            0.5 * (params.pt(1)
            - params.pt(0)) * self._m
        ), self._m)

    def _calc_mean_cores(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: the value of the height of the triangle
        """
        params = self._params()
        return self._resize_to_m(params.pt(0), self._m)

    def _calc_centroids(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: The center of mass for the right triangle
        """
        params = self._params()
        return self._resize_to_m(
            2 / 3 * params.pt(0) 
            + 1 / 3 * params.pt(1), self._m
        )
    
    def scale(self, m: torch.Tensor) -> 'DecreasingRightTriangle':
        """Update the vertical scale of the right triangle

        Args:
            m (torch.Tensor): The new vertical scale

        Returns:
            DecreasingRightTriangle: The updated vertical scale if the scale is greater
        """
        updated_m = intersect(self._m, m)
        params = self._params()
        
        return DecreasingRightTriangle(
            params, updated_m
        )

    def truncate(self, m: torch.Tensor) -> 'DecreasingRightTrapezoid':
        """
        Args:
            m (torch.Tensor): The new maximum value

        Returns:
            DecreasingRightTrapezoid: The triangle truncated into a RightTrapezoid
        """
        updated_m = intersect(self._m, m)
        params = self._params()

        pt = calc_x_linear_decreasing(
            updated_m, params.pt(0), params.pt(1), self._m
        )

        params = params.insert(pt, 1, to_unsqueeze=True, equalize_to=updated_m)
        return DecreasingRightTrapezoid(
            params, updated_m
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
        m1 = calc_m_linear_increasing(
            unsqueeze(x), params.pt(0), params.pt(1), self._m
        )
        m2 = calc_m_linear_decreasing(
            unsqueeze(x), params.pt(1), params.pt(2), self._m
        )
        return union(m1, m2)

    def _calc_areas(self):
        """
        Returns:
            torch.Tensor: The area of triangle
        """
        params = self._params()
        return self._resize_to_m((
            0.5 * (params.pt(2) 
            - params.pt(0)) * self._m
        ), self._m)

    def _calc_mean_cores(self):
        """
        Returns:
            torch.Tensor: the maximum value of the triangle
        """
        params = self._params()
        return self._resize_to_m(params.pt(1), self._m)

    def _calc_centroids(self):
        """
        Returns:
            torch.Tensor: the center of mass for the triangle
        """
        params = self._params()
        return self._resize_to_m(1 / 3 * (
            params.pt(0) + params.pt(1) + params.pt(2)
        ), self._m)
    
    def scale(self, m: torch.Tensor) -> 'Triangle':
        """Update the vertical scale of the triangle

        Args:
            m (torch.Tensor): The new vertical scale

        Returns:
            Triangle: The updated vertical scale if the scale is greater
        """
        updated_m = intersect(self._m, m)        
        return Triangle(
            self._params, updated_m
        )

    def truncate(self, m: torch.Tensor) -> 'Trapezoid':
        """
        Args:
            m (torch.Tensor): The new maximum value

        Returns:
            Trapezoid: The triangle truncated into a Trapezoid
        """
        updated_m = intersect(self._m, m)

        params = self._params()
        pt1 = calc_x_linear_increasing(updated_m, params.pt(0), params.pt(1), self._m)
        pt2 = calc_x_linear_decreasing(updated_m, params.pt(1), params.pt(2), self._m)
        to_replace = torch.cat(
            [pt1.unsqueeze(3), pt2.unsqueeze(3)], dim=3
        )
        params= params.replace(
            to_replace, 1, False, equalize_to=updated_m
        )

        return Trapezoid(
            params, updated_m
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
        left_m = calc_m_linear_increasing(
            unsqueeze(x), params.pt(0), params.pt(1), self._m
        )
        right_m = calc_m_linear_decreasing(
            unsqueeze(x), params.pt(1), 
            params.pt(1) + (params.pt(1) - params.pt(0)), 
            self._m
        )
        return union(left_m, right_m)

    def _calc_areas(self):
        """
        Returns:
            torch.Tensor: the area of the triangle
        """
        return self._resize_to_m(
            0.5 * (self._params.pt(0)
            - self._params.pt(1)) * self._m, self._m
        )

    def _calc_mean_cores(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: the maximum value for the triangle
        """
        return self._resize_to_m(self._params.pt(1), self._m)

    def _calc_centroids(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: the center of mass for the triangle
        """
        return self._resize_to_m(self._params.pt(1), self._m)

    def scale(self, m: torch.Tensor) -> 'IsoscelesTriangle':
        """Update the vertical scale of the triangle

        Args:
            m (torch.Tensor): The new vertical scale

        Returns:
            IsoscelesTriangle: The updated vertical scale if the scale is greater
        """
        updated_m = intersect(self._m, m)
        return IsoscelesTriangle(
            self._params, updated_m
        )

    def truncate(self, m: torch.Tensor) -> 'IsoscelesTrapezoid':
        """
        Args:
            m (torch.Tensor): The new maximum value

        Returns:
            IsoscelesTrapezoid: The triangle truncated into an IsoscelesTrapezoid
        """
        updated_m = intersect(self._m, m)
        pt1 = calc_x_linear_increasing(updated_m, self._params.pt(0), self._params.pt(1), self._m)
        pt2 = calc_x_linear_decreasing(
            updated_m, self._params.pt(1), self._params.pt(1) + self._params.pt(1) - self._params.pt(0), self._m)

        to_replace = torch.cat(
            [pt1.unsqueeze(3), pt2.unsqueeze(3)], dim=3
        )

        params = self._params.replace(
            to_replace, 1, False, updated_m
        )
        return IsoscelesTrapezoid(
            params, updated_m
        )
