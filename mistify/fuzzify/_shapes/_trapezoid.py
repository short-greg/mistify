from ._base import Polygon, ShapeParams
from ...utils import unsqueeze
import torch
from ._utils import (
    calc_m_flat, calc_m_linear_decreasing, calc_m_linear_increasing,
    calc_x_linear_decreasing,
    calc_x_linear_increasing
)

intersect = torch.min


class Trapezoid(Polygon):
    """A general trapezoid consisting of four points
    """

    PT = 4

    def join(self, x: torch.Tensor) -> torch.Tensor:
        """Join calculates the membership value for each section of trapezoid and uses the maximimum value
        as the value

        Args:
            x (torch.Tensor): The value to calculate the membership for

        Returns:
            torch.Tensor: The membership
        """
        x = unsqueeze(x)
        m1 = calc_m_linear_increasing(x, self._params.pt(0), self._params.pt(1), self._m)
        m2 = calc_m_flat(x, self._params.pt(1), self._params.pt(2), self._m)
        m3 = calc_m_linear_decreasing(x, self._params.pt(2), self._params.pt(3), self._m)

        return torch.max(torch.max(m1, m2), m3)

    def _calc_areas(self) -> torch.Tensor:
        """Calculates the area of each section and sums it up

        Returns:
            torch.Tensor: The area of the trapezoid
        """
        
        return self._resize_to_m((
            0.5 * (self._params.pt(2) 
            - self._params.pt(0)) * self._m
        ), self._m)

    def _calc_mean_cores(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: the mean value of the top of the Trapezoid
        """
        return self._resize_to_m(
            0.5 * (self._params.pt(1) + self._params.pt(2)), self._m
        )

    def _calc_centroids(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: The center of mass for the three sections of the trapezoid
        """
        d1 = 0.5 * (self._params.pt(1) - self._params.pt(0))
        d2 = self._params.pt(2) - self._params.pt(1)
        d3 = 0.5 * (self._params.pt(3) - self._params.pt(2))

        return self._resize_to_m((
            d1 * (2 / 3 * self._params.pt(1) + 1 / 3 * self._params.pt(0)) +
            d2 * (1 / 2 * self._params.pt(2) + 1 / 2 *  self._params.pt(1)) + 
            d3 * (1 / 3 * self._params.pt(3) + 2 / 3 * self._params.pt(2))
        ) / (d1 + d2 + d3), self._m)

    def scale(self, m: torch.Tensor) -> 'Trapezoid':
        """Update the vertical scale of the trapezoid

        Args:
            m (torch.Tensor): The new vertical scale

        Returns:
            Trapezoid: The updated vertical scale if the scale is greater
        """
        updated_m = intersect(self._m, m)
        return Trapezoid(
            self._params, updated_m
        )

    def truncate(self, m: torch.Tensor) -> 'Trapezoid':
        """Truncate the Trapezoid. This requires the points be recalculated

        Args:
            m (torch.Tensor): The new maximum value

        Returns:
            Trapezoid: The updated vertical scale if the scale is greater
        """
        updated_m = intersect(self._m, m)

        # update the 
        left_x = calc_x_linear_increasing(
            updated_m, self._params.pt(0), self._params.pt(1), self._m
        )

        right_x = calc_x_linear_decreasing(
            updated_m, self._params.pt(2), self._params.pt(3), self._m
        )

        params = self._params.replace(left_x, 1, to_unsqueeze=True, equalize_to=updated_m)
        params = params.replace(right_x, 2, to_unsqueeze=True)

        return Trapezoid(
            params, updated_m, 
        )


class IsoscelesTrapezoid(Polygon):

    PT = 3

    def join(self, x: torch.Tensor) -> 'torch.Tensor':
        """Calculates the membership value for each part of the isosceles
        trapezoid and takes the maximum

        Args:
            x (torch.Tensor): The value to get the membership value for

        Returns:
            torch.Tensor: The membership value of x
        """
        x = unsqueeze(x)
        left_m = calc_m_linear_increasing(
            x, self._params.pt(0), self._params.pt(1), self._m
        )
        middle = calc_m_flat(x, self._params.pt(1), self._params.pt(2), self._m)
        pt3 = self._params.pt(1) - self._params.pt(0) + self._params.pt(2)
        right_m = calc_m_linear_decreasing(
            x, self._params.pt(2), pt3, self._m
        )
        return torch.max(torch.max(left_m, middle), right_m)
    
    @property
    def a(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: The distance for the bottom of the trapezoid 
        """
        return (
            self._params.pt(2) - self._params.pt(0) + 
            self._params.pt(1) - self._params.pt(0)
        )

    @property
    def b(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: The distance for the top of the trapezoid 
        """
        return self._params.pt(2) - self._params.pt(1)

    def _calc_areas(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: The area of the trapezoid
        """
        
        return self._resize_to_m(
            0.5 * (self.a + self.b) * self._m, self._m
        )

    def _calc_mean_cores(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: The mean value of the top of the trapezoid
        """
        return self._resize_to_m(0.5 * (self._params.pt(2) + self._params.pt(1)), self._m)

    def _calc_centroids(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: The mean value of the top of the trapezoid
        """
        return self.mean_cores

    def scale(self, m: torch.Tensor) -> 'IsoscelesTrapezoid':
        """Updates the maximum value of the trapezoid

        Args:
            m (torch.Tensor): The value to scale the trapezoid by

        Returns:
            IsoscelesTrapezoid: The updated trapezoid
        """
        updated_m = intersect(self._m, m)
        return IsoscelesTrapezoid(self._params, updated_m)

    def truncate(self, m: torch.Tensor) -> 'IsoscelesTrapezoid':
        """Truncates the maximum value of the trapezoid. 
        It also updates the width of the top of the trapezoid

        Args:
            m (torch.Tensor): The value to scale the trapezoid by

        Returns:
            IsoscelesTrapezoid: The updated trapezoid
        """
        updated_m = intersect(self._m, m)

        left_x = calc_x_linear_increasing(
            updated_m, self._params.pt(0), self._params.pt(1), self._m
        )

        right_x = self._params.pt(2) + self._params.pt(1) - left_x

        params = self._params.replace(
            left_x, 1, True, updated_m
        )
        params = params.replace(
            right_x, 2, True
        )
        return IsoscelesTrapezoid(params, updated_m)


class IncreasingRightTrapezoid(Polygon):

    PT = 3

    def join(self, x: torch.Tensor) -> 'torch.Tensor':
        m = calc_m_linear_increasing(
            unsqueeze(x), self._params.pt(0), self._params.pt(1), self._m
        )
        m2 = calc_m_flat(unsqueeze(x), self._params.pt(1), self._params.pt(2), self._m)

        return torch.max(m, m2)
    
    @property
    def a(self):
        return (
            self._params.pt(2) - self._params.pt(0)
        )

    @property
    def b(self):
        return self._params.pt(2) - self._params.pt(1)

    def _calc_areas(self):
        
        return self._resize_to_m(
            0.5 * (self.a + self.b) * self._m, self._m
        )

    def _calc_mean_cores(self):
        return self._resize_to_m(
            0.5 * (self._params.pt(2) + self._params.pt(1)), self._m
        )

    def _calc_centroids(self):
        
        d1 = 0.5 * (self._params.pt(1) - self._params.pt(0))
        d2 = self._params.pt(2) - self._params.pt(1)

        return self._resize_to_m((
            d1 * (2 / 3 * self._params.pt(1) + 1 / 3 * self._params.pt(0)) +
            d2 * (1 / 2 * self._params.pt(2) + 1 / 2 * self._params.pt(1))
        ) / (d1 + d2), self._m)

    def scale(self, m: torch.Tensor) -> 'IncreasingRightTrapezoid':
        return IncreasingRightTrapezoid(self._params, intersect(m, self._m))

    def truncate(self, m: torch.Tensor) -> 'IncreasingRightTrapezoid':
        updated_m = intersect(m, self._m)
        
        x = calc_x_linear_increasing(
            updated_m, self._params.pt(0), self._params.pt(1), self._m
        )
        params = self._params.replace(x, 1, True, updated_m)
        return IncreasingRightTrapezoid(params, updated_m)


class DecreasingRightTrapezoid(Polygon):

    PT = 3

    def join(self, x: torch.Tensor) -> 'torch.Tensor':
        """Calculates the membership value for each part of the isosceles
        trapezoid and takes the maximum

        Args:
            x (torch.Tensor): The value to get the membership value for

        Returns:
            torch.Tensor: The membership value of x
        """
        # want to allow parameters to be trained
        # params = self._params.sorted(decreasing=False)
        # change the properties
        # i can also set them to "frozen"
        # the parameters must also be "registered" if tunable
        
        m = calc_m_linear_decreasing(
            unsqueeze(x), self._params.pt(0), self._params.pt(1), self._m
        )
        m2 = calc_m_flat(unsqueeze(x), self._params.pt(1), self._params.pt(2), self._m)

        return torch.max(m, m2)
    
    @property
    def a(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: The width of the bottom of the trapezoid
        """
        return (
            self._params.pt(2) - self._params.pt(0)
        )

    @property
    def b(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: The width of the top of the trapezoid
        """
        return self._params.pt(1) - self._params.pt(0)

    def _calc_areas(self):
        
        return self._resize_to_m((
            0.5 * (self.a + self.b) * self._m
        ), self._m)

    def _calc_mean_cores(self):
        return self._resize_to_m(
            0.5 * (self._params.pt(0) + self._params.pt(1)), self._m
        )

    def _calc_centroids(self):
        d1 = self._params.pt(1) - self._params.pt(0)
        d2 = 0.5 * (self._params.pt(2) - self._params.pt(1))
        
        return self._resize_to_m((
            d1 * (1 / 2 * self._params.pt(1) + 1 / 2 * self._params.pt(0)) +
            d2 * (1 / 3 * self._params.pt(2) + 2 / 3 * self._params.pt(1))
        ) / (d1 + d2), self._m)

    def scale(self, m: torch.Tensor) -> 'DecreasingRightTrapezoid':
        """Update the height of the trapezoid

        Args:
            m (torch.Tensor): The value to scale by

        Returns:
            DecreasingRightTrapezoid: The updated trapezoid
        """
        return DecreasingRightTrapezoid(self._params, intersect(m, self._m))

    def truncate(self, m: torch.Tensor) -> 'DecreasingRightTrapezoid':
        """Update the height of the trapezoid. This also updates the first point

        Args:
            m (torch.Tensor): The value to truncate by

        Returns:
            DecreasingRightTrapezoid: The updated trapezoid
        """
        updated_m = intersect(m, self._m)
        
        x = calc_x_linear_decreasing(
            updated_m, self._params.pt(0), self._params.pt(1), self._m
        )
        params = self._params.replace(x, 1, True, updated_m)
        return DecreasingRightTrapezoid(params, updated_m)
