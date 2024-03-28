# 3rd party
import torch

# local
from ._base import Polygon, ShapeParams
from ._utils import calc_m_linear_increasing, calc_m_linear_decreasing, calc_x_linear_decreasing, calc_x_linear_increasing
from ...utils import unsqueeze
from ._trapezoid import trapezoid_centroid, trapezoid_area, trapezoid_mean_core
from ... import _functional as functional


def triangle_area(base1: torch.Tensor, base2: torch.Tensor, height: torch.Tensor) -> torch.Tensor:

    return (base2 - base1) * height / 2.0


def triangle_centroid(x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:

    return (x1 + x2 + x3) / 3.0


def triangle_right_centroid(x1: torch.Tensor, x2: torch.Tensor, increasing: bool=True) -> torch.Tensor:

    if increasing:
        return x1 / 3. + 2 * x2 / 3.0
    return 2 * x1 / 3.0 + x2 / 3.0


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
            new_pt = params.pt(0) * m - self.params.pt(1) * (1 - m)
        else:
            new_pt = params.pt(0) * (1 - m) - self.params.pt(1) * m

        params = self.params.insert(new_pt, 1)
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

        params = self.params.insert(new_pt, 1)
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
            params = self.params
            
        return triangle_area(params.pt(0), params.pt(1), m)
    
    def mean_cores(self, m: torch.Tensor, truncate: bool = False) -> torch.Tensor:
        
        if truncate:
            params = self.truncate(m)
            if self.increasing:
                return trapezoid_mean_core(params.pt(1), params.pt(2))
            return trapezoid_mean_core(params.pt(0), params.pt(1))

        if self.increasing:
            return params.pt(1)
        return params.pt(0)

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
        
        return triangle_right_centroid(params.pt(0), params.pt(1), self.increasing)


    # def _calc_areas(self):
    #     """
    #     Returns:
    #         torch.Tensor: The area of the right triangle
    #     """
    #     params = self._params()
    #     return self._resize_to_m(
    #         0.5 * (params.pt(1)
    #         - params.pt(0)) * self._m, self._m
    #     )

    # def _calc_mean_cores(self):
    #     """
    #     Returns:
    #         torch.Tensor: the value of the height of the triangle
    #     """
    #     params = self._params()
    #     return self._resize_to_m(params.pt(1), self._m)

    # def _calc_centroids(self):
    #     """
    #     Returns:
    #         torch.Tensor: The center of mass for the right triangle
    #     """
    #     p1, p2 = 1 / 3, 2 / 3
    #     params = self._params()

    #     return self._resize_to_m(
    #         p1 * params.pt(0) + p2 * params.pt(1), self._m
    #     )
    
    # def scale(self, m: torch.Tensor) -> 'IncreasingRightTriangle':
    #     """Update the vertical scale of the right triangle

    #     Args:
    #         m (torch.Tensor): The new vertical scale

    #     Returns:
    #         IncreasingRightTriangle: The updated vertical scale if the scale is greater
    #     """
    #     updated_m = functional.inter(m, self._m)
        
    #     params = self._params()
        
    #     return IncreasingRightTriangle(
    #         params, updated_m
    #     )

    # def truncate(self, m: torch.Tensor) -> 'IncreasingRightTrapezoid':
    #     """Truncate the right triangle into a trapezoid. This requires the points be recalculated

    #     Args:
    #         m (torch.Tensor): The new maximum value

    #     Returns:
    #         IncreasingRightTrapezoid: The triangle truncated into a trapezoid
    #     """
    #     updated_m = functional.inter(self._m, m)
    #     params = self._params()

    #     pt = calc_x_linear_increasing(
    #         updated_m, params.pt(0), params.pt(1), self._m
    #     )
    #     params = params.insert(
    #         pt, 1, to_unsqueeze=True, equalize_to=updated_m
    #     )
    #     return IncreasingRightTrapezoid(
    #         params, updated_m
    #     )


# class DecreasingRightTriangle(Polygon):
#     """A right triangle with a decreasing slope
#     """

#     PT = 2
    
#     def join(self, x: torch.Tensor) -> torch.Tensor:
#         """Calculate the membership of x with the triangle.

#         Args:
#             x (torch.Tensor): The value to calculate the membership for

#         Returns:
#             torch.Tensor: The membership
#         """
#         params = self._params()
    
#         return functional.shape.right_triangle(
#             unsqueeze(x), params.pt(0), params.pt(1), False, self._m
#         )

#     def _calc_areas(self) -> torch.Tensor:
#         """
#         Returns:
#             torch.Tensor: The area of the right triangle
#         """
#         params = self._params()
#         return self._resize_to_m((
#             0.5 * (params.pt(1)
#             - params.pt(0)) * self._m
#         ), self._m)

#     def _calc_mean_cores(self) -> torch.Tensor:
#         """
#         Returns:
#             torch.Tensor: the value of the height of the triangle
#         """
#         params = self._params()
#         return self._resize_to_m(params.pt(0), self._m)

#     def _calc_centroids(self) -> torch.Tensor:
#         """
#         Returns:
#             torch.Tensor: The center of mass for the right triangle
#         """
#         params = self._params()
#         return self._resize_to_m(
#             2 / 3 * params.pt(0) 
#             + 1 / 3 * params.pt(1), self._m
#         )
    
#     def scale(self, m: torch.Tensor) -> 'DecreasingRightTriangle':
#         """Update the vertical scale of the right triangle

#         Args:
#             m (torch.Tensor): The new vertical scale

#         Returns:
#             DecreasingRightTriangle: The updated vertical scale if the scale is greater
#         """
#         updated_m = functional.inter(self._m, m)
#         params = self._params()
        
#         return DecreasingRightTriangle(
#             params, updated_m
#         )

#     def truncate(self, m: torch.Tensor) -> 'DecreasingRightTrapezoid':
#         """
#         Args:
#             m (torch.Tensor): The new maximum value

#         Returns:
#             DecreasingRightTrapezoid: The triangle truncated into a RightTrapezoid
#         """
#         updated_m = functional.inter(self._m, m)
#         params = self._params()

#         pt = calc_x_linear_decreasing(
#             updated_m, params.pt(0), params.pt(1), self._m
#         )

#         params = params.insert(pt, 1, to_unsqueeze=True, equalize_to=updated_m)
#         return DecreasingRightTrapezoid(
#             params, updated_m
#         )



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

        new_pt1 = params.pt(0) * (1 - m) - self.params.pt(1) * m
        new_pt2 = params.pt(1) * (1 - m) - self.params.pt(2) * m
        params = self.params.insert(new_pt1, 1)
        params = params.replace(new_pt2, 2)
        return params

    def areas(self, m: torch.Tensor, truncate: bool = False) -> torch.Tensor:

        if truncate:
            params = self.truncate(m)
            a = params.pt(2) - params.pt(1)
            b = params.pt(3) - params.pt(0)
            return trapezoid_area(a, b, m)
        params = self._params()

        return triangle_area(params.pt(0), params.pt(2))
    
    def mean_cores(self, m: torch.Tensor, truncate: bool = False) -> torch.Tensor:
        
        if truncate:
            params = self.truncate(m)
            return trapezoid_mean_core(params.pt(1), params.pt(2))
        return self._params().pt(1)

    def centroids(self, m: torch.Tensor, truncate: bool = False) -> torch.Tensor:
        # x = (b+2a)/(3(a+b))h
        if truncate:
            params = self.truncate(m)
            a = params.pt(2) - params.pt(1)
            b = params.pt(3) - params.pt(0)
            return trapezoid_centroid(a, b, m)
        params = self._params()

        return triangle_centroid(params.pt(0), params.pt(1), params.pt(2))
    

    # def _calc_areas(self):
    #     """
    #     Returns:
    #         torch.Tensor: The area of triangle
    #     """
    #     params = self._params()
    #     return self._resize_to_m((
    #         0.5 * (params.pt(2) 
    #         - params.pt(0)) * self._m
    #     ), self._m)

    # def _calc_mean_cores(self):
    #     """
    #     Returns:
    #         torch.Tensor: the maximum value of the triangle
    #     """
    #     params = self._params()
    #     return self._resize_to_m(params.pt(1), self._m)

    # def _calc_centroids(self):
    #     """
    #     Returns:
    #         torch.Tensor: the center of mass for the triangle
    #     """
    #     params = self._params()
    #     return self._resize_to_m(1 / 3 * (
    #         params.pt(0) + params.pt(1) + params.pt(2)
    #     ), self._m)
    
    # def scale(self, m: torch.Tensor) -> 'Triangle':
    #     """Update the vertical scale of the triangle

    #     Args:
    #         m (torch.Tensor): The new vertical scale

    #     Returns:
    #         Triangle: The updated vertical scale if the scale is greater
    #     """
    #     updated_m = functional.inter(self._m, m)        
    #     return Triangle(
    #         self._params, updated_m
    #     )

    # def truncate(self, m: torch.Tensor) -> 'Trapezoid':
    #     """
    #     Args:
    #         m (torch.Tensor): The new maximum value

    #     Returns:
    #         Trapezoid: The triangle truncated into a Trapezoid
    #     """
    #     updated_m = functional.inter(self._m, m)

    #     params = self._params()
    #     pt1 = calc_x_linear_increasing(updated_m, params.pt(0), params.pt(1), self._m)
    #     pt2 = calc_x_linear_decreasing(updated_m, params.pt(1), params.pt(2), self._m)
    #     to_replace = torch.cat(
    #         [pt1.unsqueeze(3), pt2.unsqueeze(3)], dim=3
    #     )
    #     params= params.replace(
    #         to_replace, 1, False, equalize_to=updated_m
    #     )

    #     return Trapezoid(
    #         params, updated_m
    #     )


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
            x, params.pt(0), params.pt(1), self._m
        )

    def truncate(self, m: torch.Tensor) -> ShapeParams:

        new_pt1 = (self.params.pt(1) * (1 - m) - self.params.pt(0)) * m
        new_pt2 = self.params.pt(3) * m - self.params.pt(2) * (1 - m)
        params = self.params.replace(new_pt1, 1)
        params = params.replace(new_pt2, 2)
        return params

    def areas(self, m: torch.Tensor, truncate: bool = False) -> torch.Tensor:

        if truncate:
            params = self.truncate(m)
        else:
            params = self.params
            
        a = self.a(params)
        b = self.b(params)

        return trapezoid_area(a, b, m)
    
    def mean_cores(self, m: torch.Tensor, truncate: bool = False) -> torch.Tensor:
        
        return (self.params.pt(1) - self.params.pt(0)) / 2.0

    def centroids(self, m: torch.Tensor, truncate: bool = False) -> torch.Tensor:
        # x = (b+2a)/(3(a+b))h
        if truncate:
            params = self.truncate(m)
        else:
            params = self.params
        
        a = self.a(params)
        b = self.b(params)
        return trapezoid_centroid(a, b, m)


    # def _calc_areas(self):
    #     """
    #     Returns:
    #         torch.Tensor: the area of the triangle
    #     """
    #     params = self._params()
    #     return self._resize_to_m(
    #         0.5 * (params.pt(0)
    #         - params.pt(1)) * self._m, self._m
    #     )

    # def _calc_mean_cores(self) -> torch.Tensor:
    #     """
    #     Returns:
    #         torch.Tensor: the maximum value for the triangle
    #     """
    #     params = self._params()
    #     return self._resize_to_m(params.pt(1), self._m)

    # def _calc_centroids(self) -> torch.Tensor:
    #     """
    #     Returns:
    #         torch.Tensor: the center of mass for the triangle
    #     """
    #     params = self._params()
    #     return self._resize_to_m(params.pt(1), self._m)

    # def scale(self, m: torch.Tensor) -> 'IsoscelesTriangle':
    #     """Update the vertical scale of the triangle

    #     Args:
    #         m (torch.Tensor): The new vertical scale

    #     Returns:
    #         IsoscelesTriangle: The updated vertical scale if the scale is greater
    #     """
    #     params = self._params()
    #     updated_m = functional.inter(self._m, m)
    #     return IsoscelesTriangle(
    #         params, updated_m
    #     )

    # def truncate(self, m: torch.Tensor) -> 'IsoscelesTrapezoid':
    #     """
    #     Args:
    #         m (torch.Tensor): The new maximum value

    #     Returns:
    #         IsoscelesTrapezoid: The triangle truncated into an IsoscelesTrapezoid
    #     """
    #     params = self._params()
    #     updated_m = functional.inter(self._m, m)
    #     pt1 = calc_x_linear_increasing(updated_m, params.pt(0), params.pt(1), self._m)
    #     pt2 = calc_x_linear_decreasing(
    #         updated_m, params.pt(1), params.pt(1) + params.pt(1) - params.pt(0), self._m)

    #     to_replace = torch.cat(
    #         [pt1.unsqueeze(3), pt2.unsqueeze(3)], dim=3
    #     )

    #     # This will not work right

    #     params = self._params.replace(
    #         to_replace, 1, False, updated_m
    #     )
    #     return IsoscelesTrapezoid(
    #         params, updated_m
    #     )
