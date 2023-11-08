from ._base import Polygon, ShapeParams
from ...utils import unsqueeze
import torch
from ._utils import (
    calc_area_logistic, calc_area_logistic_one_side, 
    calc_dx_logistic, calc_m_flat, calc_m_linear_decreasing, calc_m_linear_increasing,
    calc_m_logistic, calc_x_linear_decreasing, calc_x_logistic,
    calc_x_linear_increasing
)

intersect = torch.min


class Trapezoid(Polygon):

    PT = 4

    def join(self, x: torch.Tensor) -> torch.Tensor:

        x = unsqueeze(x)
        m1 = calc_m_linear_increasing(x, self._params.pt(0), self._params.pt(1), self._m)
        m2 = calc_m_flat(x, self._params.pt(1), self._params.pt(2), self._m)
        m3 = calc_m_linear_decreasing(x, self._params.pt(2), self._params.pt(3), self._m)

        return torch.max(torch.max(
            m1, m2
        ), m3)

    def _calc_areas(self):
        
        return self._resize_to_m((
            0.5 * (self._params.pt(2) 
            - self._params.pt(0)) * self._m
        ), self._m)

    def _calc_mean_cores(self):
        return self._resize_to_m(
            0.5 * (self._params.pt(1) + self._params.pt(2)), self._m
        )

    def _calc_centroids(self):
        d1 = 0.5 * (self._params.pt(1) - self._params.pt(0))
        d2 = self._params.pt(2) - self._params.pt(1)
        d3 = 0.5 * (self._params.pt(3) - self._params.pt(2))

        return self._resize_to_m((
            d1 * (2 / 3 * self._params.pt(1) + 1 / 3 * self._params.pt(0)) +
            d2 * (1 / 2 * self._params.pt(2) + 1 / 2 *  self._params.pt(1)) + 
            d3 * (1 / 3 * self._params.pt(3) + 2 / 3 * self._params.pt(2))
        ) / (d1 + d2 + d3), self._m)

    def scale(self, m: torch.Tensor) -> 'Trapezoid':
        updated_m = intersect(self._m, m)
        return Trapezoid(
            self._params, updated_m
        )

    def truncate(self, m: torch.Tensor) -> 'Trapezoid':
        updated_m = intersect(self._m, m)

        # m = ShapeParams(m, True, m.dim() == 3)
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
    def a(self):
        return (
            self._params.pt(2) - self._params.pt(0) + 
            self._params.pt(1) - self._params.pt(0)
        )

    @property
    def b(self):
        return self._params.pt(2) - self._params.pt(1)

    def _calc_areas(self):
        
        return self._resize_to_m(
            0.5 * (self.a + self.b) * self._m, self._m
        )

    def _calc_mean_cores(self):
        return self._resize_to_m(0.5 * (self._params.pt(2) + self._params.pt(1)), self._m)

    def _calc_centroids(self):
        return self.mean_cores

    def scale(self, m: torch.Tensor) -> 'IsoscelesTrapezoid':
        updated_m = intersect(self._m, m)
        return IsoscelesTrapezoid(self._params, updated_m)

    def truncate(self, m: torch.Tensor) -> 'IsoscelesTrapezoid':
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

        m = calc_m_linear_decreasing(
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
        return DecreasingRightTrapezoid(self._params, intersect(m, self._m))

    def truncate(self, m: torch.Tensor) -> 'DecreasingRightTrapezoid':
        updated_m = intersect(m, self._m)
        
        x = calc_x_linear_decreasing(
            updated_m, self._params.pt(0), self._params.pt(1), self._m
        )
        params = self._params.replace(x, 1, True, updated_m)
        return DecreasingRightTrapezoid(params, updated_m)


# class Trapezoid(Polygon):

#     PT = 4

#     def join(self, x: torch.Tensor) -> torch.Tensor:

#         x = unsqueeze(x)
#         m1 = calc_m_linear_increasing(x, self._params.pt(0), self._params.pt(1), self._m)
#         m2 = calc_m_flat(x, self._params.pt(1), self._params.pt(2), self._m)
#         m3 = calc_m_linear_decreasing(x, self._params.pt(2), self._params.pt(3), self._m)

#         return torch.max(torch.max(
#             m1, m2
#         ), m3)

#     def _calc_areas(self):
        
#         return self._resize_to_m((
#             0.5 * (self._params.pt(2) 
#             - self._params.pt(0)) * self._m
#         ), self._m)

#     def _calc_mean_cores(self):
#         return self._resize_to_m(
#             0.5 * (self._params.pt(1) + self._params.pt(2)), self._m
#         )

#     def _calc_centroids(self):
#         d1 = 0.5 * (self._params.pt(1) - self._params.pt(0))
#         d2 = self._params.pt(2) - self._params.pt(1)
#         d3 = 0.5 * (self._params.pt(3) - self._params.pt(2))

#         return self._resize_to_m((
#             d1 * (2 / 3 * self._params.pt(1) + 1 / 3 * self._params.pt(0)) +
#             d2 * (1 / 2 * self._params.pt(2) + 1 / 2 *  self._params.pt(1)) + 
#             d3 * (1 / 3 * self._params.pt(3) + 2 / 3 * self._params.pt(2))
#         ) / (d1 + d2 + d3), self._m)

#     def scale(self, m: torch.Tensor) -> 'Trapezoid':
#         updated_m = intersect(self._m, m)
#         return Trapezoid(
#             self._params, updated_m
#         )

#     def truncate(self, m: torch.Tensor) -> 'Trapezoid':
#         updated_m = intersect(self._m, m)

#         # m = ShapeParams(m, True, m.dim() == 3)
#         left_x = calc_x_linear_increasing(
#             updated_m, self._params.pt(0), self._params.pt(1), self._m
#         )

#         right_x = calc_x_linear_decreasing(
#             updated_m, self._params.pt(2), self._params.pt(3), self._m
#         )

#         params = self._params.replace(left_x, 1, to_unsqueeze=True, equalize_to=updated_m)
#         params = params.replace(right_x, 2, to_unsqueeze=True)

#         return Trapezoid(
#             params, updated_m, 
#         )