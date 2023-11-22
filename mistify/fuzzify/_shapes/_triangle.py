# 3rd party
import torch

# local
from ._base import Polygon
from ._trapezoid import IsoscelesTrapezoid, IncreasingRightTrapezoid, DecreasingRightTrapezoid, Trapezoid
from ._utils import calc_m_linear_increasing, calc_m_linear_decreasing, calc_x_linear_decreasing, calc_x_linear_increasing

from ...utils import unsqueeze

intersect = torch.min


class IncreasingRightTriangle(Polygon):

    PT = 2

    def join(self, x: torch.Tensor) -> torch.Tensor:
        return calc_m_linear_increasing(
            unsqueeze(x), self._params.pt(0), self._params.pt(1), self._m
        )

    def _calc_areas(self):
        
        return self._resize_to_m(
            0.5 * (self._params.pt(1)
            - self._params.pt(0)) * self._m, self._m
        )

    def _calc_mean_cores(self):
        return self._resize_to_m(self._params.pt(1), self._m)

    def _calc_centroids(self):
        
        p1, p2 = 1 / 3, 2 / 3

        return self._resize_to_m(
            p1 * self._params.pt(0) + p2 * self._params.pt(1), self._m
        )
    
    def scale(self, m: torch.Tensor) -> 'IncreasingRightTriangle':

        updated_m = intersect(m, self._m)
        
        return IncreasingRightTriangle(
            self._params, updated_m
        )

    def truncate(self, m: torch.Tensor) -> 'IncreasingRightTrapezoid':
        # TODO: FINISH
        updated_m = intersect(self._m, m)

        pt = calc_x_linear_increasing(
            updated_m, self._params.pt(0), self._params.pt(1), self._m
        )
        params = self._params.insert(
            pt, 1, to_unsqueeze=True, equalize_to=updated_m
        )
        return IncreasingRightTrapezoid(
            params, updated_m
        )


class DecreasingRightTriangle(Polygon):

    PT = 2
    
    def join(self, x: torch.Tensor):
    
        return calc_m_linear_decreasing(
            unsqueeze(x), self._params.pt(0), self._params.pt(1), self._m
        )

    def _calc_areas(self):
        
        return self._resize_to_m((
            0.5 * (self._params.pt(1)
            - self._params.pt(0)) * self._m
        ), self._m)

    def _calc_mean_cores(self):
        
        return self._resize_to_m(self._params.pt(0), self._m)

    def _calc_centroids(self):
        return self._resize_to_m(
            2 / 3 * self._params.pt(0) 
            + 1 / 3 * self._params.pt(1), self._m
        )
    
    def scale(self, m: torch.Tensor):
        updated_m = intersect(self._m, m)
        
        return DecreasingRightTriangle(
            self._params, updated_m
        )

    def truncate(self, m: torch.Tensor):
        updated_m = intersect(self._m, m)

        pt = calc_x_linear_decreasing(
            updated_m, self._params.pt(0), self._params.pt(1), self._m
        )

        params = self._params.insert(pt, 1, to_unsqueeze=True, equalize_to=updated_m)
        return DecreasingRightTrapezoid(
            params, updated_m
        )


class Triangle(Polygon):

    PT = 3

    def join(self, x: torch.Tensor):
        
        m1 = calc_m_linear_increasing(
            unsqueeze(x), self._params.pt(0), self._params.pt(1), self._m
        )
        m2 = calc_m_linear_decreasing(
            unsqueeze(x), self._params.pt(1), self._params.pt(2), self._m
        )
        return intersect(m1, m2)

    def _calc_areas(self):
        
        return self._resize_to_m((
            0.5 * (self._params.pt(2) 
            - self._params.pt(0)) * self._m
        ), self._m)

    def _calc_mean_cores(self):
        return self._resize_to_m(self._params.pt(1), self._m)

    def _calc_centroids(self):
        return self._resize_to_m(1 / 3 * (
            self._params.pt(0) + self._params.pt(1) + self._params.pt(2)
        ), self._m)
    
    def scale(self, m: torch.Tensor) -> 'Triangle':

        updated_m = intersect(self._m, m)        
        return Triangle(
            self._params, updated_m
        )

    def truncate(self, m: torch.Tensor) -> 'Trapezoid':
        updated_m = intersect(self._m, m)

        pt1 = calc_x_linear_increasing(updated_m, self._params.pt(0), self._params.pt(1), self._m)
        pt2 = calc_x_linear_decreasing(updated_m, self._params.pt(1), self._params.pt(2), self._m)
        to_replace = torch.cat(
            [pt1.unsqueeze(3), pt2.unsqueeze(3)], dim=3
        )
        params= self._params.replace(
            to_replace, 1, False, equalize_to=updated_m
        )

        return Trapezoid(
            params, updated_m
        )


class IncreasingRightTriangle(Polygon):

    PT = 2

    def join(self, x: torch.Tensor) -> torch.Tensor:
        return calc_m_linear_increasing(
            unsqueeze(x), self._params.pt(0), self._params.pt(1), self._m
        )

    def _calc_areas(self):
        
        return self._resize_to_m(
            0.5 * (self._params.pt(1)
            - self._params.pt(0)) * self._m, self._m
        )

    def _calc_mean_cores(self):
        return self._resize_to_m(self._params.pt(1), self._m)

    def _calc_centroids(self):
        
        p1, p2 = 1 / 3, 2 / 3

        return self._resize_to_m(
            p1 * self._params.pt(0) + p2 * self._params.pt(1), self._m
        )
    
    def scale(self, m: torch.Tensor) -> 'IncreasingRightTriangle':

        updated_m = intersect(m, self._m)
        
        return IncreasingRightTriangle(
            self._params, updated_m
        )

    def truncate(self, m: torch.Tensor) -> 'IncreasingRightTrapezoid':
        # TODO: FINISH
        updated_m = intersect(self._m, m)

        pt = calc_x_linear_increasing(
            updated_m, self._params.pt(0), self._params.pt(1), self._m
        )
        params = self._params.insert(
            pt, 1, to_unsqueeze=True, equalize_to=updated_m
        )
        return IncreasingRightTrapezoid(
            params, updated_m
        )


class DecreasingRightTriangle(Polygon):

    PT = 2
    
    def join(self, x: torch.Tensor):
    
        return calc_m_linear_decreasing(
            unsqueeze(x), self._params.pt(0), self._params.pt(1), self._m
        )

    def _calc_areas(self):
        
        return self._resize_to_m((
            0.5 * (self._params.pt(1)
            - self._params.pt(0)) * self._m
        ), self._m)

    def _calc_mean_cores(self):
        
        return self._resize_to_m(self._params.pt(0), self._m)

    def _calc_centroids(self):
        return self._resize_to_m(
            2 / 3 * self._params.pt(0) 
            + 1 / 3 * self._params.pt(1), self._m
        )
    
    def scale(self, m: torch.Tensor):
        updated_m = intersect(self._m, m)
        
        return DecreasingRightTriangle(
            self._params, updated_m
        )

    def truncate(self, m: torch.Tensor):
        updated_m = intersect(self._m, m)

        pt = calc_x_linear_decreasing(
            updated_m, self._params.pt(0), self._params.pt(1), self._m
        )

        params = self._params.insert(pt, 1, to_unsqueeze=True, equalize_to=updated_m)
        return DecreasingRightTrapezoid(
            params, updated_m
        )


class Triangle(Polygon):

    PT = 3

    def join(self, x: torch.Tensor):
        
        m1 = calc_m_linear_increasing(
            unsqueeze(x), self._params.pt(0), self._params.pt(1), self._m
        )
        m2 = calc_m_linear_decreasing(
            unsqueeze(x), self._params.pt(1), self._params.pt(2), self._m
        )
        return intersect(m1, m2)

    def _calc_areas(self):
        
        return self._resize_to_m((
            0.5 * (self._params.pt(2) 
            - self._params.pt(0)) * self._m
        ), self._m)

    def _calc_mean_cores(self):
        return self._resize_to_m(self._params.pt(1), self._m)

    def _calc_centroids(self):
        return self._resize_to_m(1 / 3 * (
            self._params.pt(0) + self._params.pt(1) + self._params.pt(2)
        ), self._m)
    
    def scale(self, m: torch.Tensor) -> 'Triangle':

        updated_m = intersect(self._m, m)        
        return Triangle(
            self._params, updated_m
        )

    def truncate(self, m: torch.Tensor) -> 'Trapezoid':
        updated_m = intersect(self._m, m)

        pt1 = calc_x_linear_increasing(updated_m, self._params.pt(0), self._params.pt(1), self._m)
        pt2 = calc_x_linear_decreasing(updated_m, self._params.pt(1), self._params.pt(2), self._m)
        to_replace = torch.cat(
            [pt1.unsqueeze(3), pt2.unsqueeze(3)], dim=3
        )
        params= self._params.replace(
            to_replace, 1, False, equalize_to=updated_m
        )

        return Trapezoid(
            params, updated_m
        )


class IsoscelesTriangle(Polygon):

    PT = 2

    def join(self, x: torch.Tensor) -> torch.Tensor:

        left_m = calc_m_linear_increasing(
            unsqueeze(x), self._params.pt(0), self._params.pt(1), self._m
        )
        right_m = calc_m_linear_decreasing(
            unsqueeze(x), self._params.pt(1), 
            self._params.pt(1) + (self._params.pt(1) - self._params.pt(0)), 
            self._m
        )
        return torch.max(left_m, right_m)

    def _calc_areas(self):
        
        return self._resize_to_m(
            0.5 * (self._params.pt(0)
            - self._params.pt(1)) * self._m, self._m
        )

    def _calc_mean_cores(self):
        return self._resize_to_m(self._params.pt(1), self._m)

    def _calc_centroids(self):
        return self._resize_to_m(self._params.pt(1), self._m)

    def scale(self, m: torch.Tensor) -> 'IsoscelesTriangle':
        updated_m = intersect(self._m, m)
        return IsoscelesTriangle(
            self._params, updated_m
        )

    def truncate(self, m: torch.Tensor) -> 'IsoscelesTrapezoid':
        
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



# class IsoscelesTriangle(Polygon):

#     PT = 2

#     def join(self, x: torch.Tensor) -> torch.Tensor:

#         left_m = calc_m_linear_increasing(
#             unsqueeze(x), self._params.pt(0), self._params.pt(1), self._m
#         )
#         right_m = calc_m_linear_decreasing(
#             unsqueeze(x), self._params.pt(1), 
#             self._params.pt(1) + (self._params.pt(1) - self._params.pt(0)), 
#             self._m
#         )
#         return torch.max(left_m, right_m)

#     def _calc_areas(self):
        
#         return self._resize_to_m(
#             0.5 * (self._params.pt(0)
#             - self._params.pt(1)) * self._m, self._m
#         )

#     def _calc_mean_cores(self):
#         return self._resize_to_m(self._params.pt(1), self._m)

#     def _calc_centroids(self):
#         return self._resize_to_m(self._params.pt(1), self._m)

#     def scale(self, m: torch.Tensor) -> 'IsoscelesTriangle':
#         updated_m = intersect(self._m, m)
#         return IsoscelesTriangle(
#             self._params, updated_m
#         )

#     def truncate(self, m: torch.Tensor) -> 'IsoscelesTrapezoid':
        
#         updated_m = intersect(self._m, m)
#         pt1 = calc_x_linear_increasing(updated_m, self._params.pt(0), self._params.pt(1), self._m)
#         pt2 = calc_x_linear_decreasing(
#             updated_m, self._params.pt(1), self._params.pt(1) + self._params.pt(1) - self._params.pt(0), self._m)

#         to_replace = torch.cat(
#             [pt1.unsqueeze(3), pt2.unsqueeze(3)], dim=3
#         )

#         params = self._params.replace(
#             to_replace, 1, False, updated_m
#         )
#         return IsoscelesTrapezoid(
#             params, updated_m
#         )

