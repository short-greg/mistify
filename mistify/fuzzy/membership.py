# 1st party
from abc import abstractmethod
from dataclasses import dataclass
import typing

# 3rd party
import torch
import torch.nn as nn
import torch.nn.functional

from abc import abstractmethod, abstractproperty
import typing
import torch
from dataclasses import dataclass
from .utils import intersect, positives
from .._base import Polygon

# local
from .._base.membership import Shape
#"""
# Classes for calculating the membership 
# """

from abc import abstractmethod, abstractproperty
import typing
import torch
from dataclasses import dataclass
from .utils import intersect, positives
from .._base import (
    check_contains, ShapeParams, calc_m_linear_decreasing, calc_area_logistic,
    calc_area_logistic_one_side, calc_dx_logistic, calc_m_linear_increasing, calc_m_logistic,
    calc_x_linear_decreasing, calc_x_linear_increasing, calc_x_logistic, 
    unsqueeze
)

"""
Classes for calculating the membership 
"""

# TODO: 
# Analyze the classes and design an approach to make
# them easier to work with
# Change so that it uses the FuzzySet class


# TODO: 
# Analyze the classes and design an approach to make
# them easier to work with
# Change so that it uses the FuzzySet class

def calc_m_flat(x, pt1: torch.Tensor, pt2: torch.Tensor, m: torch.Tensor):

    return m * check_contains(x, pt1, pt2).float()



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


class Logistic(Shape):

    def __init__(
        self, biases: ShapeParams, scales: ShapeParams, m: torch.Tensor=None
    ):
        self._biases = biases
        self._scales = scales

        self._m = m if m is not None else positives(
            self._biases.batch_size, self._biases.set_size, 
            self._biases.n_terms, device=biases.x.device
        )

        super().__init__(
            self._biases.n_variables,
            self._biases.n_terms
        )

    @property
    def biases(self):
        return self._biases
    
    @property
    def scales(self):
        return self._scales
    
    @classmethod
    def from_combined(cls, params: ShapeParams, m: torch.Tensor=None):

        #if params.x.dim() == 4:
        return cls(
            ShapeParams(params.sub((0, 1))), 
            ShapeParams(params.sub((1, 2))), m
        )
        # return cls(params[:,:,0], params[:,:,1], m)


class LogisticBell(Logistic):

    def join(self, x: torch.Tensor) -> torch.Tensor:
        z = self._scales.pt(0) * (unsqueeze(x) - self._biases.pt(0))
        sig = torch.sigmoid(z)
        # not 4 / s
        return 4  * (1 - sig) * sig * self._m

    def _calc_areas(self):
        return self._resize_to_m(4 * self._m / self._biases.pt(0), self._m)
        
    def _calc_mean_cores(self):
        return self._resize_to_m(self._biases.pt(0), self._m)

    def _calc_centroids(self):
        return self._resize_to_m(self._biases.pt(0), self._m)

    def scale(self, m: torch.Tensor) -> 'LogisticBell':
        updated_m = intersect(self._m, m)
        return LogisticBell(
            self._biases, self._scales, updated_m
        )

    def truncate(self, m: torch.Tensor) -> 'LogisticTrapezoid':

        return LogisticTrapezoid(
            self._biases, self._scales,  m, self._m 
        )


class LogisticTrapezoid(Logistic):
    
    def __init__(
        self, biases: ShapeParams, scales: ShapeParams, 
        truncated_m: torch.Tensor=None, scaled_m: torch.Tensor=None
    ):
        super().__init__(biases, scales, scaled_m)

        if truncated_m is None:
            truncated_m = positives(*self._m.size(), device=self._m.device)

        self._truncated_m = intersect(truncated_m, self._m)
        
        dx = unsqueeze(calc_dx_logistic(self._truncated_m, self._scales.pt(0), self._m))
        self._dx = ShapeParams(dx)
        self._pts = ShapeParams(torch.concat([
            self._biases.x - self._dx.x,
            self._biases.x + self._dx.x
        ], dim=dx.dim() - 1))

    @property
    def dx(self):
        return self._dx
    
    @property
    def m(self) -> torch.Tensor:
        return self._truncated_m

    def join(self, x: torch.Tensor) -> 'torch.Tensor':
        x = unsqueeze(x)
        inside = check_contains(x, self._pts.pt(0), self._pts.pt(1)).float()
        m1 = calc_m_logistic(x, self._biases.pt(0), self._scales.pt(0), self._m) * (1 - inside)
        m2 = self._truncated_m * inside
        return torch.max(m1, m2)

    def _calc_areas(self):
        # symmetrical so multiply by 2
        return self._resize_to_m(2 * calc_area_logistic_one_side(
            self._pts.pt(0), self._biases.pt(0), self._scales.pt(0), self._m
        ), self._m)
        
    def _calc_mean_cores(self):
        return self._resize_to_m(self._biases.pt(0), self._m)

    def _calc_centroids(self):
        return self._resize_to_m(self._biases.pt(0), self._m)

    def scale(self, m: torch.Tensor) -> 'LogisticTrapezoid':
        updated_m = intersect(self._m, m)
        # TODO: check if multiplication is correct
        truncated_m = self._truncated_m * updated_m

        return LogisticTrapezoid(
            self._biases, self._scales, truncated_m, updated_m
        )

    def truncate(self, m: torch.Tensor) -> 'LogisticTrapezoid':
        truncated_m = intersect(self._truncated_m, m)
        return LogisticTrapezoid(
            self._biases, self._scales, truncated_m, self._m
        )


class RightLogistic(Logistic):
    
    def __init__(
        self, biases: ShapeParams, scales: ShapeParams, is_right: bool=True,
        m: torch.Tensor= None
    ):
        super().__init__(biases, scales, m)
        self._is_right = is_right
        self._direction = is_right * 2 - 1
    
    def _on_side(self, x: torch.Tensor):
        if self._is_right:
            side = x >= self._biases.pt(0)
        else: side = x <= self._biases.pt(0)
        return side

    def join(self, x: torch.Tensor):
        x = unsqueeze(x)
        return calc_m_logistic(
            x, self._biases.pt(0), 
            self._scales.pt(0), self._m
        ) * self._on_side(x).float()

    def _calc_areas(self):
        return self._resize_to_m(2 * self._m / self._biases.pt(0), self._m)

    def _calc_mean_cores(self):
        return self._resize_to_m(self._biases.pt(0), self._m)

    def _calc_centroids(self):
        base_y = 0.75 if self._is_right else 0.25
        x = torch.logit(torch.tensor(base_y, dtype=torch.float, device=self._m.device)) / self._scales.pt(0) + self._biases.pt(0)
        return self._resize_to_m(x, self._m)

    def scale(self, m: torch.Tensor) -> 'RightLogistic':
        updated_m = intersect(self._m, m)
        
        return RightLogistic(
            self._biases, self._scales, self._is_right, updated_m
        )

    def truncate(self, m: torch.Tensor) -> 'LogisticTrapezoid':
        truncated_m = intersect(self._m, m)
        return RightLogisticTrapezoid(
            self._biases, self._scales, self._is_right, truncated_m, self._m
        )

    @classmethod
    def from_combined(cls, params: torch.Tensor, is_right: bool=True,m: torch.Tensor=None):
        # TODO: Check this and confirm
        if params.dim() == 4:
            return cls(params.sub(0), params.sub(1), is_right, m)
        return cls(params.sub(0), params.sub(1), is_right, m)


class RightLogisticTrapezoid(Logistic):

    def __init__(
        self, biases: ShapeParams, scales: ShapeParams, is_right: bool, 
        truncated_m: torch.Tensor=None, scaled_m: torch.Tensor=None
        
    ):
        super().__init__(biases, scales, scaled_m)

        if truncated_m is None:
            truncated_m = positives(self._m.size(), device=self._m.device)

        self._truncated_m = intersect(self._m, truncated_m)
        dx = unsqueeze(calc_dx_logistic(self._truncated_m, self._scales.pt(0), self._m))
        self._dx = ShapeParams(dx)
        self._is_right = is_right
        self._direction = is_right * 2 - 1
        self._pts = ShapeParams(self._biases.x + self._direction * dx)

    @property
    def dx(self):
        return self._dx

    @property
    def m(self):
        return self._truncated_m

    def _contains(self, x: torch.Tensor):
        if self._is_right:
            square_contains = (x >= self._biases.pt(0)) & (x <= self._pts.pt(0))
            logistic_contains = x >= self._pts.pt(0)
        else:
            square_contains = (x <= self._biases.pt(0)) & (x >= self._pts[0])
            logistic_contains = x <= self._pts.pt(0)
        return square_contains.float(), logistic_contains.float()

    def join(self, x: torch.Tensor) -> 'torch.Tensor':
        x = unsqueeze(x)
        
        square_contains, logistic_contains = self._contains(x)
        
        m1 = calc_m_logistic(
            x, self._biases.pt(0), self._scales.pt(0), self._m
        ) * logistic_contains
        m2 = self._m * square_contains
        return torch.max(m1, m2)

    def _calc_areas(self):
        a1 = self._resize_to_m(calc_area_logistic_one_side(
            self._pts.pt(0), self._biases.pt(0), self._scales.pt(0), 
            self._m), self._m)
        a2 = 0.5 * (self._biases.pt(0) + self._pts.pt(0)) * self._m
        return self._resize_to_m(a1 + a2, self._m)

    def _calc_mean_cores(self):
        return self._resize_to_m(0.5 * (self._biases.pt(0) + self._pts.pt(0)), self._m) 

    def _calc_centroids(self):

        # area up to "dx"
        # print('Centroids: ', self._scales.x.size(), self._dx.x.size())
        p = torch.sigmoid(self._scales.pt(0) * (-self._dx.pt(0)))
        centroid_logistic = self._biases.pt(0) + torch.logit(p / 2) / self._scales.pt(0)
        centroid_square = self._biases.pt(0) - self._dx.pt(0) / 2

        centroid = (centroid_logistic * p + centroid_square * self._dx.pt(0)) / (p + self._dx.pt(0))
        if self._is_right:
            return self._biases.pt(0) + self._biases.pt(0) - centroid
        return self._resize_to_m(centroid, self._m)

    def scale(self, m: torch.Tensor) -> 'RightLogisticTrapezoid':

        updated_m = intersect(self._m, m)

        # TODO: Confirm if this is correct
        # I think it should be intersecting rather than multiplying
        truncated_m = self._truncated_m * updated_m

        return RightLogisticTrapezoid(
            self._biases, self._scales, self._is_right, truncated_m, updated_m
        )

    def truncate(self, m: torch.Tensor) -> 'RightLogisticTrapezoid':

        truncated_m = intersect(self._truncated_m, m)
        return RightLogisticTrapezoid(
            self._biases, self._scales, self._is_right, truncated_m, self._m
        )

    @classmethod
    def from_combined(cls, params: torch.Tensor, is_right: bool=True,m: torch.Tensor=None):

        if params.dim() == 4:

            return cls(params.sub(0), params.sub(1), is_right, m)
        return cls(params.sub(0), params.sub(1), is_right, m)


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

