# 1st party
import typing

# 3rd party
import torch
import torch.nn.functional
from torch import Tensor

# local
from ._base import ShapeParams, Nonmonotonic
from ...utils import unsqueeze, check_contains
from ._utils import calc_dx_logistic, calc_area_logistic_one_side, calc_m_logistic, calc_x_logistic
from ... import _functional as functional


def logistic_area(m, scale):
    
    return 4 * m * scale


def logistic_invert(y, bias, scale):
    
    base = torch.sqrt(-2 * torch.log(y + 1e-7) / (scale ** 2))
    return bias - base, bias + base


def logistic_area_up_to_a(a, bias, scale):

    # TODO: presently incro
    z = scale * (a - bias)
    return torch.sigmoid(z) * (4 / scale)


def logistic_area_up_to_a_inv(area, bias, scale, increasing: bool=True):
    
    dx = torch.logit(area * scale / 4.0) * scale
    
    if increasing:
        return bias - dx
    return bias + dx


def logistic(x: torch.Tensor,  bias: torch.Tensor, scale: torch.Tensor):
    
    sig = torch.sigmoid(-(x - bias) / scale)
    return 4  * (1 - sig) * sig


def truncated_logistic_area(bias: torch.Tensor, std: torch.Tensor, height: torch.Tensor) -> torch.Tensor:
    
    pts = logistic_invert(height, bias, std)
    rec_area = (pts[1] - pts[0]) * height
    gauss_area = logistic_area_up_to_a(pts[1], bias, std)
    return rec_area + 2 * gauss_area


def truncated_logistic_mean_core(bias: torch.Tensor, std: torch.Tensor, height: torch.Tensor) -> torch.Tensor:
    pts = logistic_invert(height, bias, std)
    return (pts[1] + pts[2]) / 2.0


def half_logistic_area(bias: torch.Tensor, std: torch.Tensor, height: torch.Tensor) -> torch.Tensor:
    pts = logistic_invert(height, bias, std)
    rec_area = (bias - pts[0]) * height
    gauss_area = logistic_area_up_to_a(pts[1], bias, std)
    return gauss_area + rec_area


def half_logistic_centroid(bias: torch.Tensor, std: torch.Tensor, height: torch.Tensor, increasing: bool=True) -> torch.Tensor:

    return logistic_area_up_to_a_inv(height, bias, std, increasing)


def truncated_half_logistic_area(bias: torch.Tensor, std: torch.Tensor, height: torch.Tensor) -> torch.Tensor:
    pts = logistic_invert(height, bias, std)
    rec_area = (bias - pts[0]) * height
    gauss_area = logistic_area_up_to_a(pts[1], bias, std)
    return gauss_area + rec_area


def truncated_half_logistic_mean_core(bias: torch.Tensor, std: torch.Tensor, height: torch.Tensor, increasing: bool=True) -> torch.Tensor:    
    pts = logistic_invert(height, bias, std)
    if increasing:
        return pts[0]
    return pts[1]


def truncated_half_logistic_centroid(bias: torch.Tensor, std: torch.Tensor, height: torch.Tensor, increasing: bool=True) -> torch.Tensor:
    pts = logistic_invert(height, bias, std)
    rec_area = (bias - pts[0]) * height
    gauss_area = logistic_area_up_to_a(pts[1], bias, std)
    gauss_centroid = logistic_area_up_to_a_inv(gauss_area / 2.0, bias, std, increasing)
    rec_centroid = (bias + pts[0]) / 2.0 if increasing else (bias + pts[1]) / 2.0

    return (rec_centroid * rec_area + gauss_centroid * gauss_area) / (rec_area + gauss_area)


class Logistic(Nonmonotonic):
    """A logistic bell curve based on the shape of the logistic distribution
    but normalized so the maximum value is 1 by default.
    """

    def __init__(
        self, biases: ShapeParams, scales: ShapeParams
    ):
        """The base class for logistic distribution functions

        Note: Don't need to sort for this because there is only one point per parameter

        Args:
            biases (ShapeParams): The bias of the distribution
            scales (ShapeParams): The scale value for the distribution
        """
        super().__init__(
            biases.n_variables,
            biases.n_terms
        )
        self._biases = biases
        self._scales = scales

    @property
    def sigma(self) -> torch.Tensor:

        return torch.nn.functional.softplus(self._scales.pt(0))

    @property
    def biases(self) -> 'ShapeParams':
        """
        Returns:
            ShapeParams: The bias values
        """
        return self._biases
    
    @property
    def scales(self) -> 'ShapeParams':
        """
        Returns:
            ShapeParams: The scales
        """
        return self._scales
    
    @classmethod
    def from_combined(cls, params: ShapeParams, m: torch.Tensor=None) -> 'Logistic':
        """Create the shape from 

        Returns:
            Logistic: The logistic distribution function 
        """
        return cls(
            params.sub((0, 1)), 
            params.sub((1, 2)), m
        )


class LogisticBell(Logistic):
    """Use the GaussianBell function as the membership function
    """
    def join(self, x: Tensor) -> Tensor:
        return logistic(
            unsqueeze(x), self._biases.pt(0), self.sigma
        )
    
    def areas(self, m: Tensor, truncate: bool = False) -> Tensor:
        
        if truncate:
            return truncated_logistic_area(m, self.sigma)
        return logistic_area(m, self.sigma)
    
    def mean_cores(self, m: Tensor, truncate: bool = False) -> Tensor:
        
        if truncate:
            return truncated_logistic_mean_core(self._biases.pt(0), self.sigma)
        return self._biases.pt(0)
    
    def centroids(self, m: Tensor, truncate: bool = False) -> Tensor:
        
        return self._biases.pt(0)
    

class HalfLogisticBell(Logistic):
    """Use the Half Logistic Bell function as the membership function
    """
    def __init__(self, biases: ShapeParams, scales: ShapeParams, increasing: bool=True):
        super().__init__(biases, scales)
        self.increasing = increasing

    def join(self, x: Tensor) -> Tensor:

        if self.increasing:
            contains = (x <= self._biases.pt(0))
        else:
            contains = (x >= self._biases.pt(0))

        return logistic(
            unsqueeze(x), self._biases.pt(0), self.sigma
        ) * contains
    
    def areas(self, m: Tensor, truncate: bool = False) -> Tensor:
        
        if truncate:
            return truncated_half_logistic_area(self._biases.pt(0), self.sigma, m)
        return half_logistic_area(self._biases.pt(0), self.sigma, m)
    
    def mean_cores(self, m: Tensor, truncate: bool = False) -> Tensor:
        
        if truncate:
            return truncated_half_logistic_mean_core(self._biases.pt(0), self.sigma, m, self.increasing)
        return self._biases.pt(0)
    
    def centroids(self, m: Tensor, truncate: bool = False) -> Tensor:
        
        if truncate:
            return truncated_half_logistic_centroid(self._biases.pt(0), self.sigma, m, self.increasing)
        return half_logistic_centroid(self._biases.pt(0), self.sigma, m, self.increasing)



# class LogisticBell(Logistic):
#     """Use the LogisticBell function as the membership function
#     """

#     def join(self, x: torch.Tensor) -> torch.Tensor:
#         # z = self._scales.pt(0) * (unsqueeze(x) - self._biases.pt(0))
#         # sig = torch.sigmoid(z)
#         # not 4 / s
#         # return 4  * (1 - sig) * sig * self._m
#         return logistic(unsqueeze(x), self._m, self._biases.pt(0), self._scales.pt(0))

#     def _calc_areas(self):

#         return self._resize_to_m(
#             logistic_area(self._m, self._scales.pt(0)), self._m
#         )
#         # return self._resize_to_m(4 * self._m / self._biases.pt(0), self._m)
        
#     def _calc_mean_cores(self):
#         return self._resize_to_m(self._biases.pt(0), self._m)

#     def _calc_centroids(self):
#         return self._resize_to_m(self._biases.pt(0), self._m)

#     def scale(self, m: torch.Tensor) -> 'LogisticBell':
#         """Scale the height of the LogisticBell

#         Args:
#             m (torch.Tensor): The new height

#         Returns:
#             LogisticBell: The updated LogisticBell
#         """
#         updated_m = functional.inter(self._m, m)
#         return LogisticBell(
#             self._biases, self._scales, updated_m
#         )

#     def truncate(self, m: torch.Tensor) -> 'LogisticTrapezoid':
#         """Truncate the height of the LogisticBell

#         Args:
#             m (torch.Tensor): The new height

#         Returns:
#             LogisticBell: The updated LogisticBell
#         """
#         m = functional.inter(self._m, m)
#         return LogisticTrapezoid(
#             self._biases, self._scales, m, self._m 
#         )


# class LogisticTrapezoid(Logistic):
#     """A membership function that has a ceiling on the heighest value
#     """
    
#     def __init__(
#         self, biases: ShapeParams, scales: ShapeParams, 
#         truncated_m: torch.Tensor=None, scaled_m: torch.Tensor=None
#     ):
#         """Create a membership function that has a ceiling on the heighest value

#         Note: Don't need to sort for this because it is derived

#         Args:
#             biases (ShapeParams): The biases for the logistic part of the funciton
#             scales (ShapeParams): The scales for the logistic part of teh function
#             truncated_m (torch.Tensor, optional): The maximum height of the membership. Defaults to None.
#             scaled_m (torch.Tensor, optional): The scale of the LogisticTrapezoid. Defaults to None.
#         """
#         super().__init__(biases, scales, scaled_m)

#         truncated_m = self._init_m(truncated_m, biases.device)
#         self._truncated_m = functional.inter(truncated_m, self._m)
        
#         left = self._resize_to_m(logistic_area_up_to_a_inv(
#             self._truncated_m, self._m, self._biases.pt(0), self.sigma
#         ), self._truncated_m)
#         self._left = ShapeParams(unsqueeze(left))
#         self._right = ShapeParams(unsqueeze(2 * self._biases.pt(0) - self._left.pt(0)))
#         # TODO: Update
#         # dx = unsqueeze(calc_dx_logistic(self._truncated_m, self._scales.pt(0), self._m))
#         # self._dx = ShapeParams(dx)
#         # self._pts = ShapeParams(torch.concat([
#         #     self._biases.x - self._dx.x,
#         #     self._biases.x + self._dx.x
#         # ], dim=dx.dim() - 1))

#     @property
#     def dx(self):
#         return self._dx
    
#     @property
#     def m(self) -> torch.Tensor:
#         return self._truncated_m

#     def join(self, x: torch.Tensor) -> 'torch.Tensor':
#         x = unsqueeze(x)
#         inside = check_contains(x, self._left.pt(0), self._right.pt(0)).float()
#         m1 = logistic(x, self._m, self._biases.pt(0), self.sigma)
#         # m1 = calc_m_logistic(x, self._biases.pt(0), self._scales.pt(0), self._m) * (1 - inside)
#         m2 = self._truncated_m * inside
#         return torch.max(m1, m2)

#     def _calc_areas(self):
#         # symmetrical so multiply by 2

#         logist = 2 * logistic_area_up_to_a(
#             self._left.pt(0), self._m, self._biases.pt(0), self.sigma
#         )
#         flat = (self._left.pt(0) - self._right.pt(0)) * self._m
#         return self._resize_to_m(
#             logist + flat, self._m
#         )

#         # return self._resize_to_m(2 * calc_area_logistic_one_side(
#         #     self._pts.pt(0), self._biases.pt(0), self._scales.pt(0), self._m
#         # ), self._m)
        
#     def _calc_mean_cores(self):
#         return self._resize_to_m(self._biases.pt(0), self._m)

#     def _calc_centroids(self):
#         return self._resize_to_m(self._biases.pt(0), self._m)

#     def scale(self, m: torch.Tensor) -> 'LogisticTrapezoid':
#         updated_m = functional.inter(self._m, m)
#         # TODO: check if multiplication is correct
#         truncated_m = self._truncated_m * updated_m

#         return LogisticTrapezoid(
#             self._biases, self._scales, truncated_m, updated_m
#         )

#     def truncate(self, m: torch.Tensor) -> 'LogisticTrapezoid':
#         truncated_m = functional.inter(self._truncated_m, m)
#         return LogisticTrapezoid(
#             self._biases, self._scales, truncated_m, self._m
#         )


# class RightLogistic(Logistic):
#     """A Logistic shaped membership function that contains only one side
#     """
    
#     def __init__(
#         self, biases: ShapeParams, scales: ShapeParams, is_right: bool=True,
#         m: torch.Tensor= None
#     ):
#         """Create a Logistic shaped membership function that contains only one side

#         Args:
#             biases (ShapeParams): The bias for the logistic function
#             scales (ShapeParams): The scale of the logistic function
#             is_right (bool, optional): Whether it is pointed right or left. Defaults to True.
#             m (torch.Tensor, optional): The max membership of the function. Defaults to None.
#         """
#         super().__init__(biases, scales, m)
#         self._is_right = is_right
    
#     def _on_side(self, x: torch.Tensor):
#         if self._is_right:
#             side = x >= self._biases.pt(0)
#         else: side = x <= self._biases.pt(0)
#         return side

#     def join(self, x: torch.Tensor) -> torch.Tensor:
#         """

#         Args:
#             x (torch.Tensor): The value to join with

#         Returns:
#             torch.Tensor: The membership
#         """
#         x = unsqueeze(x)
#         return logistic(
#             x, self._m, self._biases.pt(0), self.sigma
#         ) * self._on_side(x).float()

#         # return calc_m_logistic(
#         #     x, self._biases.pt(0), 
#         #     self._scales.pt(0), self._m
#         # ) * self._on_side(x).float()

#     def _calc_areas(self) -> torch.Tensor:
#         """Calculates the area of each section and sums it up

#         Returns:
#             torch.Tensor: The area of the trapezoid
#         """
#         return self._resize_to_m(
#             logistic_area(self._m, self.sigma) / 2.0, self._m
#         )
#         # return self._resize_to_m(2 * self._m / self._biases.pt(0), self._m)

#     def _calc_mean_cores(self) -> torch.Tensor:
#         """
#         Returns:
#             torch.Tensor: the mode of the curve
#         """
#         return self._resize_to_m(self._biases.pt(0), self._m)

#     def _calc_centroids(self):
#         """
#         Returns:
#             torch.Tensor: The centroid of the curve
#         """

#         if self._is_right:
#             centroid = logistic_area_up_to_a_inv(
#                 0.75, self._m, self._biases.pt(0), self.sigma
#             )
#         else:
#             centroid = logistic_area_up_to_a_inv(
#                 0.25, self._m, self._biases.pt(0), self.sigma
#             )
#         centroid = self._resize_to_m(
#             centroid, self._m
#         )
#         return centroid

#         # base_y = 0.75 if self._is_right else 0.25
#         # x = torch.logit(torch.tensor(base_y, dtype=torch.float, device=self._m.device)) / self._scales.pt(0) + self._biases.pt(0)
#         # return self._resize_to_m(x, self._m)

#     def scale(self, m: torch.Tensor) -> 'RightLogistic':
#         """Update the vertical scale of the right logistic

#         Args:
#             m (torch.Tensor): The new vertical scale

#         Returns:
#             RightLogistic: The updated vertical scale if the scale is greater
#         """
#         updated_m = functional.inter(self._m, m)
        
#         return RightLogistic(
#             self._biases, self._scales, self._is_right, updated_m
#         )

#     def truncate(self, m: torch.Tensor) -> 'LogisticTrapezoid':
#         """Truncate the right logistic. This requires the points be recalculated

#         Args:
#             m (torch.Tensor): The new maximum value

#         Returns:
#             LogisticTrapezoid: The logistic with the top truncated
#         """
#         truncated_m = functional.inter(self._m, m)
#         return RightLogisticTrapezoid(
#             self._biases, self._scales, self._is_right, truncated_m, self._m
#         )

#     @classmethod
#     def from_combined(cls, params: torch.Tensor, is_right: bool=True,m: torch.Tensor=None):
#         # TODO: Check this and confirm
#         if params.dim() == 4:
#             return cls(params.sub(0), params.sub(1), is_right, m)
#         return cls(params.sub(0), params.sub(1), is_right, m)


# class RightLogisticTrapezoid(Logistic):
#     """A LogisticTrapezoid shaped membership function that contains only one side
#     """

#     def __init__(
#         self, biases: ShapeParams, scales: ShapeParams, is_right: bool, 
#         truncated_m: torch.Tensor=None, scaled_m: torch.Tensor=None
#     ):
#         """Create a RightLogistic shaped membership function that contains only one side

#         Args:
#             biases (ShapeParams): The bias for the logistic function
#             scales (ShapeParams): The scale of the logistic function
#             is_right (bool, optional): Whether it is pointed right or left. Defaults to True.
#             truncated_m (torch.Tensor, optional): The max membership of the function. Defaults to None.
#             scaled_m (torch.Tensor, optional): The scale of the membership function. Defaults to None.
#         """
#         super().__init__(biases, scales, scaled_m)

#         truncated_m = self._init_m(truncated_m, biases.device)
#         self._truncated_m = functional.inter(self._m, truncated_m)
#         # dx = unsqueeze(calc_dx_logistic(self._truncated_m, self._scales.pt(0), self._m))
#         # self._dx = ShapeParams(dx)
#         self._left = ShapeParams(unsqueeze(logistic_area_up_to_a_inv(
#             self._truncated_m, self._m, self._biases.pt(0), self.sigma
#         )))
#         self._right = ShapeParams(unsqueeze(2 * self._biases.pt(0) - self._left.pt(0)))
#         self._is_right = is_right
#         # self._direction = is_right * 2 - 1
#         # self._pts = ShapeParams(self._biases.x + self._direction * dx)

#     @property
#     def m(self):
#         return self._truncated_m

#     def _contains(self, x: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
#         """Check whether x is contained in the membership function

#         Args:
#             x (torch.Tensor): the value to check

#         Returns:
#             typing.Tuple[torch.Tensor, torch.Tensor]: Multipliers for whether the value is contained
#         """
#         if self._is_right:
#             square_contains = (x >= self._biases.pt(0)) & (x <= self._right.pt(0))
#             logistic_contains = x >= self._right.pt(0)
#         else:
#             square_contains = (x <= self._biases.pt(0)) & (x >= self._left.pt(0))
#             logistic_contains = x <= self._left.pt(0)
#         return square_contains.float(), logistic_contains.float()

#     def join(self, x: torch.Tensor) -> 'torch.Tensor':
#         """Join calculates the membership value for each section of right logistic and uses the maximimum value
#         as the value

#         Args:
#             x (torch.Tensor): The value to calculate the membership for

#         Returns:
#             torch.Tensor: The membership
#         """
#         x = unsqueeze(x)
        
#         square_contains, logistic_contains = self._contains(x)

#         m1 = logistic(
#             x, self._m, self._biases.pt(0), self.sigma
#         ) * logistic_contains
#         # m1 = calc_m_logistic(
#         #     x, self._biases.pt(0), self._scales.pt(0), self._m
#         # ) * logistic_contains
#         m2 = self._m * square_contains
#         return torch.max(m1, m2)

#     def _calc_areas(self):
#         """Calculates the area of each logistic and sum up

#         Returns:
#             torch.Tensor: The area of the trapezoid
#         """
#         a1 = logistic_area_up_to_a(
#             self._left.pt(0), self._m, self._biases.pt(0),
#             self.sigma
#         )
#         a2 = (self._biases.pt(0) - self._left.pt(0)) * self._m
#         return self._resize_to_m(
#             a1 + a2, self._m
#         )

#         # a1 = self._resize_to_m(calc_area_logistic_one_side(
#         #     self._pts.pt(0), self._biases.pt(0), self._scales.pt(0), 
#         #     self._m), self._m)
#         # a2 = 0.5 * (self._biases.pt(0) + self._pts.pt(0)) * self._m
#         # return self._resize_to_m(a1 + a2, self._m)

#     def _calc_mean_cores(self):
#         """
#         Returns:
#             torch.Tensor: the mean value of the top of the Trapezoid
#         """
#         return self._resize_to_m(0.5 * (self._biases.pt(0) + self._left.pt(0)), self._m) 

#     def _calc_centroids(self):
#         """
#         Returns:
#             torch.Tensor: The center of mass for the three sections of the trapezoid
#         """
#         # area up to "dx"

#         a1 = logistic_area_up_to_a(
#             self._left.pt(0), self._m, self._biases.pt(0),
#             self.sigma
#         )
#         a2 = (self._biases.pt(0) - self._left.pt(0)) * self._m
#         x1 = logistic_area_up_to_a_inv(
#             a1 / 2, self._m, self._biases.pt(0), self.sigma
#         )
#         x2 = (self._biases.pt(0) + self._left.pt(0)) / 2.0

#         centroid = (
#             a1 * x1 + a2 * x2
#         ) / (a1 + a2)
#         if self._is_right:
#             centroid = 2 * self._biases.pt(0) - centroid
#         return self._resize_to_m(
#             centroid, self._m
#         )

#         # p = torch.sigmoid(self._scales.pt(0) * (-self._dx.pt(0)))
#         # centroid_logistic = self._biases.pt(0) + torch.logit(p / 2) / self._scales.pt(0)
#         # centroid_square = self._biases.pt(0) - self._dx.pt(0) / 2

#         # centroid = (centroid_logistic * p + centroid_square * self._dx.pt(0)) / (p + self._dx.pt(0))
#         # if self._is_right:
#         #     return self._biases.pt(0) + self._biases.pt(0) - centroid
#         # return self._resize_to_m(centroid, self._m)

#     def scale(self, m: torch.Tensor) -> 'RightLogisticTrapezoid':
#         """Update the vertical scale of the logistic

#         Args:
#             m (torch.Tensor): The new vertical scale

#         Returns:
#             RightLogisticTrapezoid: The updated vertical scale if the scale is greater
#         """
#         updated_m = functional.inter(self._m, m)

#         # TODO: Confirm if this is correct
#         # I think it should be intersecting rather than multiplying
#         truncated_m = self._truncated_m * updated_m

#         return RightLogisticTrapezoid(
#             self._biases, self._scales, self._is_right, truncated_m, updated_m
#         )

#     def truncate(self, m: torch.Tensor) -> 'RightLogisticTrapezoid':
#         """Truncate the right logistic. This requires the points be recalculated

#         Args:
#             m (torch.Tensor): The new maximum value

#         Returns:
#             RightLogisticTrapezoid: The updated vertical scale if the scale is greater
#         """
#         truncated_m = functional.inter(self._truncated_m, m)
#         return RightLogisticTrapezoid(
#             self._biases, self._scales, self._is_right, truncated_m, self._m
#         )

#     @classmethod
#     def from_combined(cls, params: torch.Tensor, is_right: bool=True,m: torch.Tensor=None):

#         if params.dim() == 4:

#             return cls(params.sub(0), params.sub(1), is_right, m)
#         return cls(params.sub(0), params.sub(1), is_right, m)
