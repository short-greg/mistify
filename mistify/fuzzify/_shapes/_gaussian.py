# TODO: Implement

# # 1st party
# import typing

# # 3rd party
# import torch
# from torch import Tensor

# # local
# from ._base import ShapeParams, Nonmonotonic
# from ...utils import unsqueeze, check_contains
# from ._utils import calc_dx_logistic, calc_area_logistic_one_side, calc_m_logistic, calc_x_logistic
# from ... import _functional as functional

# class Gaussian(Nonmonotonic):

#     def __init__(
#         self, biases: ShapeParams, scales: ShapeParams, m: torch.Tensor=None
#     ):
#         """The base class for logistic distribution functions

#         Note: Don't need to sort for this because there is only one point per parameter

#         Args:
#             biases (ShapeParams): The bias of the distribution
#             scales (ShapeParams): The scale value for the distribution
#             m (torch.Tensor, optional): The max membership. Defaults to None.
#         """
#         super().__init__(
#             biases.n_variables,
#             biases.n_terms
#         )
#         self._m = self._init_m(m, biases.device)
#         self._biases = biases
#         self._scales = scales

    
#     @property
#     def biases(self) -> 'ShapeParams':
#         """
#         Returns:
#             ShapeParams: The bias values
#         """
#         return self._biases
    
#     @property
#     def scales(self) -> 'ShapeParams':
#         """
#         Returns:
#             ShapeParams: The scales
#         """
#         return self._scales
    

# class GaussianBell(Gaussian):
#     """Use the GaussianBell function as the membership function
#     """

#     def join(self, x: Tensor) -> Tensor:

#         scale = torch.nn.functional.softplus(self._scale)
#         return torch.exp(
#             -0.5 * ((x.unsqueeze(-1) - self._loc) / scale) ** 2
#         )

#     def _calc_areas(self):

#         return self._resize_to_m(4 * self._m / self._biases.pt(0), self._m)
        
#     def _calc_mean_cores(self):
#         return self._resize_to_m(self._biases.pt(0), self._m)

#     def _calc_centroids(self):
#         return self._resize_to_m(self._biases.pt(0), self._m)

#     def scale(self, m: torch.Tensor) -> 'GaussianBell':
#         """Scale the height of the GaussianBell

#         Args:
#             m (torch.Tensor): The new height

#         Returns:
#             GaussianBell: The updated GaussianBell
#         """
#         updated_m = functional.intersect(self._m, m)
#         return GaussianBell(
#             self._biases, self._scales, updated_m
#         )

#     def truncate(self, m: torch.Tensor) -> 'GaussianTrapezoid':
#         """Truncate the height of the GaussianBell

#         Args:
#             m (torch.Tensor): The new height

#         Returns:
#             GaussianBell: The updated GaussianBell
#         """
#         return GaussianTrapezoid(
#             self._biases, self._scales, m, self._m 
#         )


# class GaussianTrapezoid(Gaussian):
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
#             scaled_m (torch.Tensor, optional): The scale of the GaussianTrapezoid. Defaults to None.
#         """
#         super().__init__(biases, scales, scaled_m)

#         truncated_m = self._init_m(truncated_m, biases.device)
#         self._truncated_m = functional.inter(truncated_m, self._m)
        
#         dx = unsqueeze(calc_dx_logistic(self._truncated_m, self._scales.pt(0), self._m))
#         self._dx = ShapeParams(dx)
#         self._pts = ShapeParams(torch.concat([
#             self._biases.x - self._dx.x,
#             self._biases.x + self._dx.x
#         ], dim=dx.dim() - 1))

#     @property
#     def dx(self):
#         return self._dx
    
#     @property
#     def m(self) -> torch.Tensor:
#         return self._truncated_m

#     def join(self, x: torch.Tensor) -> 'torch.Tensor':
#         x = unsqueeze(x)
#         inside = check_contains(x, self._pts.pt(0), self._pts.pt(1)).float()
#         m1 = calc_m_logistic(x, self._biases.pt(0), self._scales.pt(0), self._m) * (1 - inside)
#         m2 = self._truncated_m * inside
#         return torch.max(m1, m2)

#     def _calc_areas(self):
#         # symmetrical so multiply by 2
#         return self._resize_to_m(2 * calc_area_logistic_one_side(
#             self._pts.pt(0), self._biases.pt(0), self._scales.pt(0), self._m
#         ), self._m)
        
#     def _calc_mean_cores(self):
#         return self._resize_to_m(self._biases.pt(0), self._m)

#     def _calc_centroids(self):
#         return self._resize_to_m(self._biases.pt(0), self._m)

#     def scale(self, m: torch.Tensor) -> 'GaussianTrapezoid':
#         updated_m = functional.inter(self._m, m)
#         # TODO: check if multiplication is correct
#         truncated_m = self._truncated_m * updated_m

#         return GaussianTrapezoid(
#             self._biases, self._scales, truncated_m, updated_m
#         )

#     def truncate(self, m: torch.Tensor) -> 'GaussianTrapezoid':
#         truncated_m = functional.inter(self._truncated_m, m)
#         return GaussianTrapezoid(
#             self._biases, self._scales, truncated_m, self._m
#         )



# class RightGaussian(Gaussian):
#     """A Gaussian shaped membership function that contains only one side
#     """
    
#     def __init__(
#         self, biases: ShapeParams, scales: ShapeParams, is_right: bool=True,
#         m: torch.Tensor= None
#     ):
#         """Create a Gaussian shaped membership function that contains only one side

#         Args:
#             biases (ShapeParams): The bias for the logistic function
#             scales (ShapeParams): The scale of the logistic function
#             is_right (bool, optional): Whether it is pointed right or left. Defaults to True.
#             m (torch.Tensor, optional): The max membership of the function. Defaults to None.
#         """
#         super().__init__(biases, scales, m)
#         self._is_right = is_right
#         self._direction = is_right * 2 - 1
    
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
#         return calc_m_logistic(
#             x, self._biases.pt(0), 
#             self._scales.pt(0), self._m
#         ) * self._on_side(x).float()

#     def _calc_areas(self) -> torch.Tensor:
#         """Calculates the area of each section and sums it up

#         Returns:
#             torch.Tensor: The area of the trapezoid
#         """
#         return self._resize_to_m(2 * self._m / self._biases.pt(0), self._m)

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
#         base_y = 0.75 if self._is_right else 0.25
#         x = torch.logit(torch.tensor(base_y, dtype=torch.float, device=self._m.device)) / self._scales.pt(0) + self._biases.pt(0)
#         return self._resize_to_m(x, self._m)

#     def scale(self, m: torch.Tensor) -> 'RightGaussian':
#         """Update the vertical scale of the right logistic

#         Args:
#             m (torch.Tensor): The new vertical scale

#         Returns:
#             RightGaussian: The updated vertical scale if the scale is greater
#         """
#         updated_m = functional.inter(self._m, m)
        
#         return RightGaussian(
#             self._biases, self._scales, self._is_right, updated_m
#         )

#     def truncate(self, m: torch.Tensor) -> 'GaussianTrapezoid':
#         """Truncate the right logistic. This requires the points be recalculated

#         Args:
#             m (torch.Tensor): The new maximum value

#         Returns:
#             GaussianTrapezoid: The logistic with the top truncated
#         """
#         truncated_m = functional.inter(self._m, m)
#         return RightGaussianTrapezoid(
#             self._biases, self._scales, self._is_right, truncated_m, self._m
#         )

#     @classmethod
#     def from_combined(cls, params: torch.Tensor, is_right: bool=True,m: torch.Tensor=None):
#         # TODO: Check this and confirm
#         if params.dim() == 4:
#             return cls(params.sub(0), params.sub(1), is_right, m)
#         return cls(params.sub(0), params.sub(1), is_right, m)


# class RightGaussianTrapezod(Gaussian):
#     """A GaussianTrapezoid shaped membership function that contains only one side
#     """

#     def __init__(
#         self, biases: ShapeParams, scales: ShapeParams, is_right: bool, 
#         truncated_m: torch.Tensor=None, scaled_m: torch.Tensor=None
#     ):
#         """Create a RightGaussian shaped membership function that contains only one side

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
#         dx = unsqueeze(calc_dx_logistic(self._truncated_m, self._scales.pt(0), self._m))
#         self._dx = ShapeParams(dx)
#         self._is_right = is_right
#         self._direction = is_right * 2 - 1
#         self._pts = ShapeParams(self._biases.x + self._direction * dx)

#     @property
#     def dx(self):
#         return self._dx

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
#             square_contains = (x >= self._biases.pt(0)) & (x <= self._pts.pt(0))
#             logistic_contains = x >= self._pts.pt(0)
#         else:
#             square_contains = (x <= self._biases.pt(0)) & (x >= self._pts[0])
#             logistic_contains = x <= self._pts.pt(0)
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
        
#         m1 = calc_m_logistic(
#             x, self._biases.pt(0), self._scales.pt(0), self._m
#         ) * logistic_contains
#         m2 = self._m * square_contains
#         return torch.max(m1, m2)

#     def _calc_areas(self):
#         """Calculates the area of each logistic and sum up

#         Returns:
#             torch.Tensor: The area of the trapezoid
#         """
#         a1 = self._resize_to_m(calc_area_logistic_one_side(
#             self._pts.pt(0), self._biases.pt(0), self._scales.pt(0), 
#             self._m), self._m)
#         a2 = 0.5 * (self._biases.pt(0) + self._pts.pt(0)) * self._m
#         return self._resize_to_m(a1 + a2, self._m)

#     def _calc_mean_cores(self):
#         """
#         Returns:
#             torch.Tensor: the mean value of the top of the Trapezoid
#         """
#         return self._resize_to_m(0.5 * (self._biases.pt(0) + self._pts.pt(0)), self._m) 

#     def _calc_centroids(self):
#         """
#         Returns:
#             torch.Tensor: The center of mass for the three sections of the trapezoid
#         """
#         # area up to "dx"
#         p = torch.sigmoid(self._scales.pt(0) * (-self._dx.pt(0)))
#         centroid_logistic = self._biases.pt(0) + torch.logit(p / 2) / self._scales.pt(0)
#         centroid_square = self._biases.pt(0) - self._dx.pt(0) / 2

#         centroid = (centroid_logistic * p + centroid_square * self._dx.pt(0)) / (p + self._dx.pt(0))
#         if self._is_right:
#             return self._biases.pt(0) + self._biases.pt(0) - centroid
#         return self._resize_to_m(centroid, self._m)

#     def scale(self, m: torch.Tensor) -> 'RightGaussianTrapezoid':
#         """Update the vertical scale of the logistic

#         Args:
#             m (torch.Tensor): The new vertical scale

#         Returns:
#             RightGaussianTrapezoid: The updated vertical scale if the scale is greater
#         """
#         updated_m = functional.inter(self._m, m)

#         # TODO: Confirm if this is correct
#         # I think it should be intersecting rather than multiplying
#         truncated_m = self._truncated_m * updated_m

#         return RightGaussianTrapezoid(
#             self._biases, self._scales, self._is_right, truncated_m, updated_m
#         )

#     def truncate(self, m: torch.Tensor) -> 'RightGaussianTrapezoid':
#         """Truncate the right logistic. This requires the points be recalculated

#         Args:
#             m (torch.Tensor): The new maximum value

#         Returns:
#             RightGaussianTrapezoid: The updated vertical scale if the scale is greater
#         """
#         truncated_m = functional.inter(self._truncated_m, m)
#         return RightGaussianTrapezoid(
#             self._biases, self._scales, self._is_right, truncated_m, self._m
#         )

#     @classmethod
#     def from_combined(cls, params: torch.Tensor, is_right: bool=True,m: torch.Tensor=None):

#         if params.dim() == 4:

#             return cls(params.sub(0), params.sub(1), is_right, m)
#         return cls(params.sub(0), params.sub(1), is_right, m)




#     # # updated the formula so need to update here
#     # def integral(self, x: torch.Tensor):

#     #     return self._scale * torch.tensor(-torch.pi / 2.0, device=x.device) * (
#     #         torch.erf((self._loc - x) / (self._scale * torch.sqrt(torch.tensor(2.0))) 
#     #         ))

#     # def hypo(self, m: torch.Tensor) -> HypoM:
        
#     #     # get the lower bound
#     #     inv = torch.sqrt(-torch.log(m) * (2 * self._scale ** 2))
#     #     lhs = -inv + self._loc
#     #     rhs = inv + self._loc
#     #     sum_left = self.integral(lhs)

#     #     x = -torch.sqrt(-torch.log(m))

#     #     sum_left = (torch.sqrt(torch.tensor(torch.pi)) /  2 ) * (
#     #         (1 + torch.erf(x))
#     #     )
#     #     sum_rec = (rhs - lhs) * m
#     #     return HypoM(
#     #         sum_left * 2 + sum_rec, m
#     #     )


#     # def join(self, m: Tensor) -> Tensor:

#     #     scale = torch.nn.functional.softplus(self._scale)
#     #     return torch.exp(
#     #         -0.5 * ((m.unsqueeze(-1) - self._loc) / scale) ** 2
#     #     )
    


#     # def _calc_areas(self):
#     #     return super()._calc_areas()
    
#     # def _calc_centroids(self) -> Tensor:
#     #     return super()._calc_centroids()
    
#     # def _calc_mean_cores(self) -> Tensor:
#     #     return super()._calc_mean_cores()


#     # def defuzzify(self, x, m):
#     #     """
#     #     Defuzzify the membership tensor using the Center of Sums method.
#     #     :param x: Input tensor that was fuzzified.
#     #     :param m: Membership tensor resulting from fuzzification.
#     #     :return: Defuzzified value using the Center of Sums method.
#     #     """
#     #     # Calculate the weighted sum of the input values, using membership values as weights
#     #     weighted_sum = torch.sum(x * m)
#     #     # Sum of the membership values
#     #     sum_of_memberships = torch.sum(m)
#     #     # Compute the weighted average (center of sums)
#     #     if sum_of_memberships > 0:
#     #         cos_value = weighted_sum / sum_of_memberships
#     #     else:
#     #         cos_value = torch.tensor(0.0)  # Fallback in case of zero division
#     #     return cos_value
    
#     # def tunable(self, tunable: bool=True):

#     #     for p in self.parameters():
#     #         p.requires_grad_(tunable)

#     # def resp_loss(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        
#     #     # assume that each of the components has some degree of
#     #     # responsibility
#     #     t = t.clamp(min=0.0) + torch.rand_like(t) * 1e-6
#     #     r = t / t.sum(dim=-1, keepdim=True)
#     #     Nk = r.sum(dim=0, keepdim=True)
#     #     target_loc = (r * x[:,:,None]).sum(dim=0, keepdim=True) / Nk

#     #     target_scale = (r * (x[:,:,None] - target_loc) ** 2).sum(dim=0, keepdim=True) / Nk

#     #     cur_scale = torch.nn.functional.softplus(self._scale)
        
#     #     scale_loss = self._fit_loss(cur_scale, target_scale.detach())
#     #     loc_loss = self._fit_loss(self._loc, target_loc.detach())
#     #     return scale_loss + loc_loss
