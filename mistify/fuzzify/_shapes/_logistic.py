# 1st party
import typing

# 3rd party
import torch

# local
from ._base import ShapeParams, Nonmonotonic
from ...utils import unsqueeze, check_contains
from ._utils import calc_dx_logistic, calc_area_logistic_one_side, calc_m_logistic, calc_x_logistic
from ... import _functional as functional


class Logistic(Nonmonotonic):
    """A logistic bell curve based on the shape of the logistic distribution
    but normalized so the maximum value is 1 by default.
    """

    def __init__(
        self, biases: ShapeParams, scales: ShapeParams, m: torch.Tensor=None
    ):
        """The base class for logistic distribution functions

        Note: Don't need to sort for this because there is only one point per parameter

        Args:
            biases (ShapeParams): The bias of the distribution
            scales (ShapeParams): The scale value for the distribution
            m (torch.Tensor, optional): The max membership. Defaults to None.
        """
        super().__init__(
            biases.n_variables,
            biases.n_terms
        )
        self._m = self._init_m(m, biases.device)
        self._biases = biases
        self._scales = scales

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
    """Use the LogisticBell function as the membership function
    """

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
        """Scale the height of the LogisticBell

        Args:
            m (torch.Tensor): The new height

        Returns:
            LogisticBell: The updated LogisticBell
        """
        updated_m = functional.inter(self._m, m)
        return LogisticBell(
            self._biases, self._scales, updated_m
        )

    def truncate(self, m: torch.Tensor) -> 'LogisticTrapezoid':
        """Truncate the height of the LogisticBell

        Args:
            m (torch.Tensor): The new height

        Returns:
            LogisticBell: The updated LogisticBell
        """
        return LogisticTrapezoid(
            self._biases, self._scales, m, self._m 
        )


class LogisticTrapezoid(Logistic):
    """A membership function that has a ceiling on the heighest value
    """
    
    def __init__(
        self, biases: ShapeParams, scales: ShapeParams, 
        truncated_m: torch.Tensor=None, scaled_m: torch.Tensor=None
    ):
        """Create a membership function that has a ceiling on the heighest value

        Note: Don't need to sort for this because it is derived

        Args:
            biases (ShapeParams): The biases for the logistic part of the funciton
            scales (ShapeParams): The scales for the logistic part of teh function
            truncated_m (torch.Tensor, optional): The maximum height of the membership. Defaults to None.
            scaled_m (torch.Tensor, optional): The scale of the LogisticTrapezoid. Defaults to None.
        """
        super().__init__(biases, scales, scaled_m)

        truncated_m = self._init_m(truncated_m, biases.device)
        self._truncated_m = functional.inter(truncated_m, self._m)
        
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
        updated_m = functional.inter(self._m, m)
        # TODO: check if multiplication is correct
        truncated_m = self._truncated_m * updated_m

        return LogisticTrapezoid(
            self._biases, self._scales, truncated_m, updated_m
        )

    def truncate(self, m: torch.Tensor) -> 'LogisticTrapezoid':
        truncated_m = functional.inter(self._truncated_m, m)
        return LogisticTrapezoid(
            self._biases, self._scales, truncated_m, self._m
        )


class RightLogistic(Logistic):
    """A Logistic shaped membership function that contains only one side
    """
    
    def __init__(
        self, biases: ShapeParams, scales: ShapeParams, is_right: bool=True,
        m: torch.Tensor= None
    ):
        """Create a Logistic shaped membership function that contains only one side

        Args:
            biases (ShapeParams): The bias for the logistic function
            scales (ShapeParams): The scale of the logistic function
            is_right (bool, optional): Whether it is pointed right or left. Defaults to True.
            m (torch.Tensor, optional): The max membership of the function. Defaults to None.
        """
        super().__init__(biases, scales, m)
        self._is_right = is_right
        self._direction = is_right * 2 - 1
    
    def _on_side(self, x: torch.Tensor):
        if self._is_right:
            side = x >= self._biases.pt(0)
        else: side = x <= self._biases.pt(0)
        return side

    def join(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): The value to join with

        Returns:
            torch.Tensor: The membership
        """
        x = unsqueeze(x)
        return calc_m_logistic(
            x, self._biases.pt(0), 
            self._scales.pt(0), self._m
        ) * self._on_side(x).float()

    def _calc_areas(self) -> torch.Tensor:
        """Calculates the area of each section and sums it up

        Returns:
            torch.Tensor: The area of the trapezoid
        """
        return self._resize_to_m(2 * self._m / self._biases.pt(0), self._m)

    def _calc_mean_cores(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: the mode of the curve
        """
        return self._resize_to_m(self._biases.pt(0), self._m)

    def _calc_centroids(self):
        """
        Returns:
            torch.Tensor: The centroid of the curve
        """
        base_y = 0.75 if self._is_right else 0.25
        x = torch.logit(torch.tensor(base_y, dtype=torch.float, device=self._m.device)) / self._scales.pt(0) + self._biases.pt(0)
        return self._resize_to_m(x, self._m)

    def scale(self, m: torch.Tensor) -> 'RightLogistic':
        """Update the vertical scale of the right logistic

        Args:
            m (torch.Tensor): The new vertical scale

        Returns:
            RightLogistic: The updated vertical scale if the scale is greater
        """
        updated_m = functional.inter(self._m, m)
        
        return RightLogistic(
            self._biases, self._scales, self._is_right, updated_m
        )

    def truncate(self, m: torch.Tensor) -> 'LogisticTrapezoid':
        """Truncate the right logistic. This requires the points be recalculated

        Args:
            m (torch.Tensor): The new maximum value

        Returns:
            LogisticTrapezoid: The logistic with the top truncated
        """
        truncated_m = functional.inter(self._m, m)
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
    """A LogisticTrapezoid shaped membership function that contains only one side
    """

    def __init__(
        self, biases: ShapeParams, scales: ShapeParams, is_right: bool, 
        truncated_m: torch.Tensor=None, scaled_m: torch.Tensor=None
    ):
        """Create a RightLogistic shaped membership function that contains only one side

        Args:
            biases (ShapeParams): The bias for the logistic function
            scales (ShapeParams): The scale of the logistic function
            is_right (bool, optional): Whether it is pointed right or left. Defaults to True.
            truncated_m (torch.Tensor, optional): The max membership of the function. Defaults to None.
            scaled_m (torch.Tensor, optional): The scale of the membership function. Defaults to None.
        """
        super().__init__(biases, scales, scaled_m)

        truncated_m = self._init_m(truncated_m, biases.device)
        self._truncated_m = functional.inter(self._m, truncated_m)
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

    def _contains(self, x: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """Check whether x is contained in the membership function

        Args:
            x (torch.Tensor): the value to check

        Returns:
            typing.Tuple[torch.Tensor, torch.Tensor]: Multipliers for whether the value is contained
        """
        if self._is_right:
            square_contains = (x >= self._biases.pt(0)) & (x <= self._pts.pt(0))
            logistic_contains = x >= self._pts.pt(0)
        else:
            square_contains = (x <= self._biases.pt(0)) & (x >= self._pts[0])
            logistic_contains = x <= self._pts.pt(0)
        return square_contains.float(), logistic_contains.float()

    def join(self, x: torch.Tensor) -> 'torch.Tensor':
        """Join calculates the membership value for each section of right logistic and uses the maximimum value
        as the value

        Args:
            x (torch.Tensor): The value to calculate the membership for

        Returns:
            torch.Tensor: The membership
        """
        x = unsqueeze(x)
        
        square_contains, logistic_contains = self._contains(x)
        
        m1 = calc_m_logistic(
            x, self._biases.pt(0), self._scales.pt(0), self._m
        ) * logistic_contains
        m2 = self._m * square_contains
        return torch.max(m1, m2)

    def _calc_areas(self):
        """Calculates the area of each logistic and sum up

        Returns:
            torch.Tensor: The area of the trapezoid
        """
        a1 = self._resize_to_m(calc_area_logistic_one_side(
            self._pts.pt(0), self._biases.pt(0), self._scales.pt(0), 
            self._m), self._m)
        a2 = 0.5 * (self._biases.pt(0) + self._pts.pt(0)) * self._m
        return self._resize_to_m(a1 + a2, self._m)

    def _calc_mean_cores(self):
        """
        Returns:
            torch.Tensor: the mean value of the top of the Trapezoid
        """
        return self._resize_to_m(0.5 * (self._biases.pt(0) + self._pts.pt(0)), self._m) 

    def _calc_centroids(self):
        """
        Returns:
            torch.Tensor: The center of mass for the three sections of the trapezoid
        """
        # area up to "dx"
        p = torch.sigmoid(self._scales.pt(0) * (-self._dx.pt(0)))
        centroid_logistic = self._biases.pt(0) + torch.logit(p / 2) / self._scales.pt(0)
        centroid_square = self._biases.pt(0) - self._dx.pt(0) / 2

        centroid = (centroid_logistic * p + centroid_square * self._dx.pt(0)) / (p + self._dx.pt(0))
        if self._is_right:
            return self._biases.pt(0) + self._biases.pt(0) - centroid
        return self._resize_to_m(centroid, self._m)

    def scale(self, m: torch.Tensor) -> 'RightLogisticTrapezoid':
        """Update the vertical scale of the logistic

        Args:
            m (torch.Tensor): The new vertical scale

        Returns:
            RightLogisticTrapezoid: The updated vertical scale if the scale is greater
        """
        updated_m = functional.inter(self._m, m)

        # TODO: Confirm if this is correct
        # I think it should be intersecting rather than multiplying
        truncated_m = self._truncated_m * updated_m

        return RightLogisticTrapezoid(
            self._biases, self._scales, self._is_right, truncated_m, updated_m
        )

    def truncate(self, m: torch.Tensor) -> 'RightLogisticTrapezoid':
        """Truncate the right logistic. This requires the points be recalculated

        Args:
            m (torch.Tensor): The new maximum value

        Returns:
            RightLogisticTrapezoid: The updated vertical scale if the scale is greater
        """
        truncated_m = functional.inter(self._truncated_m, m)
        return RightLogisticTrapezoid(
            self._biases, self._scales, self._is_right, truncated_m, self._m
        )

    @classmethod
    def from_combined(cls, params: torch.Tensor, is_right: bool=True,m: torch.Tensor=None):

        if params.dim() == 4:

            return cls(params.sub(0), params.sub(1), is_right, m)
        return cls(params.sub(0), params.sub(1), is_right, m)
