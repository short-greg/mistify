# 1st party
import typing

# 3rd party
import torch
import torch.nn.functional
from torch import Tensor

# local
from ._base import ShapeParams, Nonmonotonic
from ...utils import unsqueeze


def logistic_area(scale: torch.Tensor) -> torch.Tensor:
    """
    Args:
        scale (torch.Tensor): The scale of the logistic

    Returns:
        torch.Tensor: The area
    """
    return scale * 4


def logistic_invert(y: torch.Tensor, bias: torch.Tensor, scale: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """

    Args:
        y (torch.Tensor): The output
        bias (torch.Tensor): The bias
        scale (torch.Tensor): The scale 

    Returns:
        typing.Tuple[torch.Tensor, torch.Tensor]: Both values from invert
    """
    y = torch.clamp(y, 1e-7, 1.0)
    s = 2 * torch.sqrt(1 - y) + 2
    print(y, s)
    x2 = torch.log((-y + s) / y)
    return bias - scale * x2, bias + scale * x2


def logistic_area_up_to(x: torch.Tensor, bias: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x (torch.Tensor): The value to calculate the area up to
        bias (torch.Tensor): The bias for the logistic
        scale (torch.Tensor): The scale

    Returns:
        torch.Tensor: 
    """
    # TODO: presently incro
    z = (x - bias) / scale
    return torch.sigmoid(z) * scale * 4


def logistic_area_up_to_inv(y: torch.Tensor, bias: torch.Tensor, scale: torch.Tensor, increasing: bool=True) -> torch.Tensor:
    """
    Args:
        y (torch.Tensor): The area to calculate the x for
        bias (torch.Tensor): The bias
        scale (torch.Tensor): The scale of the logistic distribution
        increasing (bool, optional): Whether increasing or decreasing. Defaults to True.

    Returns:
        torch.Tensor: The x value outputting that area
    """
    dx = torch.logit(y / scale) * scale
    
    if increasing:
        return bias - dx
    return bias + dx


def logistic(x: torch.Tensor,  bias: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x (torch.Tensor): The x value
        bias (torch.Tensor): The bias of the logistic
        scale (torch.Tensor): The scale of the logistic

    Returns:
        torch.Tensor: the output fo the logistic
    """
    sig = torch.sigmoid(-(x - bias) / scale)
    return 4  * (1 - sig) * sig


def truncated_logistic_area(bias: torch.Tensor, std: torch.Tensor, height: torch.Tensor) -> torch.Tensor:
    
    pts = logistic_invert(height, bias, std)
    rec_area = (pts[1] - pts[0]) * height
    logistic_area = logistic_area_up_to(pts[0], bias, std)
    return rec_area + 2 * logistic_area


def truncated_logistic_mean_core(bias: torch.Tensor, std: torch.Tensor, height: torch.Tensor) -> torch.Tensor:
    pts = logistic_invert(height, bias, std)
    return (pts[0] + pts[1]) / 2.0


def half_logistic_area(scale: torch.Tensor) -> torch.Tensor:
    return scale * 2


def half_logistic_centroid(bias: torch.Tensor, scale: torch.Tensor, height: torch.Tensor, increasing: bool=True) -> torch.Tensor:

    return logistic_area_up_to_inv(height, bias, scale, increasing)


def truncated_half_logistic_area(bias: torch.Tensor, scale: torch.Tensor, height: torch.Tensor) -> torch.Tensor:
    pts = logistic_invert(height, bias, scale)
    rec_area = (bias - pts[0]) * height
    gauss_area = logistic_area_up_to(pts[0], bias, scale)
    return gauss_area + rec_area


def truncated_half_logistic_mean_core(bias: torch.Tensor, scale: torch.Tensor, height: torch.Tensor, increasing: bool=True) -> torch.Tensor:    
    pts = logistic_invert(height, bias, scale)
    if increasing:
        return pts[0]
    return pts[1]


def truncated_half_logistic_centroid(bias: torch.Tensor, scale: torch.Tensor, height: torch.Tensor, increasing: bool=True) -> torch.Tensor:
    pts = logistic_invert(height, bias, scale)
    rec_area = (bias - pts[0]) * height
    gauss_area = logistic_area_up_to(pts[0], bias, scale)
    gauss_centroid = logistic_area_up_to_inv(gauss_area / 2.0, bias, scale, increasing)
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
    def from_combined(cls, params: ShapeParams) -> 'Logistic':
        """Create the shape from 

        Returns:
            Logistic: The logistic distribution function 
        """
        return cls(
            params.sub((0, 1)), 
            params.sub((1, 2))
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
            return truncated_logistic_area(self._biases.pt(0), self.sigma, m)
        return self._resize_to_m(logistic_area(self.sigma), m)
    
    def mean_cores(self, m: Tensor, truncate: bool = False) -> Tensor:
        
        if truncate:
            return self._resize_to_m(
                truncated_logistic_mean_core(self._biases.pt(0), self.sigma, m), m
            )
        return self._resize_to_m(self._biases.pt(0), m)
    
    def centroids(self, m: Tensor, truncate: bool = False) -> Tensor:
        
        return self._resize_to_m(self._biases.pt(0), m)
    

class HalfLogisticBell(Logistic):
    """Use the Half Logistic Bell function as the membership function
    """
    def __init__(self, biases: ShapeParams, scales: ShapeParams, increasing: bool=True):
        super().__init__(biases, scales)
        self.increasing = increasing

    def join(self, x: Tensor) -> Tensor:
        x = unsqueeze(x)
        if self.increasing:
            contains = (x <= self._biases.pt(0))
        else:
            contains = (x >= self._biases.pt(0))

        return logistic(
            x, self._biases.pt(0), self.sigma
        ) * contains
    
    def areas(self, m: Tensor, truncate: bool = False) -> Tensor:
        
        if truncate:
            return truncated_half_logistic_area(self._biases.pt(0), self.sigma, m)
        return self._resize_to_m(half_logistic_area(self.sigma), m)
    
    def mean_cores(self, m: Tensor, truncate: bool = False) -> Tensor:
        
        if truncate:
            return truncated_half_logistic_mean_core(self._biases.pt(0), self.sigma, m, self.increasing)
        return self._resize_to_m(self._biases.pt(0), m)
    
    def centroids(self, m: Tensor, truncate: bool = False) -> Tensor:
        
        if truncate:
            return truncated_half_logistic_centroid(self._biases.pt(0), self.sigma, m, self.increasing)
        return half_logistic_centroid(self._biases.pt(0), self.sigma, m, self.increasing)
