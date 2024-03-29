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


def logistic_area(scale):
    
    return 4 * scale


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
    return (pts[0] + pts[1]) / 2.0


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
        return half_logistic_area(self._biases.pt(0), self.sigma, m)
    
    def mean_cores(self, m: Tensor, truncate: bool = False) -> Tensor:
        
        if truncate:
            return truncated_half_logistic_mean_core(self._biases.pt(0), self.sigma, m, self.increasing)
        return self._resize_to_m(self._biases.pt(0), m)
    
    def centroids(self, m: Tensor, truncate: bool = False) -> Tensor:
        
        if truncate:
            return truncated_half_logistic_centroid(self._biases.pt(0), self.sigma, m, self.increasing)
        return half_logistic_centroid(self._biases.pt(0), self.sigma, m, self.increasing)


