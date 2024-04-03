# TODO: Implement

# 1st party
import typing
import math

# 3rd party
import torch
from torch import Tensor

# local
from ._base import ShapeParams, Nonmonotonic
from ...utils import unsqueeze
import torch.nn.functional

# TODO: Review and update
# Try to get simpler functions


def gaussian_area(scale: torch.Tensor) -> torch.Tensor:
    """
    Args:
        scale (torch.Tensor): The scale paramter (deviation) for the Gaussian

    Returns:
        torch.Tensor: the area of the Gaussian
    """
    return math.sqrt(2 * torch.pi) * scale


def gaussian_invert(y: torch.Tensor, bias: torch.Tensor, scale: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """Invert the Gaussian

    Args:
        y (torch.Tensor): The output of the Gaussian
        bias (torch.Tensor): The bias of the Gaussian
        scale (torch.Tensor): The scale (deviation) of the Gaussian

    Returns:
        typing.Tuple[torch.Tensor, torch.Tensor]: The two possible inverses for the Gaussian (lhs and rhs)
    """
    base = torch.sqrt(-2 * scale ** 2 * torch.log(y + 1e-7))
    return bias - base, bias + base


def gaussian_area_up_to(x: torch.Tensor, bias: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Calculate the area up to a value

    Args:
        x (torch.Tensor): The value to calculate the area up to
        bias (torch.Tensor): The bias of the Gaussian
        scale (torch.Tensor): The scale (deviation) of the Gaussian

    Returns:
        torch.Tensor: The area
    """
    return math.sqrt(math.pi / 2) * (
        1 / torch.sqrt(scale ** -2) - scale * torch.erf(
            (bias - x) / (math.sqrt(2) * scale)
        )
    )


def gaussian_area_up_to_inv(area: torch.Tensor, bias: torch.Tensor, scale: torch.Tensor, increasing: bool=True) -> torch.Tensor:
    """Invert the area function of the Gaussian

    Args:
        area (torch.Tensor): The area to invert
        bias (torch.Tensor): The bias of the Gaussian
        scale (torch.Tensor): The scale (deviation) of the Gaussian 
        increasing (bool, optional): Whether it is increasing or decreasing. Defaults to True.

    Returns:
        torch.Tensor: The x value
    """
    left = -area * math.sqrt(2 * math.pi)
    right = math.pi / torch.sqrt(scale ** -2)
    dx = math.sqrt(2.) * scale * torch.erfinv(
        (left + right) / (math.pi * scale)
    )
    return bias - dx if increasing else bias + dx


def gaussian(x: torch.Tensor, bias: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """The Gaussian function

    Args:
        x (torch.Tensor): The input
        bias (torch.Tensor): The bias of the Gaussian
        scale (torch.Tensor): The scale (deviation) of the Gaussian

    Returns:
        torch.Tensor: The output of the Gaussian
    """
    return torch.exp(-((x - bias) ** 2) / (2 * (scale **2)))


def truncated_gaussian_area(bias: torch.Tensor, std: torch.Tensor, height: torch.Tensor) -> torch.Tensor:
    """

    Args:
        bias (torch.Tensor): The bias of the Gaussian
        std (torch.Tensor): The deviation of the Gaussian
        height (torch.Tensor): The height of the truncation

    Returns:
        torch.Tensor: The area of the truncated Gaussian
    """
    pts = gaussian_invert(height, bias, std)
    rec_area = (pts[1] - pts[0]) * height
    gauss_area = gaussian_area_up_to(pts[0], bias, std)
    return rec_area + 2 * gauss_area


def truncated_gaussian_mean_core(bias: torch.Tensor, scale: torch.Tensor, height: torch.Tensor) -> torch.Tensor:
    """
    Args:
        bias (torch.Tensor): The bias of the Gaussian
        scale (torch.Tensor): The scale (deviation) of the Gaussian
        height (torch.Tensor): The height of the Gaussian

    Returns:
        torch.Tensor: The mean of the core of the Gaussian
    """
    pts = gaussian_invert(height, bias, scale)
    return (pts[0] + pts[1]) / 2.0


def half_gaussian_area(scale: torch.Tensor) -> torch.Tensor:
    return gaussian_area(scale) / 2.0


def half_gaussian_centroid(bias: torch.Tensor, std: torch.Tensor, height: torch.Tensor, increasing: bool=True) -> torch.Tensor:

    return gaussian_area_up_to_inv(height, bias, std, increasing)


def truncated_half_gaussian_area(bias: torch.Tensor, std: torch.Tensor, height: torch.Tensor) -> torch.Tensor:
    pts = gaussian_invert(height, bias, std)
    rec_area = (bias - pts[0]) * height
    gauss_area = gaussian_area_up_to(pts[0], bias, std)
    return gauss_area + rec_area


def truncated_half_gaussian_mean_core(bias: torch.Tensor, std: torch.Tensor, height: torch.Tensor, increasing: bool=True) -> torch.Tensor:    
    pts = gaussian_invert(height, bias, std)
    if increasing:
        return pts[0]
    return pts[1]


def truncated_half_gaussian_centroid(bias: torch.Tensor, std: torch.Tensor, height: torch.Tensor, increasing: bool=True) -> torch.Tensor:
    pts = gaussian_invert(height, bias, std)
    rec_area = (bias - pts[0]) * height
    gauss_area = gaussian_area_up_to(pts[1], bias, std)
    gauss_centroid = gaussian_area_up_to_inv(gauss_area / 2.0, bias, std, increasing)
    rec_centroid = (bias + pts[0]) / 2.0 if increasing else (bias + pts[1]) / 2.0

    return (rec_centroid * rec_area + gauss_centroid * gauss_area) / (rec_area + gauss_area)


class Gaussian(Nonmonotonic):

    def __init__(
        self, biases: ShapeParams, scales: ShapeParams
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
    def from_combined(cls, params: ShapeParams) -> 'Gaussian':
        """Create the shape from 

        Returns:
            Logistic: The logistic distribution function 
        """
        return cls(
            params.sub((0, 1)), 
            params.sub((1, 2))
        )


class GaussianBell(Gaussian):
    """Use the GaussianBell function as the membership function
    """

    def __init__(self, biases: ShapeParams, scales: ShapeParams, hookf=None):
        """Use a Gaussian as the membership function

        Args:
            biases (ShapeParams): The biases for the Gaussian
            scales (ShapeParams): The scale parameter for the Gaussian (deviation)
            hookf (_type_, optional): The GradHook for the join function. Defaults to None.
        """
        super().__init__(biases, scales)
        self.joinf = gaussian if hookf is None else hookf.f(gaussian)

    def join(self, x: Tensor) -> Tensor:
        return self.joinf(
            unsqueeze(x), self._biases.pt(0), self.sigma
        )
    
    def areas(self, m: Tensor, truncate: bool = False) -> Tensor:
        
        if truncate:
            return self._resize_to_m(
                truncated_gaussian_area(self._biases.pt(0), self._scales.pt(0), m),
                m)
        return self._resize_to_m(
            gaussian_area(self._scales.pt(0)), m
        )
    
    def mean_cores(self, m: Tensor, truncate: bool = False) -> Tensor:
        
        if truncate:
            return self._resize_to_m(
                truncated_gaussian_mean_core(self._biases.pt(0), self._scales.pt(0), m), m
            )
        return self._resize_to_m(self._biases.pt(0), m)
    
    def centroids(self, m: Tensor, truncate: bool = False) -> Tensor:
        
        return self._resize_to_m(self._biases.pt(0), m)
    

class HalfGaussianBell(Gaussian):
    """Use the GaussianBell function as the membership function
    """
    def __init__(self, biases: ShapeParams, scales: ShapeParams, increasing: torch.Tensor):
        super().__init__(biases, scales)
        self.increasing = increasing

    def join(self, x: Tensor) -> Tensor:

        x = unsqueeze(x)
        if self.increasing:
            contains = (x <= self._biases.pt(0))
        else:
            contains = (x >= self._biases.pt(0))

        return gaussian(
            x, self._biases.pt(0), self.sigma
        ) * contains
    
    def areas(self, m: Tensor, truncate: bool = False) -> Tensor:
        
        if truncate:
            return truncated_half_gaussian_area(self._biases.pt(0), self.sigma, m)
        return self._resize_to_m(half_gaussian_area(self.sigma), m)
    
    def mean_cores(self, m: Tensor, truncate: bool = False) -> Tensor:
        
        if truncate:
            return self._resize_to_m(
                truncated_half_gaussian_mean_core(
                    self._biases.pt(0), self.sigma, m, self.increasing
                ), m)
        return self._resize_to_m(self._biases.pt(0), m)
    
    def centroids(self, m: Tensor, truncate: bool = False) -> Tensor:
        
        if truncate:
            return truncated_half_gaussian_centroid(self._biases.pt(0), self.sigma, m, self.increasing)
        return half_gaussian_centroid(self._biases.pt(0), self.sigma, m, self.increasing)
