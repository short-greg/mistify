# TODO: Implement

# 1st party
import typing
import math

# 3rd party
import torch
from torch import Tensor
import torch.nn.functional

# local
from ._base import Coords, Nonmonotonic
from ...utils import unsqueeze


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
        self, biases: torch.Tensor, scales: torch.Tensor
    ):
        """The base class for logistic distribution functions

        Note: Don't need to sort for this because there is only one point per parameter

        Args:
            biases (torch.Tensor): The bias of the distribution
            scales (torch.Tensor): The scale value for the distribution
        """
        if biases.dim() == 2:
            biases = biases[None]
        if scales.dim() == 2:
            scales = scales[None]

        super().__init__(
            biases.shape[1],
            biases.shape[2]
        )
        self._biases = torch.nn.parameter.Parameter(biases)
        self._scales = torch.nn.parameter.Parameter(scales)

    @property
    def sigma(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self._scales)

    @property
    def biases(self) -> torch.Tensor:
        """
        Returns:
            ShapeParams: The bias values
        """
        return self._biases
    
    @property
    def scales(self) -> torch.Tensor:
        """
        Returns:
            ShapeParams: The scales
        """
        return self._scales

    @classmethod
    def from_combined(cls, params: torch.Tensor) -> 'Gaussian':
        """Create the shape from 

        Returns:
            Logistic: The logistic distribution function 
        """
        return cls(
            params[...,0], 
            params[...,1]
        )


class GaussianBell(Gaussian):
    """Use the GaussianBell function as the membership function
    """

    def join(self, x: Tensor) -> Tensor:
        """Convert x to a membership value

        Args:
            x (Tensor): The value to convert

        Returns:
            Tensor: The membership
        """
        return gaussian(
            unsqueeze(x), self._biases, self.sigma
        )
    
    def areas(self, m: Tensor, truncate: bool = False) -> Tensor:
        """Calculate the area of the Gaussian for a membership

        Args:
            m (Tensor): The membership to calculate the area for
            truncate (bool, optional): Whether to truncate the Gaussian (or scale). Defaults to False.

        Returns:
            Tensor: The area
        """
        if truncate:
            return self._resize_to_m(
                truncated_gaussian_area(self._biases, self._scales, m),
                m)
        return self._resize_to_m(
            gaussian_area(self._scales), m
        )
    
    def mean_cores(self, m: Tensor, truncate: bool = False) -> Tensor:
        """Calculate the 'mean core' of the Gaussian for a membership

        Args:
            m (Tensor): The membership to calculate the area for
            truncate (bool, optional): Whether to truncate the Gaussian (or scale). Defaults to False.

        Returns:
            Tensor: The mean of the "core" of the Gaussian
        """
        if truncate:
            return self._resize_to_m(
                truncated_gaussian_mean_core(self._biases, self._scales, m), m
            )
        return self._resize_to_m(self._biases, m)
    
    def centroids(self, m: Tensor, truncate: bool = False) -> Tensor:
        """Calculate the 'centroid' of the Gaussian for a membership

        Args:
            m (Tensor): The membership to calculate the area for
            truncate (bool, optional): Whether to truncate the Gaussian (or scale). Defaults to False.

        Returns:
            Tensor: The centroid of the Gaussian
        """
        return self._resize_to_m(self._biases, m)
    

class HalfGaussianBell(Gaussian):
    """Use the GaussianBell function as the membership function
    """
    def __init__(self, biases: Coords, scales: Coords, increasing: torch.Tensor):
        """Create a membership that comprises half of a Gaussian function either increasing or decreasing

        Args:
            biases (ShapeParams): The bias for the half bell
            scales (ShapeParams): The scales for the half bell
            increasing (torch.Tensor): Whether the function is increasing or decreasing
        """
        super().__init__(biases, scales)
        self.increasing = increasing

    def half_gaussian(self, x: torch.Tensor, biases: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Function for joining the Half-Gaussian

        Args:
            x (torch.Tensor): The x value
            biases (torch.Tensor): The bias
            scale (torch.Tensor): The scale

        Returns:
            torch.Tensor: The membership values
        """
        if self.increasing:
            contains = (x <= biases)
        else:
            contains = (x >= biases)

        return gaussian(
            x, biases, scale
        ) * contains

    def join(self, x: Tensor) -> Tensor:
        """Convert x to a membership value

        Args:
            x (Tensor): The value to convert

        Returns:
            Tensor: The membership
        """
        x = unsqueeze(x)
        return self.half_gaussian(
            x, self._biases, self.sigma
        )
    
    def areas(self, m: Tensor, truncate: bool = False) -> Tensor:
        """Calculate the area of the Gaussian for a membership

        Args:
            m (Tensor): The membership to calculate the area for
            truncate (bool, optional): Whether to truncate the Gaussian (or scale). Defaults to False.

        Returns:
            Tensor: The area
        """   
        if truncate:
            return truncated_half_gaussian_area(self._biases, self.sigma, m)
        return self._resize_to_m(half_gaussian_area(self.sigma), m)
    
    def mean_cores(self, m: Tensor, truncate: bool = False) -> Tensor:
        """Calculate the 'mean core' of the Gaussian for a membership

        Args:
            m (Tensor): The membership to calculate the area for
            truncate (bool, optional): Whether to truncate the Gaussian (or scale). Defaults to False.

        Returns:
            Tensor: The mean of the "core" of the Gaussian
        """
        if truncate:
            return self._resize_to_m(
                truncated_half_gaussian_mean_core(
                    self._biases, self.sigma, m, self.increasing
                ), m)
        return self._resize_to_m(self._biases, m)
    
    def centroids(self, m: Tensor, truncate: bool = False) -> Tensor:
        """Calculate the 'centroid' of the Gaussian for a membership

        Args:
            m (Tensor): The membership to calculate the area for
            truncate (bool, optional): Whether to truncate the Gaussian (or scale). Defaults to False.

        Returns:
            Tensor: The centroid of the Gaussian
        """
        if truncate:
            return truncated_half_gaussian_centroid(self._biases, self.sigma, m, self.increasing)
        return half_gaussian_centroid(self._biases, self.sigma, m, self.increasing)
