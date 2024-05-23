# 1st party
from typing_extensions import Self

# 3rd party
import torch

# local
from ._base import Coords, Monotonic


class Sigmoid(Monotonic):

    def __init__(
        self, biases: torch.Tensor, scales: torch.Tensor
    ):
        """Create a Sigmoid membership function

        Args:
            biases (torch.Tensor): The biases for the sigmoid function
            scales (torch.Tensor): The scales for the sigmoid function
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
    def biases(self) -> Coords:
        """
        Returns:
            ShapeParams: The bias parameter for the sigmoid 
        """
        return self._biases
    
    @property
    def scales(self) -> Coords:
        """
        Returns:
            ShapeParams: The scale parameters for the sigmoid
        """
        return self._scales
    
    @classmethod
    def from_combined(cls, params: torch.Tensor) -> Self:
        """Create the Sigmoid with biases and scales combined
        into one parameter

        Args:
            params (ShapeParams): Bias/Scale

        Returns:
            Self: The Sigmoid
        """
        return cls(
            params[...,0], 
            params[...,1]
        )

    def join(self, x: torch.Tensor) -> torch.Tensor:
        """Join the set with the sigmoid fuzzifier

        Args:
            x (torch.Tensor): The Tensor to join with

        Returns:
            torch.Tensor: The membership value
        """
        z = (x[...,None] - self._biases) / self._scales
        
        return torch.sigmoid(z)

    def min_cores(self, m: torch.Tensor) -> torch.Tensor:
        """Calculate the minimum x for which m is a maximum

        Args:
            m (torch.Tensor): The membership

        Returns:
            torch.Tensor: The "min core"
        """
        m = m.clamp(1e-7, 1. - 1e7)
        return torch.logit(m) * self._scales + self._biases
