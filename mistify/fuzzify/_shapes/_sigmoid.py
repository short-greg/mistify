# 1st party
from typing_extensions import Self

# 3rd party
import torch

# local
from ._base import ShapeParams, Monotonic
from ...utils import unsqueeze


class Sigmoid(Monotonic):

    def __init__(
        self, biases: ShapeParams, scales: ShapeParams
    ):
        """Create a Sigmoid membership function

        Args:
            biases (ShapeParams): The biases for the sigmoid function
            scales (ShapeParams): The scales for the sigmoid function
        """
        super().__init__(
            biases.n_variables,
            biases.n_terms
        )
        self._biases = biases
        self._scales = scales

    @property
    def biases(self) -> ShapeParams:
        """
        Returns:
            ShapeParams: The bias parameter for the sigmoid 
        """
        return self._biases
    
    @property
    def scales(self) -> ShapeParams:
        """
        Returns:
            ShapeParams: The scale parameters for the sigmoid
        """
        return self._scales
    
    @classmethod
    def from_combined(cls, params: ShapeParams) -> Self:
        """Create the Sigmoid with biases and scales combined
        into one parameter

        Args:
            params (ShapeParams): Bias/Scale

        Returns:
            Self: The Sigmoid
        """
        return cls(
            params.sub((0, 1)), 
            params.sub((1, 2))
        )

    def join(self, x: torch.Tensor) -> torch.Tensor:
        """Join the set with the sigmoid fuzzifier

        Args:
            x (torch.Tensor): The Tensor to join with

        Returns:
            torch.Tensor: The membership value
        """
        z = (unsqueeze(x) - self._biases.pt(0)) / self._scales.pt(0)
        
        return torch.sigmoid(z)

    def min_cores(self, m: torch.Tensor) -> torch.Tensor:
        """Calculate the minimum x for which m is a maximum

        Args:
            m (torch.Tensor): The membership

        Returns:
            torch.Tensor: The "min core"
        """
        m = m.clamp(1e-7, 1. - 1e7)
        return torch.logit(m) * self._scales.pt(0) + self._biases.pt(0)
