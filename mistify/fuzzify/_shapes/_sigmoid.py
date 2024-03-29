# 1st party

# 3rd party
import torch

# local
from ._base import ShapeParams, Monotonic
from ...utils import unsqueeze
from ... import _functional as functional


class Sigmoid(Monotonic):
    """
    """

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
    def biases(self):
        return self._biases
    
    @property
    def scales(self):
        return self._scales
    
    @classmethod
    def from_combined(cls, params: ShapeParams):

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

        m = m.clamp(1e-7, 1. - 1e7)
        return torch.logit(m) * self._scales.pt(0) + self._biases.pt(0)
