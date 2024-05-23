# 1st party
from typing_extensions import Self

# 3rd party
import torch

# local
from ._base import Monotonic
from ... import _functional as functional


class Step(Monotonic):
    """A step membership function
    """

    def __init__(
        self, threshold: torch.Tensor
    ):
        """Create a step function

        Args:
            threshold (ShapeParams): The threshold where the step occurs
        """
        if threshold.dim() == 2:
            threshold = threshold[None]
        
        super().__init__(
            threshold.shape[1],
            threshold.shape[2]
        )
        self._threshold = torch.nn.parameter.Parameter(threshold)

    @property
    def thresholds(self) -> torch.Tensor:
        """
        Returns:
            ShapeParams: The threshold where the step occurs
        """
        return self._threshold
    
    @classmethod
    def from_combined(cls, params: torch.Tensor) -> Self:
        """Create Step from combined parameters

        Args:
            params (torch.Tensor): The parameters (with a threshold point)

        Returns:
            Step: The Step shape
        """
        return cls(
            params[...,0]
        )

    def join(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The value to calculate the membership for 

        Returns:
            torch.Tensor: The membership value of x
        """
        x = x[...,None]
        return functional.threshold(x, self._threshold)
    
    def min_cores(self, m: torch.Tensor) -> torch.Tensor:
        """
        Args:
            m (torch.Tensor): The membershp

        Returns:
            torch.Tensor: The value of the threshold
        """
        return self._resize_to_m(self._threshold, m)
