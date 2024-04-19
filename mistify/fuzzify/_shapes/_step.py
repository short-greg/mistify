# 1st party
from typing_extensions import Self

# 3rd party
import torch

# local
from ._base import ShapeParams, Monotonic
from ...utils import unsqueeze
from ... import _functional as functional


class Step(Monotonic):
    """A step membership function
    """

    def __init__(
        self, threshold: ShapeParams
    ):
        """Create a step function

        Args:
            threshold (ShapeParams): The threshold where the step occurs
        """
        super().__init__(
            threshold.n_variables,
            threshold.n_terms
        )
        self._threshold = threshold

    @property
    def thresholds(self) -> ShapeParams:
        """
        Returns:
            ShapeParams: The threshold where the step occurs
        """
        return self._threshold
    
    @classmethod
    def from_combined(cls, params: ShapeParams, m: torch.Tensor=None) -> Self:
        """Create Step from combined parameters

        Args:
            params (ShapeParams): The parameters (with a threshold point)
            m (torch.Tensor, optional): The. Defaults to None.

        Returns:
            Step: The Step shape
        """
        return cls(
            params.sub((0, 1)), m
        )

    def join(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The value to calculate the membership for 

        Returns:
            torch.Tensor: The membership value of x
        """
        x = unsqueeze(x)
        return functional.threshold(x, self._threshold.pt(0))
    
    def min_cores(self, m: torch.Tensor) -> torch.Tensor:
        """
        Args:
            m (torch.Tensor): The membershp

        Returns:
            torch.Tensor: The value of the threshold
        """
        return self._resize_to_m(self._threshold.pt(0), m)
