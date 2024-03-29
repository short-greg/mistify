# 1st party

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
    def from_combined(cls, params: ShapeParams, m: torch.Tensor=None):

        return cls(
            params.sub((0, 1)), m
        )

    def join(self, x: torch.Tensor) -> torch.Tensor:
        x = unsqueeze(x)
        # return self._m * functional.binarize(x - self._threshold.pt(0))
        # return intersect(self._m, (unsqueeze(x) >= self._threshold.pt(0)).type_as(x))
        return functional.threshold(x, self._threshold.pt(0))
    
    def min_cores(self, m: torch.Tensor) -> torch.Tensor:
        return self._resize_to_m(self._threshold.pt(0), m)

