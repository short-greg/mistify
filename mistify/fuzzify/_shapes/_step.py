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
        self, threshold: ShapeParams, m: torch.Tensor=None
    ):
        """Create a step function

        Args:
            threshold (ShapeParams): The threshold where the step occurs
            m (torch.Tensor, optional): The max value of membership. Defaults to None.
        """
        super().__init__(
            threshold.n_variables,
            threshold.n_terms
        )
        self._threshold = threshold
        self._m = self._init_m(m, threshold.device)

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
        return functional.threshold(x, self._threshold.pt(0)) * self._m

    def _calc_min_cores(self):
        # NOTE: not correct if m is 0
        return self._threshold.pt(0) * torch.ones_like(self._m)

    # def _calc_area(self):
    #     # NOTE: not correct if m is 0
    #     return self._threshold.pt(0) * torch.zeros_like(self._m)

    def truncate(self, m: torch.Tensor) -> 'Step':
        """Reduce the height of teh step function

        Args:
            m (torch.Tensor): The value to reduce the height by

        Returns:
            Step: The updated step function
        """
        m = functional.inter(self._m, m)
        return Step(self._threshold, m)
