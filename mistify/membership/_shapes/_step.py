# 1st party

# 3rd party
import torch

# local
from ._base import ShapeParams, Monotonic
from ...utils import unsqueeze

intersect = torch.min


class Step(Monotonic):

    def __init__(
        self, threshold: ShapeParams, m: torch.Tensor=None
    ):
        self._threshold = threshold

        self._m = self._init_m(m, threshold.device)

        super().__init__(
            self._threshold.n_variables,
            self._threshold.n_terms
        )

    @property
    def thresholds(self):
        return self._threshold
    
    @classmethod
    def from_combined(cls, params: ShapeParams, m: torch.Tensor=None):

        return cls(
            params.sub((0, 1)), m
        )

    def join(self, x: torch.Tensor) -> torch.Tensor:
        return intersect(self._m, (unsqueeze(x) >= self._threshold.pt(0)).type_as(x))

    def _calc_min_cores(self):
        # NOTE: not correct if m is 0
        return self._threshold.pt(0).squeeze(-1)

    def scale(self, m: torch.Tensor) -> 'Step':
        m = self._m * m
        return Step(
            self._threshold, m
        )

    def truncate(self, m: torch.Tensor) -> 'Step':

        m = intersect(self._m, m)
        return Step(
            self._threshold, m
        )
