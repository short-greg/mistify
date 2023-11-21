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
        z = (unsqueeze(x) - self._biases.pt(0)) / self._scales.pt(0)
        return self._m_mul * torch.sigmoid(z)

    def _calc_min_cores(self):
        return self._m

    def scale(self, m: torch.Tensor) -> 'Step':
        if self._m is not None:
            m = self._m * m
        return Step(
            self._threshold, m
        )

    def truncate(self, m: torch.Tensor) -> 'Step':

        m = intersect(self._m, m)
        return Step(
            self._threshold, m
        )
