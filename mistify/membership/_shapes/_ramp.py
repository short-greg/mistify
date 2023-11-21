# 1st party

# 3rd party
import torch

# local
from ._base import ShapeParams, Monotonic
from ...utils import unsqueeze

intersect = torch.min


class Ramp(Monotonic):

    def __init__(
        self, lower: ShapeParams, upper: ShapeParams, m: torch.Tensor=None
    ):
        self._lower = lower
        self._upper = upper

        self._m = self._init_m(m, lower.device)

        super().__init__(
            self._lower.n_variables,
            self._lower.n_terms
        )

    @property
    def lower(self):
        return self._lower

    @property
    def upper(self):
        return self._upper

    @classmethod
    def from_combined(cls, params: ShapeParams, m: torch.Tensor=None):

        return cls(
            params.sub((0, 1)), m
        )

    def join(self, x: torch.Tensor) -> torch.Tensor:
        z = (unsqueeze(x) - self._biases.pt(0)) / self._scales.pt(0)
        return self._m_mul * torch.sigmoid(z)

    def _calc_min_cores(self):
        return self._upper

    def _calc_area(self):
        return 0.5 * (self._upper - self._lower) * self._m

    def scale(self, m: torch.Tensor) -> 'Ramp':
        m = self._m * m
        
        return Ramp(
            self._lower, self._upper, m
        )

    def truncate(self, m: torch.Tensor) -> 'Ramp':

        m = intersect(self._m, m)
        return Ramp(
            self._threshold, m
        )
