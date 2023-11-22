# 1st party

# 3rd party
import torch

# local
from ._base import ShapeParams, Monotonic
from ...utils import unsqueeze

intersect = torch.min


class Ramp(Monotonic):

    def __init__(
        self, coords: ShapeParams, m: torch.Tensor=None, scale_m: torch.Tensor=None
    ):
        self._coords = coords
        self._m = self._init_m(m, coords.device)
        self._scale_m = self._init_m(scale_m, coords.device)

        super().__init__(
            self._coords.n_variables,
            self._coords.n_terms
        )

    @property
    def coords(self):
        return self._coords

    @classmethod
    def from_combined(cls, params: ShapeParams, m: torch.Tensor=None):

        return cls(
            params.sub((0, 1)), m
        )

    def join(self, x: torch.Tensor) -> torch.Tensor:
        
        # min_ = torch.tensor(0, dtype=x.dtype, device=x.device)
        m = (unsqueeze(x) * ((self._m / (self._coords.pt(1) - self._coords.pt(0))) - self._coords.pt(0)))
        return torch.clamp(torch.clamp(m, max=self._m), 0.0)
    
    def _calc_min_cores(self):

        return self._coords.pt(1)

    def _calc_area(self):
        return 0.5 * (self._coords.pt(1) - self._coords.pt(0)) * self._m

    def scale(self, m: torch.Tensor) -> 'Ramp':
        return Ramp(self._coords, m, self._m * m)

    def truncate(self, m: torch.Tensor) -> 'Ramp':

        truncate_m = intersect(self._m, m)
        pt = (truncate_m + self._coords.pt(0)) * (self._coords.pt(1) - self._coords.pt(0)) / self._m
        coords = self._coords.replace(pt, 1)

        return Ramp(coords, m)
