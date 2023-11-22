# 1st party

# 3rd party
import torch

# local
from ._base import ShapeParams, Monotonic
from ...utils import unsqueeze

intersect = torch.min


class Sigmoid(Monotonic):

    def __init__(
        self, biases: ShapeParams, scales: ShapeParams, 
        m_mul: torch.Tensor=None, m: torch.Tensor=None
    ):
        self._biases = biases
        self._scales = scales

        self._m = self._init_m(m, biases.device)
        self._m_mul = self._init_m(m_mul, biases.device)

        super().__init__(
            self._biases.n_variables,
            self._biases.n_terms
        )

    @property
    def biases(self):
        return self._biases
    
    @property
    def scales(self):
        return self._scales
    
    @classmethod
    def from_combined(cls, params: ShapeParams, m: torch.Tensor=None):

        return cls(
            params.sub((0, 1)), 
            params.sub((1, 2)), m
        )

    def join(self, x: torch.Tensor) -> torch.Tensor:
        z = (unsqueeze(x) - self._biases.pt(0)) / self._scales.pt(0)
        
        return intersect(self._m, self._m_mul * torch.sigmoid(z))

    def _calc_areas(self):
        # TODO: Need the integral of it
        return self._m_mul * torch.log(torch.exp(self._m) + 1)
        # return self._m * torch.log(self._m) + (0.5 - self._m) * torch.log(1 - self._m) + 0.5 * torch.log(2 * self._m - 2)
        
    def _calc_min_cores(self):

        result = torch.logit(self._m / self._m_mul, 1e-7)
        return result * self._scales.pt(0) + self._biases.pt(0)

    def scale(self, m: torch.Tensor) -> 'Sigmoid':
        updated_mul = self._m_mul * m
        
        return Sigmoid(
            self._biases, self._scales, updated_mul, intersect(updated_mul, self._m)
        )

    def truncate(self, m: torch.Tensor) -> 'Sigmoid':
        updated_m = intersect(self._m, m)
        return Sigmoid(
            self._biases, self._scales, self._m_mul, updated_m 
        )
