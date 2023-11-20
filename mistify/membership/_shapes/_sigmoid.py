# 1st party

# 3rd party
import torch

# local
from ._base import ShapeParams, Shape
from ...utils import unsqueeze, check_contains
from ._utils import calc_area_logistic, calc_dx_logistic, calc_area_logistic_one_side, calc_m_logistic, calc_x_logistic

intersect = torch.min


class Sigmoid(Shape):

    def __init__(
        self, biases: ShapeParams, scales: ShapeParams, 
        m_mul: torch.Tensor=None, m: torch.Tensor=None
    ):
        self._biases = biases
        self._scales = scales

        self._m = m if m is not None else torch.ones(
            self._biases.batch_size, self._biases.set_size, 
            self._biases.n_terms, device=biases.x.device
        )
        self._m_mul = m_mul

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

        #if params.x.dim() == 4:
        return cls(
            params.sub((0, 1)), 
            params.sub((1, 2)), m
        )
        # return cls(params[:,:,0], params[:,:,1], m)

    def join(self, x: torch.Tensor) -> torch.Tensor:
        z = (unsqueeze(x) - self._biases.pt(0)) / self._scales.pt(0)
        return self._m_mul * torch.sigmoid(z)

    def _calc_areas(self):
        # TODO: Need the integral of it
        return torch.log(torch.exp(self._m) + 1)
        # return self._m * torch.log(self._m) + (0.5 - self._m) * torch.log(1 - self._m) + 0.5 * torch.log(2 * self._m - 2)
        
    def _calc_mean_cores(self):
        return self._m_mul * torch.logit(self._m, 1e-7)

    def _calc_centroids(self):
        # TODO: Determine the centroid - need the integral
        raise NotImplementedError

    def scale(self, m: torch.Tensor) -> 'Sigmoid':
        if self._m_mul is not None:
            updated_mul = self._m_mul * m
        return Sigmoid(
            self._biases, self._scales, updated_mul, self._m
        )

    def truncate(self, m: torch.Tensor) -> 'SigmoidTruncated':

        updated_m = intersect(self._m, m)
        return SigmoidTruncated(
            self._biases, self._scales,  updated_m 
        )


class SigmoidTruncated(Shape):

    def __init__(
        self, biases: ShapeParams, scales: ShapeParams, m: torch.Tensor=None
    ):
        self._biases = biases
        self._scales = scales

        self._m = m if m is not None else torch.ones(
            self._biases.batch_size, self._biases.set_size, 
            self._biases.n_terms, device=biases.x.device
        )

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

        #if params.x.dim() == 4:
        return cls(
            params.sub((0, 1)), 
            params.sub((1, 2)), m
        )
        # return cls(params[:,:,0], params[:,:,1], m)

    def join(self, x: torch.Tensor) -> torch.Tensor:
        z = self._scales.pt(0) * (unsqueeze(x) - self._biases.pt(0))
        sig = torch.sigmoid(z)
        # not 4 / s
        return 4  * (1 - sig) * sig * self._m

    def _calc_areas(self):
        pass
        # return self._resize_to_m(4 * self._m / self._biases.pt(0), self._m)
        
    def _calc_mean_cores(self):
        pass
        # return self._resize_to_m(self._biases.pt(0), self._m)

    def _calc_centroids(self):
        pass
        # return self._resize_to_m(self._biases.pt(0), self._m)

    def scale(self, m: torch.Tensor) -> 'Sigmoid':
        updated_m = intersect(self._m, m)
        return Sigmoid(
            self._biases, self._scales, updated_m
        )

    def truncate(self, m: torch.Tensor) -> 'Sigmoid':

        return Sigmoid(
            self._biases, self._scales,  m, self._m 
        )
