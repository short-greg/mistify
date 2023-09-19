from ..base import Polygon
import torch
from .utils import intersect


class Square(Polygon):

    PT = 2

    def join(self, x: torch.Tensor):
        return (
            (x[:,:,None] >= self._params.pt(0)) 
            & (x[:,:,None] <= self._params.pt(1))
        ).type_as(x) * self._m

    def _calc_areas(self):
        
        return self._resize_to_m((
            (self._params.pt(1) 
            - self._params.pt(0)) * self._m
        ), self._m)

    def _calc_mean_cores(self):
        return self._resize_to_m(1 / 2 * (
            self._params.pt(0) + self._params.pt(1)
        ), self._m)

    def _calc_centroids(self):
        return self._resize_to_m(1 / 2 * (
            self._params.pt(0) + self._params.pt(1)
        ), self._m)
    
    def scale(self, m: torch.Tensor) -> 'Square':
        updated_m = intersect(m, self._m)
        
        return Square(
            self._params, updated_m
        )

    def truncate(self, m: torch.Tensor) -> 'Square':
        updated_m = intersect(m, self._m)

        return Square(
            self._params, updated_m
        )
