from ._base import Polygon
import torch
from ... import _functional as functional
from ...utils import unsqueeze


class Square(Polygon):
    """A square or rectangular shaped member function. This member function
    only outputs two values
    """

    PT = 2

    def join(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The value to calculate the membership for 

        Returns:
            torch.Tensor: The membership value of x
        """
        params = self._params()
        x = unsqueeze(x)
        # return (
        #     (x[:,:,None] >= params.pt(0)) 
        #     & (x[:,:,None] <= params.pt(1))
        # ).type_as(x) * self._m
        return functional.shape.square(
            x, params.pt(0), params.pt(1), self._m
        )

    def _calc_areas(self):
        """Calculates the area of each section and sums it up

        Returns:
            torch.Tensor: The area of the square
        """
        params = self._params()
        return self._resize_to_m((
            (params.pt(1) 
            - params.pt(0)) * self._m
        ), self._m)

    def _calc_mean_cores(self):
        """
        Returns:
            torch.Tensor: the mean value of the top of the square
        """
        params = self._params()
        return self._resize_to_m(1 / 2 * (
            params.pt(0) + params.pt(1)
        ), self._m)

    def _calc_centroids(self):
        """
        Returns:
            torch.Tensor: The center of mass for the square
        """
        params = self._params()
        return self._resize_to_m(1 / 2 * (
            params.pt(0) + params.pt(1)
        ), self._m)
    
    def scale(self, m: torch.Tensor) -> 'Square':
        """Scale the square's height

        Args:
            m (torch.Tensor): The value to scale by

        Returns:
            Square: The scaled square
        """
        updated_m = functional.inter(m, self._m)
        
        return Square(
            self._params, updated_m
        )

    def truncate(self, m: torch.Tensor) -> 'Square':
        """Scale the square's height

        Args:
            m (torch.Tensor): The value to truncate by

        Returns:
            Square: The truncated square
        """
        updated_m = functional.inter(m, self._m)

        return Square(
            self._params, updated_m
        )
