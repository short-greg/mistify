from ._base import Polygon
import torch


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
        """Scale the square's height

        Args:
            m (torch.Tensor): The value to scale by

        Returns:
            Square: The scaled square
        """
        updated_m = torch.min(m, self._m)
        
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
        updated_m = torch.min(m, self._m)

        return Square(
            self._params, updated_m
        )
