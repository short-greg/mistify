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
        return functional.shape.square(
            x, params.pt(0), params.pt(1)
        )

    def areas(self, m: torch.Tensor, truncate: bool = False) -> torch.Tensor:
        """
        Args:
            m (torch.Tensor): The membership
            truncate (bool, optional): Whether to truncate or scale. 
              Note that for square they are the same. Defaults to False.

        Returns:
            torch.Tensor: The area of the square
        """
        return (self._params.pt(1) - self._params.pt(0)) * m
    
    def mean_cores(self, m: torch.Tensor, truncate: bool = False) -> torch.Tensor:
        """
        Args:
            m (torch.Tensor): The membership
            truncate (bool, optional): Whether to truncate or scale. 
              Note that for square they are the same. Defaults to False.

        Returns:
            torch.Tensor: The mean of the core
        """
        return self._resize_to_m((self._params.pt(1) + self._params.pt(0)) / 2.0, m)
    
    def centroids(self, m: torch.Tensor, truncate: bool = False) -> torch.Tensor:
        """
        Args:
            m (torch.Tensor): The membership
            truncate (bool, optional): Whether to truncate or scale. 
              Note that for square they are the same. Defaults to False.

        Returns:
            torch.Tensor: The centroid of the square
        """
        return self._resize_to_m((self._params.pt(1) + self._params.pt(0)) / 2.0, m)
