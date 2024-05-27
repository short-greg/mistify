from mistify.fuzzify._shapes._base import Coords
from ._base import Polygon
import torch
from ... import _functional as functional
from ..._functional import G
from ...utils import unsqueeze


class Square(Polygon):
    """A square or rectangular shaped member function. This member function
    only outputs two values
    """

    PT = 2

    def __init__(self, coords: Coords, g: G=None):
        super().__init__(coords)
        self.g = g

    def join(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The value to calculate the membership for 

        Returns:
            torch.Tensor: The membership value of x
        """
        params = self._coords()
        x = unsqueeze(x)
        return functional.shape.square(
            x, params[...,0], params[...,1], g=self.g
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
        coords = self._coords()
        return (coords[...,1] - coords[...,0]) * m
    
    def mean_cores(self, m: torch.Tensor, truncate: bool = False) -> torch.Tensor:
        """
        Args:
            m (torch.Tensor): The membership
            truncate (bool, optional): Whether to truncate or scale. 
              Note that for square they are the same. Defaults to False.

        Returns:
            torch.Tensor: The mean of the core
        """
        coords = self._coords()
        return self._resize_to_m((coords[...,1] + coords[...,0]) / 2.0, m)
    
    def centroids(self, m: torch.Tensor, truncate: bool = False) -> torch.Tensor:
        """
        Args:
            m (torch.Tensor): The membership
            truncate (bool, optional): Whether to truncate or scale. 
              Note that for square they are the same. Defaults to False.

        Returns:
            torch.Tensor: The centroid of the square
        """
        coords = self._coords()
        return self._resize_to_m((coords[...,1] + coords[...,0]) / 2.0, m)
