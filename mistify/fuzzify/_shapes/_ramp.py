# 1st party
from typing_extensions import Self

# 3rd party
import torch

# local
from ._base import Coords, Monotonic
from ..._functional import G
from ... import _functional as functional


class Ramp(Monotonic):
    """A membership function in the shape of a ramp
    """

    def __init__(
        self, coords: Coords, g: G=None
    ):
        """Create a ramp function that has a lower bound and an upper bound

        Args:
            coords (ShapeParams): The parameters for the ramp. Defining the lower and upper parameter
            m (torch.Tensor, optional): The membership value for the ramp. Defaults to None.
            scale_m (torch.Tensor, optional): The degree the ramp is scaled by. Defaults to None.
        """
        if coords.n_points != 2:
            raise ValueError('Ramp only works if the number of coordinates is two')
        super().__init__(
            coords.shape[1], coords.shape[2]
        )
        self.g = g
        self._coords = coords

    @property
    def coords(self) -> Coords:
        """
        Returns:
            Coords: The coordinates for the ramp function. It has two points, the lower bound and upper bound
        """
        return self._coords

    @classmethod
    def from_combined(cls, coords: Coords) -> Self:
        """Create the ramp from combined coords

        Args:
            params (ShapeParams): Shape params with first and second point combined

        Returns:
            Ramp: A ramp function
        """
        return cls(coords)

    def constrain(self):

        self._coords.constrain(self._eps)

    def join(self, x: torch.Tensor) -> torch.Tensor:
        """Join calculates the membership value for each section of Ramp and uses the maximimum value
        as the value

        Args:
            x (torch.Tensor): The value to calculate the membership for

        Returns:
            torch.Tensor: The membership
        """
        x = x[...,None]
        coords = self._coords()
        m = functional.ramp(x, coords[...,0], coords[...,1], self.g)
        return m

    def min_cores(self, m: torch.Tensor) -> torch.Tensor:
        """Calculates an average of the two points disregarding
        other points (min support + min core)

        Args:
            m (torch.Tensor): The membership

        Returns:
            torch.Tensor: The "min core"
        """
        coords = self._coords()
        return coords[...,0] * (1 - m) + coords[...,1] * m
