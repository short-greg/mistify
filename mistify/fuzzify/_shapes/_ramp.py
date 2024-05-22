# 1st party
from typing_extensions import Self

# 3rd party
import torch

# local
from ._base import ShapeParams, Monotonic
from ...utils import unsqueeze
from ... import _functional as functional


class Ramp(Monotonic):
    """A membership function in the shape of a ramp
    """

    def __init__(
        self, coords: ShapeParams
    ):
        """Create a ramp function that has a lower bound and an upper bound

        Args:
            coords (ShapeParams): The parameters for the ramp. Defining the lower and upper parameter
            m (torch.Tensor, optional): The membership value for the ramp. Defaults to None.
            scale_m (torch.Tensor, optional): The degree the ramp is scaled by. Defaults to None.
        """

        super().__init__(
            coords.n_vars,
            coords.n_terms
        )
        self._coords = coords

    @property
    def coords(self) -> 'ShapeParams':
        """
        Returns:
            ShapeParams: The coordinates for the ramp function. It has two points, the lower bound and upper bound
        """
        return self._coords

    @classmethod
    def from_combined(cls, params: ShapeParams) -> Self:
        """Create the ramp from 

        Args:
            params (ShapeParams): Shape params with first and second point combined

        Returns:
            Ramp: A ramp function
        """
        return cls(
            params.sub((0, 1))
        )

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
        x = unsqueeze(x)
        coords = self._coords()
        return functional.ramp(x, coords.pt(0), coords.pt(1))

    def min_cores(self, m: torch.Tensor) -> torch.Tensor:
        """Calculates an average of the two points disregarding
        other points (min support + min core)

        Args:
            m (torch.Tensor): The membership

        Returns:
            torch.Tensor: The "min core"
        """
        coords = self._coords()
        # print(coords.pt(1)[0, 0], coords.pt(0)[0, 0])
        return coords.pt(1) * (1 - m) + coords.pt(0) * m
