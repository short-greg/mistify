# 1st party

# 3rd party
import torch

# local
from ._base import ShapeParams, Monotonic
from ...utils import unsqueeze

intersect = torch.min


class Ramp(Monotonic):
    """A membership function in the shape of a ramp
    """

    def __init__(
        self, coords: ShapeParams, m: torch.Tensor=None# , scale_m: torch.Tensor=None
    ):
        """Create a ramp function that has a lower bound and an upper bound

        Args:
            coords (ShapeParams): The parameters for the ramp. Defining the lower and upper parameter
            m (torch.Tensor, optional): The membership value for the ramp. Defaults to None.
            scale_m (torch.Tensor, optional): The degree the ramp is scaled by. Defaults to None.
        """

        super().__init__(
            coords.n_variables,
            coords.n_terms
        )
        self._coords = coords
        self._m = self._init_m(m, coords.device)

    @property
    def coords(self) -> 'ShapeParams':
        """
        Returns:
            ShapeParams: The coordinates for the ramp function. It has two points, the lower bound and upper bound
        """
        return self._coords

    @classmethod
    def from_combined(cls, params: ShapeParams, m: torch.Tensor=None):

        return cls(
            params.sub((0, 1)), m
        )

    def join(self, x: torch.Tensor) -> torch.Tensor:
        """Join calculates the membership value for each section of Ramp and uses the maximimum value
        as the value

        Args:
            x (torch.Tensor): The value to calculate the membership for

        Returns:
            torch.Tensor: The membership
        """
        # min_ = torch.tensor(0, dtype=x.dtype, device=x.device)
        m = (unsqueeze(x) * ((self._m / (self._coords.pt(1) - self._coords.pt(0))) - self._coords.pt(0)))
        return torch.clamp(torch.clamp(m, max=self._m), 0.0)
    
    def _calc_min_cores(self):
        """
        Returns:
            torch.Tensor: the minimum value of the start of the core of the set
        """
        return self._resize_to_m(self._coords.pt(1), self._m)

    def truncate(self, m: torch.Tensor) -> 'Ramp':
        """Truncate the Ramp function. This results in changing the m value as well
        as the point for the upper bound

        Args:
            m (torch.Tensor): The value to truncate by

        Returns:
            Ramp: The truncated ramp
        """
        truncate_m = intersect(self._m, m)
        pt = (truncate_m + self._coords.pt(0)) * (self._coords.pt(1) - self._coords.pt(0)) / self._m
        coords = self._coords.replace(pt, 1, True, truncate_m)

        return Ramp(coords, m)
