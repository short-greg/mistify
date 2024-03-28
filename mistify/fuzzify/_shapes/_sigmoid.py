# 1st party

# 3rd party
import torch

# local
from ._base import ShapeParams, Monotonic
from ...utils import unsqueeze
from ... import _functional as functional


class Sigmoid(Monotonic):
    """
    """

    def __init__(
        self, biases: ShapeParams, scales: ShapeParams
    ):
        """Create a Sigmoid membership function

        Args:
            biases (ShapeParams): The biases for the sigmoid function
            scales (ShapeParams): The scales for the sigmoid function
            truncate_m (torch.Tensor, optional): The value the sigmoid is truncated by. Defaults to None.
        """
        super().__init__(
            biases.n_variables,
            biases.n_terms
        )
        self._biases = biases
        self._scales = scales

        # self._truncate_m = self._init_m(truncate_m, biases.device)

    @property
    def biases(self):
        return self._biases
    
    @property
    def scales(self):
        return self._scales
    
    @classmethod
    def from_combined(cls, params: ShapeParams):

        return cls(
            params.sub((0, 1)), 
            params.sub((1, 2))
        )

    def join(self, x: torch.Tensor) -> torch.Tensor:
        """Join the set with the sigmoid fuzzifier

        Args:
            x (torch.Tensor): The Tensor to join with

        Returns:
            torch.Tensor: The membership value
        """
        z = (unsqueeze(x) - self._biases.pt(0)) / self._scales.pt(0)
        
        return torch.sigmoid(z)

    # def _calc_areas(self):
    #     # TODO: Need the integral of it
    #     return torch.log(torch.exp(self._truncate_m) + 1)
    #     # return self._m * torch.log(self._m) + (0.5 - self._m) * torch.log(1 - self._m) + 0.5 * torch.log(2 * self._m - 2)
        
    # def _calc_min_cores(self):

    #     result = torch.logit(self._truncate_m, 1e-7)
    #     return result * self._scales.pt(0) + self._biases.pt(0)


    def min_cores(self, m: torch.Tensor) -> torch.Tensor:

        m = m.clamp(1e-7, 1. - 1e7)
        return torch.logit(m) * self._scales.pt(0) + self._biases.pt(0)

    # def truncate(self, m: torch.Tensor) -> 'Sigmoid':
    #     """
    #     Args:
    #         m (torch.Tensor): the truncated value for the sigmoid

    #     Returns:
    #         Sigmoid: The updated sigmoid
    #     """
        
    #     updated_m = functional.inter(self._truncate_m, m)
    #     return Sigmoid(
    #         self._biases, self._scales, updated_m 
    #     )
