# 1st party
import typing

# 3rd party
import torch

# local
from ._base import Shape, Nonmonotonic, Monotonic


class Composite(Nonmonotonic, Monotonic):
    """A shape that wraps several nonmonotonic shapes
    """

    def __init__(self, shapes: typing.List[typing.Union[Monotonic, Nonmonotonic]]):
        """Create a composite of Nonmonotonic shapes. If monotonic and non_montonic shapes
        are mixed only the area can be used for defuzzification. Fuzzification will work fine
        though
        """
        n_terms = 0
        n_variables = -1
        for shape in shapes:
            n_terms += shape.n_terms
            if n_variables == -1:
                n_variables = shape.n_variables
            else:
                if n_variables != shape.n_variables:
                    raise ValueError('Number of variables must be the same for all shapes')
        super().__init__(n_variables=n_variables, n_terms=n_terms)
        self._shapes = shapes

    @property
    def shapes(self) -> typing.List[Nonmonotonic]:
        """
        Returns:
            typing.List[Nonmonotonic]: The shapes making up the CompositeNonmonotonic
        """
        return [*self._shapes]

    def join(self, x: torch.Tensor) -> torch.Tensor:
        """Join over each of the shapes and concatenate the results

        Args:
            x (torch.Tensor): The value to get the membership value for

        Returns:
            torch.Tensor: The membership
        """
        return torch.cat(
            [shape.join(x) for shape in self._shapes], dim=2
        )

    def areas(self, m: torch.Tensor, truncate: bool = False) -> torch.Tensor:
        return torch.cat(
            [shape.areas(m, truncate) for shape in self._shapes], dim=-1
        )
    
    def min_cores(self, m: torch.Tensor, truncate: bool = False) -> torch.Tensor:
        return torch.cat(
            [shape.min_cores(m, truncate) for shape in self._shapes], dim=-1
        )
    
    def mean_cores(self, m: torch.Tensor, truncate: bool = False) -> torch.Tensor:
        return torch.cat(
            [shape.mean_cores(m, truncate) for shape in self._shapes], dim=-1
        )
    
    def centroids(self, m: torch.Tensor, truncate: bool = False) -> torch.Tensor:
        return torch.cat(
            [shape.centroids(m, truncate) for shape in self._shapes], dim=-1
        )
