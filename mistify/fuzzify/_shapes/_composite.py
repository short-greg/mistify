# 1st party
import typing

# 3rd party
import torch

# local
from ._base import Shape, Nonmonotonic, Monotonic


class Composite(Nonmonotonic, Monotonic):
    """A shape that wraps several nonmonotonic shapes
    """

    def __init__(self, shapes: typing.List[Shape]):
        """Create a composite of Nonmonotonic shapes
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

        return torch.cat(
            [shape.join(x) for shape in self._shapes], dim=2
        )
    
    def _calc_areas(self):
        return torch.cat(
            [shape.areas for shape in self._shapes], dim=2
        )

    def _calc_centroids(self) -> torch.Tensor:
        return torch.cat(
            [shape.centroids for shape in self._shapes], dim=2
        )

    def _calc_mean_cores(self) -> torch.Tensor:
        return torch.cat(
            [shape.mean_cores for shape in self._shapes], dim=2
        )

    def _calc_min_cores(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: The minimum value of the core of the set
        """
        return torch.cat(
            [shape.mean_cores for shape in self._shapes], dim=2
        )

    def truncate(self, m: torch.Tensor) -> Shape:
        """Truncate each of the shapes

        Args:
            m (torch.Tensor): The membership value to truncate by

        Returns:
            CompositeNonmonotonic: Composite with all shapes scaled 
        """
        truncated = []
        start = 0
        last = None
        print(m.shape)
        for shape in self._shapes:
            
            # (batch, n_variables, n_terms, )
            last = shape.n_terms + start
            m_cur = m[:,:,start:last]
            truncated.append(shape.truncate(m_cur))
            start = last
    
        return Composite(truncated)

    def scale(self, m: torch.Tensor) -> 'Composite':
        """Scale each of the shapes

        Args:
            m (torch.Tensor): The membership value to truncate by

        Returns:
            CompositeNonmonotonic: Composite with all shapes scaled 
        """
        scaled = []
        start = 0
        last = None
        for shape in self._shapes:
            
            last = shape.n_terms + start
            m_cur = m[:,:,start:last]
            scaled.append(shape.scale(m_cur))
            start = last
    
        return Composite(scaled)
