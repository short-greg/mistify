# 1st party
from abc import abstractmethod, abstractproperty
from dataclasses import dataclass
import typing

# 3rd party
import torch

# local
from ...utils._utils import resize_to, unsqueeze


class Shape(object):
    """To convert an input into a membership or vice-versa
    """

    def __init__(self, n_variables: int, n_terms: int):
        """Create the shape

        Args:
            n_variables (int): the number of linguistic variables
            n_terms (int): the number of terms for each variable
        """
        super().__init__()
        self._n_variables = n_variables
        self._n_terms = n_terms
        self._areas = None

    def _init_m(self, m: torch.Tensor=None, device='cpu') -> torch.Tensor:
        """initialize m

        Args:
            m (torch.Tensor, optional): the membership. Defaults to None.
            device (str, optional): the device for the membership. Defaults to 'cpu'.

        Returns:
            torch.Tensor: the output membership
        """
        if m is None:
            return torch.tensor(1., device=device)
        return m.to(device)

    @abstractmethod
    def _calc_areas(self):
        pass

    @property
    def areas(self) -> torch.Tensor:
        if self._areas is None:
            self._areas = self._calc_areas()
        return self._areas

    @property
    def n_terms(self):
        return self._n_terms
    
    @property
    def n_variables(self):
        return self._n_variables

    def join(self, x: torch.Tensor):
        pass

    @abstractproperty
    def m(self):
        """
        Returns:
            torch.Tensor: 
        """
        pass
    
    @abstractmethod
    def scale(self, m: torch.Tensor) -> 'Shape':
        """Scale the shape by a membership tensor

        Args:
            m (torch.Tensor): Membership tensor to scale by

        Returns:
            Shape: Scaled shape
        """
        pass

    @abstractmethod
    def truncate(self, m: torch.Tensor) -> 'Shape':
        """Truncate the shape by a membership tensor

        Args:
            m (torch.Tensor): Membership tensor to truncate by

        Returns:
            Shape: Scaled shape
        """
        pass

    @abstractmethod
    def join(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the membership of the value x

        Args:
            x (torch.Tensor): Tensor to cacluate membership for

        Returns:
            torch.Tensor: membership tensor
        """
        pass

    def _resize_to_m(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Convenience method to resize x to ahve the same batch size
        as m

        Args:
            x (torch.Tensor): Tensor to resize batch size for
            m (torch.Tensor): Tensor to resize based on

        Returns:
            torch.Tensor: Resized tensor
        """
        if x.size(0) == 1 and m.size(0) != 1:
            return x.repeat(m.size(0), *[1] * (m.dim() - 1))
        return x    


class Monotonic(Shape):
    """A shape that monotonically increases or decreases
    """

    def __init__(self, n_variables: int, n_terms: int):

        super().__init__(n_variables, n_terms)
        self._min_cores = None
    
    @abstractmethod
    def _calc_min_cores(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: The minimum of the "core" of the shape
        """
        pass

    @property
    def min_cores(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: The mean of the core of the shape
        """
        if self._min_cores is None:
            self._min_cores = self._calc_min_cores()
        return self._min_cores


class Concave(Shape):
    """A shape that is concave. It is uncertain 
    """

    def __init__(self, n_variables: int, n_terms: int):
        """

        Args:
            n_variables (int): the number of linguistic variables
            n_terms (int): the number of terms for each variable
        """
        super().__init__(n_variables, n_terms)
        self._mean_cores = None
        self._centroids = None

    @abstractmethod
    def _calc_mean_cores(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: The mean of the core of the shape
        """
        pass

    @property
    def mean_cores(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: The mean of the core of the shape
        """
        if self._mean_cores is None:
            self._mean_cores = self._calc_mean_cores()
        return self._mean_cores

    @abstractmethod
    def _calc_centroids(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: The centroid of the shape
        """
        pass

    @property
    def centroids(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: Centroid for the 
        """
        if self._centroids is None:
            self._centroids = self._calc_centroids()
        return self._centroids


class ShapeParams:
    """Parameters to specify the Shapes
    """
    
    # batch, set, index
    param: torch.Tensor

    def __init__(self, x: torch.Tensor):
        if x.dim() == 3:
            x = x[None]
        assert x.dim() == 4
        self._x = x

    def sub(self, index: typing.Union[int, typing.Tuple[int, int]]) -> 'ShapeParams':
        """Extract a subset of the parameters

        Args:
            index (typing.Union[int, typing.Tuple[int, int]]): Index to extract with

        Returns:
            ShapeParams: Subset of the shape parameters
        """
        if isinstance(index, int):
            index = slice(index, index + 1)
        else:
            index = slice(*index)
        return ShapeParams(self._x[:, :, :, index])

    @property
    def device(self) -> torch.device:
        
        return self._x.device

    def pt(self, index: int) -> torch.Tensor:
        """Retrieve a given point in the shape parameters

        Args:
            index (int): The point to retrieve

        Returns:
            torch.Tensor: _description_
        """

        assert isinstance(index, int)
        return self._x[:,:,:,index]

    def sample(self, index: int) -> torch.Tensor:
        """Retrieve one sample from the shape parameters

        Args:
            index (int): The sampel to retrieve

        Returns:
            torch.Tensor: 
        """
        return self._x[index]

    def samples(self, indices) -> torch.Tensor:
        """Retrieve multiple samples

        Args:
            indices (typing.Iterable[int]): The indices of the samples to retrieve

        Returns:
            torch.Tensor: The samples to retrieve
        """
        return self._x[indices]
        
    @property
    def x(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: The data stored in the ShapeParams
        """
        return self._x

    @property
    def batch_size(self) -> int:
        return self._x.size(0)

    @property
    def set_size(self) -> int:
        return self._x.size(1)

    @property
    def n_variables(self) -> int:
        return self._x.size(1)

    @property
    def n_terms(self) -> int:
        return self._x.size(2)

    @property
    def n_points(self) -> int:
        return self._x.size(3)

    def contains(self, x: torch.Tensor, index1: int, index2: int) -> torch.BoolTensor:
        return (x >= self.pt(index1)) & (x <= self.pt(index2))

    def insert(self, x: torch.Tensor, idx: int, to_unsqueeze: bool=False, equalize_to: torch.Tensor=None):
        x = x if not to_unsqueeze else unsqueeze(x)

        mine = resize_to(self.x, x)
        if equalize_to is not None:
            mine = resize_to(mine, equalize_to, 1)
        if not (0 <= idx <= mine.size(3)):
            raise ValueError(f'Argument idx must be in range of [0, {mine.size(3)}] not {idx}')
        
        return ShapeParams(
            torch.concat([mine[:,:,:,:idx], x, mine[:,:,:,idx:]], dim=3)
        )

    def replace(self, x: torch.Tensor, idx: int, to_unsqueeze: bool=False, equalize_to: torch.Tensor=None):
        x = x if not to_unsqueeze else unsqueeze(x)
        mine = resize_to(self.x, x)
        if equalize_to is not None:
            mine = resize_to(mine, equalize_to, 1)
        if not (0 <= idx < self._x.size(3)):
            raise ValueError(f'Argument idx must be in range of [0, {mine.size(3)}) not {idx}')
        
        return ShapeParams(
            torch.concat([mine[:,:,:,:idx], x, mine[:,:,:,idx+1:]], dim=3)
        )

    def replace_slice(self, x: torch.Tensor, pt_range: typing.Tuple[int, int], to_unsqueeze: bool=False, equalize_to: torch.Tensor=None):
        x = x if not to_unsqueeze else unsqueeze(x)
        
        mine = resize_to(self.x, x)
        if equalize_to is not None:
            mine = resize_to(mine, equalize_to, 1)
        return ShapeParams(
            torch.concat([mine[:,:,:,:pt_range[0]], x, mine[:,:,:,pt_range[1]+1:]], dim=3)
        )

    @classmethod
    def from_sub(cls, *sub: 'ShapeParams'):
        
        return ShapeParams(
            torch.cat([sub_i._x for sub_i in sub], dim=3)
        )


class Polygon(Concave):
    """
    """
    PT = None

    def __init__(self, params: ShapeParams, m: typing.Optional[torch.Tensor]=None):

        assert params.x.size(3) == self.PT, f'Number of points must be {self.PT} not {params.x.size(3)}'
        self._params = params
        # Change to fuzzy.positives
        print(self._params.batch_size, self._params.n_variables, self._params.n_terms)
        self._m = m if m is not None else torch.ones(
            self._params.batch_size, self._params.set_size, 
            self._params.n_terms, device=params.x.device,        
        )

        super().__init__(self._params.set_size, self._params.n_terms)