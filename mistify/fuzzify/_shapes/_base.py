# 1st party
from abc import abstractmethod, abstractproperty
from dataclasses import dataclass
import typing

# 3rd party
import torch
import torch.nn as nn

# local
from ...utils._utils import resize_to, unsqueeze


class Shape(nn.Module):
    """Convert an input into a membership or vice-versa
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


    @property
    def n_terms(self) -> int:
        """
        Returns:
            int: The number of terms for the shape
        """
        return self._n_terms
    
    @property
    def n_variables(self) -> int:
        """
        Returns:
            int: the number of variables for the shape
        """
        return self._n_variables

    @abstractmethod
    def join(self, x: torch.Tensor) -> torch.Tensor:
        """Join the fuzzy set defined by the shape. This is the membership function

        Args:
            x (torch.Tensor): the value to join

        Returns:
            torch.Tensor: The membership value for x
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.join(x)


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
    

    # def _init_m(self, m: torch.Tensor=None, device='cpu') -> torch.Tensor:
    #     """Set m to 1 if m is None

    #     Args:
    #         m (torch.Tensor, optional): the membership. Defaults to None.
    #         device (str, optional): the device for the membership. Defaults to 'cpu'.

    #     Returns:
    #         torch.Tensor: the output membership
    #     """
    #     if m is None:
    #         return torch.tensor(1., device=device)
    #     return m.to(device)

    # @abstractproperty
    # def m(self):
    #     """
    #     Returns:
    #         torch.Tensor: The max membership for the set
    #     """
    #     pass
    
    # @abstractmethod
    # def truncate(self, m: torch.Tensor) -> 'Shape':
    #     """Truncate the shape by a membership tensor

    #     Args:
    #         m (torch.Tensor): Membership tensor to truncate by

    #     Returns:
    #         Shape: Scaled shape
    #     """
    #     pass

    # @abstractmethod
    # def _calc_areas(self):
    #     """Method to override to calculate the area of the shape
    #     """
    #     pass

class Monotonic(Shape):
    """A nondecreasing or nonincreasing shape
    """

    def __init__(self, n_variables: int, n_terms: int):
        """Create a shape

        Args:
            n_variables (int): The number of 
            n_terms (int): The number of 
        """
        super().__init__(n_variables, n_terms)
        self._min_cores = None

    @abstractmethod
    def min_cores(self, m: torch.Tensor) -> torch.Tensor:
        pass

    # @abstractmethod
    # def _calc_min_cores(self) -> torch.Tensor:
    #     """
    #     Returns:
    #         torch.Tensor: The minimum of the "core" of the shape
    #     """
    #     pass

    # @property
    # def min_cores(self) -> torch.Tensor:
    #     """
    #     Returns:
    #         torch.Tensor: The mean of the core of the shape
    #     """
    #     if self._min_cores is None:
    #         self._min_cores = self._calc_min_cores()
    #     return self._min_cores


class Nonmonotonic(Shape):
    """A shape that has both an increasing and decreasing part
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
    def mean_cores(self, m: torch.Tensor, truncate: bool=False) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: The mean of the core of the shape
        """
        pass
        # if self._mean_cores is None:
        #     self._mean_cores = self._calc_mean_cores()
        # return self._mean_cores

    @abstractmethod
    def areas(self, m: torch.Tensor, truncate: bool=False) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: Centroid for the 
        """
        pass

    @abstractmethod
    def centroids(self, m: torch.Tensor, truncate: bool=False) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: Centroid for the 
        """
        pass
        # if self._centroids is None:
        #     self._centroids = self._calc_centroids()
        # return self._centroids

    # @abstractmethod
    # def scale(self, m: torch.Tensor) -> 'Shape':
    #     """Scale the shape by a membership tensor

    #     Args:
    #         m (torch.Tensor): Membership tensor to scale by

    #     Returns:
    #         Shape: Scaled shape
    #     """
    #     pass

    # @abstractmethod
    # def _calc_centroids(self) -> torch.Tensor:
    #     """
    #     Returns:
    #         torch.Tensor: The centroid of the shape
    #     """
    #     pass

    # @abstractmethod
    # def _calc_mean_cores(self) -> torch.Tensor:
    #     """
    #     Returns:
    #         torch.Tensor: The mean of the core of the shape
    #     """
    #     pass


class ShapeParams(nn.Module):
    """A convenience class to wrap a tensor for specifying a Shape
    """

    def __init__(self, x: torch.Tensor, tunable: bool=False, descending: typing.Union[None, bool]=False):

        super().__init__()
        if x.dim() == 3:
            x = x[None]
        assert x.dim() == 4
            
        if not tunable:
            self._x = x
        else:
            self._x = nn.parameter.Parameter(x.detach())
        self._tunable = tunable
        self._descending = descending

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
        return ShapeParams(self._x[:, :, :, index], False, self._descending)

    @property
    def device(self) -> torch.device:
        
        return self._x.device

    def pt(self, index: int) -> torch.Tensor:
        """Retrieve a given point in the shape parameters

        Args:
            index (int): The point to retrieve

        Returns:
            torch.Tensor: 
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
    
    def sort(self) -> 'ShapeParams':
        """

        Args:
            descending (bool, optional): Sort the parameters by the . Defaults to False.

        Returns:
            ShapeParams: 
        """
        if self._descending is not None:
            return ShapeParams(
                self._x.sort(descending=self._descending)[0], tunable=False, descending=self._descending
            )
        return self

    def contains(self, x: torch.Tensor, index1: int, index2: int) -> torch.BoolTensor:
        return (x >= self.pt(index1)) & (x <= self.pt(index2))

    def insert(self, x: torch.Tensor, idx: int, to_unsqueeze: bool=False, equalize_to: torch.Tensor=None) -> 'ShapeParams':
        """Insert a value into the params

        Args:
            x (torch.Tensor): The value to insert
            idx (int): The index to insert at
            to_unsqueeze (bool, optional): Whether to unsqueeze x. Defaults to False.
            equalize_to (torch.Tensor, optional): Whether to equalize the size. Defaults to None.

        Returns:
            ShapeParams: The ShapeParmas inserted
        """
        x = x if not to_unsqueeze else unsqueeze(x)

        mine = resize_to(self.x, x)
        if equalize_to is not None:
            mine = resize_to(mine, equalize_to, 1)
        if not (0 <= idx <= mine.size(3)):
            raise ValueError(f'Argument idx must be in range of [0, {mine.size(3)}] not {idx}')
        
        return ShapeParams(
            torch.concat([mine[:,:,:,:idx], x, mine[:,:,:,idx:]], dim=3), False, self._descending
        )

    def replace(
        self, x: torch.Tensor, idx: int, to_unsqueeze: bool=False, 
        equalize_to: torch.Tensor=None
    ) -> 'ShapeParams':
        """Replace the value in the params

        Args:
            x (torch.Tensor): The value to replace with
            idx (int): The index to replace at
            to_unsqueeze (bool, optional): Whether to unsqueeze x. Defaults to False.
            equalize_to (torch.Tensor, optional): Whether to equalize x's size to the params. Defaults to None.

        Returns:
            ShapeParams: The ShapeParams with the value replaced
        """
        x = x if not to_unsqueeze else unsqueeze(x)
        mine = resize_to(self.x, x)
        if equalize_to is not None:
            mine = resize_to(mine, equalize_to, 1)
        if not (0 <= idx < self._x.size(3)):
            raise ValueError(f'Argument idx must be in range of [0, {mine.size(3)}) not {idx}')
        
        return ShapeParams(
            torch.concat([mine[:,:,:,:idx], x, mine[:,:,:,idx+1:]], dim=3), False, self._descending
        )

    def replace_slice(
        self, x: torch.Tensor, pt_range: typing.Tuple[int, int], 
        to_unsqueeze: bool=False, equalize_to: torch.Tensor=None
    ) -> 'ShapeParams':
        """Replace a range of values in the ShapeParams

        Args:
            x (torch.Tensor): The value to replace iwth
            pt_range (typing.Tuple[int, int]): The range of values to replace
            to_unsqueeze (bool, optional): Whether to unsqueeze x. Defaults to False.
            equalize_to (torch.Tensor, optional): Whether to equalize the shape of x to the Params. Defaults to None.

        Returns:
            ShapeParams: The ShapeParams with the values replaced
        """
        x = x if not to_unsqueeze else unsqueeze(x)
        
        mine = resize_to(self.x, x)
        if equalize_to is not None:
            mine = resize_to(mine, equalize_to, 1)
        return ShapeParams(
            torch.concat([mine[:,:,:,:pt_range[0]], x, mine[:,:,:,pt_range[1]+1:]], dim=3), False, self._descending
        )

    @classmethod
    def from_sub(cls, *sub: 'ShapeParams', tunable: bool=False, descending: bool=False):
        
        return ShapeParams(
            torch.cat([sub_i._x for sub_i in sub], dim=3), tunable, descending
        )
    
    @property
    def shape(self) -> torch.Size:
        return self.x.shape
    
    def forward(self) -> 'ShapeParams':
        return self.sort()
    
    def __call__(self) -> 'ShapeParams':
        return super().__call__()


class Polygon(Nonmonotonic):
    """A class that defines a polygonal shape
    """
    PT = None

    def __init__(self, params: ShapeParams):
        """Create a polygon consisting of Nonmonotonic shapes

        Args:
            params (ShapeParams): The parameters of the shapes
            m (typing.Optional[torch.Tensor], optional): The membership value. Defaults to None.

        Raises:
            ValueError: If the number of points is not valid
        """
        if params.x.size(3) != self.PT:
            raise ValueError(f'Number of points must be {self.PT} not {params.x.size(3)}')
        
        super().__init__(params.set_size, params.n_terms)
        self._params = params

    def params(self) -> ShapeParams:
        return self._params()
