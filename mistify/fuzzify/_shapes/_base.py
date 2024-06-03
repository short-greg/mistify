# 1st party
from abc import abstractmethod
import typing

# 3rd party
import torch
import torch.nn as nn
import torch.nn.functional

# local
from ...utils._utils import resize_dim_to, unsqueeze
from ..._base import Constrained


class Shape(nn.Module, Constrained):
    """Convert an input into a membership or vice-versa
    """

    def __init__(self, n_vars: int, n_terms: int):
        """Create the shape

        Args:
            n_vars (int): the number of linguistic variables
            n_terms (int): the number of terms for each variable
        """
        super().__init__()
        self._n_vars = n_vars
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
    def n_vars(self) -> int:
        """
        Returns:
            int: the number of variables for the shape
        """
        return self._n_vars

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
        repeat = [1] * m.dim()
        if x.size(0) == 1 and m.size(0) != 1:
            repeat[0] = m.size(0) 
        elif x.size(0) != m.size(0):
            raise ValueError(
                'Cannot resize to m since xs dimension '
                f'is {x.size(0)} not 1')
        if x.size(1) == 1 and m.size(1) != 1:
            repeat[1] = m.size(1)
        elif x.size(1) != m.size(1):
            raise ValueError(
                'Cannot resize to m since xs dimension '
                f'is {x.size(1)} not 1')
        return x.repeat(repeat)

    def constrain(self):
        pass


class Monotonic(Shape):
    """A nondecreasing or nonincreasing shape
    """

    def __init__(self, n_vars: int, n_terms: int):
        """Create a shape

        Args:
            n_vars (int): The number of 
            n_terms (int): The number of 
        """
        super().__init__(n_vars, n_terms)
        self._min_cores = None

    @abstractmethod
    def min_cores(self, m: torch.Tensor) -> torch.Tensor:
        pass


class Nonmonotonic(Shape):
    """A shape that has both an increasing and decreasing part
    """

    def __init__(self, n_vars: int, n_terms: int):
        """

        Args:
            n_vars (int): the number of linguistic variables
            n_terms (int): the number of terms for each variable
        """
        super().__init__(n_vars, n_terms)
        self._mean_cores = None
        self._centroids = None


    @abstractmethod
    def mean_cores(self, m: torch.Tensor, truncate: bool=False) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: The mean of the core of the shape
        """
        pass

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
            torch.Tensor: Centroid for the shape
        """
        pass


class Coords(nn.Module):
    """A convenience class to wrap a tensor for specifying a Shape
    """

    def __init__(self, x: torch.Tensor):

        super().__init__()
        if x.dim() == 3:
            x = x[None]
        assert x.dim() == 4

        x_base = x[...,:-1]
        x_offset = x[...,1:]
        dx = x_offset - x_base
        dx = torch.log(torch.exp(dx) - 1)

        if dx.isnan().any():
            raise ValueError(
                'The coordinates must be monotonically increasing. '
                'It seems there are some that are <= to the previous.')
    
        self._x = nn.parameter.Parameter(x_base[...,0:1].detach())
        self._dx = nn.parameter.Parameter(dx.detach())

    @property
    def device(self) -> torch.device:
        
        return self._x.device
    
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
    def n_vars(self) -> int:
        return self._x.size(1)

    @property
    def n_terms(self) -> int:
        return self._x.size(2)

    @property
    def n_points(self) -> int:
        return self._x.size(3) + self._dx.size(3)

    @property
    def shape(self) -> torch.Size:
        return self._x.shape
    
    def forward(self) -> torch.Tensor:
        
        others = torch.cumsum(
            torch.nn.functional.softplus(self._dx), -1
        ) + self._x
        return torch.cat(
            [self._x, others], dim=-1
        )

    # def constrain(self, eps: float=1e-7):

    #     prev = None
    #     for i in range(self.n_points):
    #         if prev is not None:
    #             if self._descending:
    #                 self._x[...,prev].data = torch.clamp(
    #                     self._x[...,prev], self._x[...,prev], self._x[...,i] - eps
    #                 )
    #             else:
    #                 self._x[...,i].data = torch.clamp(
    #                     self._x[...,i], self._x[...,prev] + eps, self._x[...,i]
    #                 )
    #         prev = i

    # def sub(self, index: typing.Union[int, typing.Tuple[int, int]]) -> 'Coords':
    #     """Extract a subset of the parameters

    #     Args:
    #         index (typing.Union[int, typing.Tuple[int, int]]): Index to extract with

    #     Returns:
    #         ShapeParams: Subset of the shape parameters
    #     """
    #     if isinstance(index, int):
    #         index = slice(index, index + 1)
    #     elif isinstance(index, typing.List):
    #         index = index
    #     else:
    #         index = slice(*index)
    #     return Coords(self._x[..., index], False, self._descending)

    # def pt(self, index: int) -> torch.Tensor:
    #     """Retrieve a given point in the shape parameters

    #     Args:
    #         index (int): The point to retrieve

    #     Returns:
    #         torch.Tensor: 
    #     """
    #     assert isinstance(index, int)
    #     return self._x[...,index]

    # def sample(self, index: int) -> torch.Tensor:
    #     """Retrieve one sample from the shape parameters

    #     Args:
    #         index (int): The sampel to retrieve

    #     Returns:
    #         torch.Tensor: 
    #     """
    #     return self._x[index]

    # def samples(self, indices) -> torch.Tensor:
    #     """Retrieve multiple samples

    #     Args:
    #         indices (typing.Iterable[int]): The indices of the samples to retrieve

    #     Returns:
    #         torch.Tensor: The samples to retrieve
    #     """
    #     return self._x[indices]
    # def sort(self) -> 'Coords':
    #     """

    #     Args:
    #         descending (bool, optional): Sort the parameters by the . Defaults to False.

    #     Returns:
    #         ShapeParams: 
    #     """
    #     if self._descending is not None:
    #         return Coords(
    #             self._x.sort(descending=self._descending)[0], tunable=False, descending=self._descending
    #         )
    #     return self

    def contains(self, x: torch.Tensor, index1: int, index2: int) -> torch.BoolTensor:
        pts = self()
        return (x >= pts[...,index1]) & (x <= pts[...,index2])

    # def insert(self, x: torch.Tensor, idx: int, to_unsqueeze: bool=False, equalize_to: torch.Tensor=None) -> 'Coords':
    #     """Insert a value into the params

    #     Args:
    #         x (torch.Tensor): The value to insert
    #         idx (int): The index to insert at
    #         to_unsqueeze (bool, optional): Whether to unsqueeze x. Defaults to False.
    #         equalize_to (torch.Tensor, optional): Whether to equalize the size. Defaults to None.

    #     Returns:
    #         ShapeParams: The ShapeParmas inserted
    #     """
    #     x = x if not to_unsqueeze else unsqueeze(x)

    #     mine = resize_to(self.x, x)
    #     if equalize_to is not None:
    #         mine = resize_to(mine, equalize_to, 1)
    #     if not (0 <= idx <= mine.size(3)):
    #         raise ValueError(f'Argument idx must be in range of [0, {mine.size(3)}] not {idx}')
        
    #     return Coords(
    #         torch.concat([mine[:,:,:,:idx], x, mine[:,:,:,idx:]], dim=3), False, self._descending
    #     )

    # def replace(
    #     self, x: torch.Tensor, idx: int, to_unsqueeze: bool=False, 
    #     equalize_to: torch.Tensor=None
    # ) -> 'Coords':
    #     """Replace the value in the params

    #     Args:
    #         x (torch.Tensor): The value to replace with
    #         idx (int): The index to replace at
    #         to_unsqueeze (bool, optional): Whether to unsqueeze x. Defaults to False.
    #         equalize_to (torch.Tensor, optional): Whether to equalize x's size to the params. Defaults to None.

    #     Returns:
    #         ShapeParams: The ShapeParams with the value replaced
    #     """
    #     x = x if not to_unsqueeze else unsqueeze(x)
    #     mine = resize_to(self.x, x)
    #     if equalize_to is not None:
    #         mine = resize_to(mine, equalize_to, 1)
    #     if not (0 <= idx < self._x.size(3)):
    #         raise ValueError(f'Argument idx must be in range of [0, {mine.size(3)}) not {idx}')
        
    #     return Coords(
    #         torch.concat([mine[:,:,:,:idx], x, mine[:,:,:,idx+1:]], dim=3), False, self._descending
    #     )

    # def replace_slice(
    #     self, x: torch.Tensor, pt_range: typing.Tuple[int, int], 
    #     to_unsqueeze: bool=False, equalize_to: torch.Tensor=None
    # ) -> 'Coords':
    #     """Replace a range of values in the ShapeParams

    #     Args:
    #         x (torch.Tensor): The value to replace iwth
    #         pt_range (typing.Tuple[int, int]): The range of values to replace
    #         to_unsqueeze (bool, optional): Whether to unsqueeze x. Defaults to False.
    #         equalize_to (torch.Tensor, optional): Whether to equalize the shape of x to the Params. Defaults to None.

    #     Returns:
    #         ShapeParams: The ShapeParams with the values replaced
    #     """
    #     x = x if not to_unsqueeze else unsqueeze(x)
        
    #     mine = resize_to(self.x, x)
    #     if equalize_to is not None:
    #         mine = resize_to(mine, equalize_to, 1)
    #     return Coords(
    #         torch.concat([mine[:,:,:,:pt_range[0]], x, mine[:,:,:,pt_range[1]+1:]], dim=3), False, self._descending
    #     )

    # @classmethod
    # def from_sub(cls, *sub: 'Coords', tunable: bool=False, descending: bool=False):
        
    #     return Coords(
    #         torch.cat([sub_i._x for sub_i in sub], dim=3), tunable, descending
    #     )


def insert(insert_to: torch.Tensor, to_insert: torch.Tensor, idx: int, to_unsqueeze: bool=False, equalize_to: torch.Tensor=None) -> torch.Tensor:
    """Insert a value into the params

    Args:
        to_insert (torch.Tensor): The value to insert
        idx (int): The index to insert at
        to_unsqueeze (bool, optional): Whether to unsqueeze x. Defaults to False.
        equalize_to (torch.Tensor, optional): Whether to equalize the size. Defaults to None.

    Returns:
        torch.Tensor: The ShapeParmas inserted
    """
    to_insert = to_insert if not to_unsqueeze else unsqueeze(to_insert)

    mine = resize_dim_to(insert_to, to_insert)
    if equalize_to is not None:
        mine = resize_dim_to(mine, equalize_to, 1)
    # if not (0 <= idx <= mine.size(3)):
    #     raise ValueError(f'Argument idx must be in range of [0, {mine.size(3)}] not {idx}')
    
    return torch.concat([mine[...,:idx], to_insert, mine[...,idx:]], dim=3)


def replace(
    insert_to: torch.Tensor, to_insert: torch.Tensor, idx: int, to_unsqueeze: bool=False, 
    equalize_to: torch.Tensor=None
) -> torch.Tensor:
    """Replace the value in the params

    Args:
        x (torch.Tensor): The value to replace with
        idx (int): The index to replace at
        to_unsqueeze (bool, optional): Whether to unsqueeze x. Defaults to False.
        equalize_to (torch.Tensor, optional): Whether to equalize x's size to the params. Defaults to None.

    Returns:
        ShapeParams: The ShapeParams with the value replaced
    """
    to_insert = to_insert if not to_unsqueeze else unsqueeze(to_insert)
    mine = resize_dim_to(insert_to, to_insert)
    if equalize_to is not None:
        mine = resize_dim_to(mine, equalize_to, 1)

    return torch.concat([mine[...,:idx], to_insert, mine[...,idx+1:]], dim=3)


def replace_slice(
    insert_to: torch.Tensor, to_insert: torch.Tensor, pt_range: typing.Tuple[int, int], 
    to_unsqueeze: bool=False, equalize_to: torch.Tensor=None
) -> torch.Tensor:
    """Replace a range of values in the ShapeParams

    Args:
        insert_to (torch.Tensor): The value to replace 
        to_insert (torch.Tensor): The value to replace with
        pt_range (typing.Tuple[int, int]): The range of values to replace
        to_unsqueeze (bool, optional): Whether to unsqueeze x. Defaults to False.
        equalize_to (torch.Tensor, optional): Whether to equalize the shape of x to the Params. Defaults to None.

    Returns:
        ShapeParams: The ShapeParams with the values replaced
    """
    to_insert = to_insert if not to_unsqueeze else unsqueeze(to_insert)
    
    mine = resize_dim_to(insert_to, to_insert)
    if equalize_to is not None:
        mine = resize_dim_to(mine, equalize_to, 1)
    return torch.concat([mine[...,:pt_range[0]], to_insert, mine[...,pt_range[1]+1:]], dim=3)


class Polygon(Nonmonotonic):
    """A class that defines a polygonal shape
    """
    PT = None

    def __init__(self, coords: Coords):
        """Create a polygon consisting of Nonmonotonic shapes

        Args:
            params (ShapeParams): The parameters of the shapes
            m (typing.Optional[torch.Tensor], optional): The membership value. Defaults to None.

        Raises:
            ValueError: If the number of points is not valid
        """
        if coords.n_points != self.PT:
            raise ValueError(f'Number of points must be {self.PT} not {coords.n_points}')
        
        super().__init__(coords.n_vars, coords.n_terms)
        self._coords = coords

    def coords(self) -> Coords:
        return self._coords()

    def constrain(self):
        self._coords.constrain(self._eps)

