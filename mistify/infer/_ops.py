# 1st party
import typing

# 3rd party
import torch
import torch.nn as nn

# local
from .. import _functional
from ..utils import EnumFactory
from .._functional import _set_ops as set_ops, G


class InterOnBase(nn.Module):
    """Intersect sets that comprise a fuzzy set on a dimension
    """

    F = EnumFactory(
        inter_on=_functional.inter_on,
        smooth_inter_on=_functional.smooth_inter_on,
        bounded_inter_on=_functional.bounded_inter_on,
        prob_inter_on=_functional.prob_inter_on,
        ada_inter_on=_functional.ada_inter_on
    )

    def __init__(self, f: typing.Union[typing.Callable, str]='inter_on', dim: int=-1, keepdim: bool=False):
        """Join sets that comprise a fuzzy set on a specified dimension

        Args:
            f (str, optional): The function to use for junction. Defaults to 'min'.
            dim (int, optional): Dimension to junction on. Defaults to -1.
            keepdim (bool, optional): Whether to keep the dim or not. Defaults to False.

        Raises:
            ValueError: If the junction function is invalid
        """
        super().__init__()
        self._f = self.F.f(f)
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, m: torch.Tensor, dim: int=None) -> torch.Tensor:
        return self._f(m, dim=dim if dim is not None else self.dim, keepdim=self.keepdim)


class UnionOnBase(nn.Module):
    """Union on a specific dimension
    """
    F = EnumFactory(
        union_on = _functional.union_on,
        smooth_union_on = _functional.smooth_union_on,
        bounded_inter_on=_functional.bounded_union_on,
        ada_union_on=_functional.ada_union_on,
        prob_union_on=_functional.prob_union_on
    )

    def __init__(self, f: typing.Union[typing.Callable, str]='union_on', dim: int=-1, keepdim: bool=False):
        """Union sets that comprise a fuzzy set on a specified dimension

        Args:
            f (str, optional): The function to use for union. Defaults to 'min'.
            dim (int, optional): Dimension to union on. Defaults to -1.
            keepdim (bool, optional): Whether to keep the dim or not. Defaults to False.

        Raises:
            ValueError: If the union function is invalid
        """
        super().__init__()
        self._f = self.F.f(f)
        self.dim = dim
        self.keepdim = keepdim


class UnionOn(UnionOnBase):
    """Union on a specific dimension
    """

    def __init__(self, dim: int=-1, keepdim: bool=False, g: G=None):
        """Union sets that comprise a fuzzy set on a specified dimension

        Args:
            f (str, optional): The function to use for union. Defaults to 'min'.
            dim (int, optional): Dimension to union on. Defaults to -1.
            keepdim (bool, optional): Whether to keep the dim or not. Defaults to False.

        Raises:
            ValueError: If the union function is invalid
        """
        super().__init__(_functional.union_on, dim, keepdim)
        self.g = g

    def forward(self, m: torch.Tensor, dim: int=None) -> torch.Tensor:
        """Calculate the hard union

        Args:
            m (torch.Tensor): The input
            dim (int, optional): The dimension. Defaults to None.

        Returns:
            torch.Tensor: The union on the dimension
        """
        return self._f(m, g=self.g, keepdim=self.keepdim, dim=self.dim if dim is None else dim)


class ProbUnionOn(UnionOnBase):
    """Union on a specific dimension
    """

    def __init__(self, dim: int=-1, keepdim: bool=False):
        """Union sets that comprise a fuzzy set on a specified dimension

        Args:
            f (str, optional): The function to use for union. Defaults to 'min'.
            dim (int, optional): Dimension to union on. Defaults to -1.
            keepdim (bool, optional): Whether to keep the dim or not. Defaults to False.

        Raises:
            ValueError: If the union function is invalid
        """
        super().__init__(_functional.prob_union_on, dim, keepdim)

    def forward(self, m: torch.Tensor, dim: int=None) -> torch.Tensor:
        """Calculate the probabilistic union

        Args:
            m (torch.Tensor): The input
            dim (int, optional): The dimension. Defaults to None.

        Returns:
            torch.Tensor: The union on the dimension
        """
        return self._f(m, dim=self.dim if dim is None else dim)


class SmoothUnionOn(UnionOnBase):
    """Union on a specific dimension
    """

    def __init__(self, dim: int=-1, keepdim: bool=False, a: float=None):
        """Union sets that comprise a fuzzy set on a specified dimension

        Args:
            f (str, optional): The function to use for union. Defaults to 'min'.
            dim (int, optional): Dimension to union on. Defaults to -1.
            keepdim (bool, optional): Whether to keep the dim or not. Defaults to False.

        Raises:
            ValueError: If the union function is invalid
        """
        super().__init__(_functional.smooth_union_on, dim, keepdim)
        self.a = a

    def forward(self, m: torch.Tensor, dim: int=None) -> torch.Tensor:
        """Calculate the smooth union

        Args:
            m (torch.Tensor): The input
            dim (int, optional): The dimension. Defaults to None.

        Returns:
            torch.Tensor: The union on the dimension
        """
        return self._f(m, a=self.a, keepdim=self.keepdim, dim=self.dim if dim is None else dim)


class BoundedUnionOn(UnionOnBase):
    """Union on a specific dimension
    """

    def __init__(self, dim: int=-1, keepdim: bool=False, g: G=None):
        """Union sets that comprise a fuzzy set on a specified dimension

        Args:
            f (str, optional): The function to use for union. Defaults to 'min'.
            dim (int, optional): Dimension to union on. Defaults to -1.
            keepdim (bool, optional): Whether to keep the dim or not. Defaults to False.

        Raises:
            ValueError: If the union function is invalid
        """
        super().__init__(_functional.bounded_union_on, dim, keepdim)
        self.g = g

    def forward(self, m: torch.Tensor, dim: int=None) -> torch.Tensor:
        """Calculate the bounded union

        Args:
            m (torch.Tensor): The input
            dim (int, optional): The dimension. Defaults to None.

        Returns:
            torch.Tensor: The union on the dimension
        """
        return self._f(m, g=self.g, keepdim=self.keepdim, dim=self.dim if dim is None else dim)


class InterOn(InterOnBase):
    """Union on a specific dimension
    """

    def __init__(self, dim: int=-1, keepdim: bool=False, g: G=None):
        """Union sets that comprise a fuzzy set on a specified dimension

        Args:
            f (str, optional): The function to use for union. Defaults to 'min'.
            dim (int, optional): Dimension to union on. Defaults to -1.
            keepdim (bool, optional): Whether to keep the dim or not. Defaults to False.

        Raises:
            ValueError: If the union function is invalid
        """
        super().__init__(_functional.inter_on, dim, keepdim)
        self.g = g

    def forward(self, m: torch.Tensor, dim: int=None) -> torch.Tensor:
        """Calculate the hard intersection

        Args:
            m (torch.Tensor): The input
            dim (int, optional): The dimension. Defaults to None.

        Returns:
            torch.Tensor: The intersection on the dimension
        """
        dim = dim or self.dim
        return self._f(m, g=self.g,keepdim=self.keepdim,  dim=self.dim if dim is None else dim)


class ProbInterOn(InterOnBase):
    """Union on a specific dimension
    """

    def __init__(self, dim: int=-1, keepdim: bool=False):
        """Union sets that comprise a fuzzy set on a specified dimension

        Args:
            f (str, optional): The function to use for union. Defaults to 'min'.
            dim (int, optional): Dimension to union on. Defaults to -1.
            keepdim (bool, optional): Whether to keep the dim or not. Defaults to False.

        Raises:
            ValueError: If the union function is invalid
        """
        super().__init__(_functional.prob_inter_on, dim, keepdim)

    def forward(self, m: torch.Tensor, dim: int=None) -> torch.Tensor:
        """Calculate the probabilistic intersection

        Args:
            m (torch.Tensor): The input
            dim (int, optional): The dimension. Defaults to None.

        Returns:
            torch.Tensor: The intersection on the dimension
        """
        return self._f(m, keepdim=self.keepdim, dim=self.dim if dim is None else dim)


class SmoothInterOn(InterOnBase):
    """Union on a specific dimension
    """

    def __init__(self, dim: int=-1, keepdim: bool=False, a: float=None):
        """Union sets that comprise a fuzzy set on a specified dimension

        Args:
            f (str, optional): The function to use for union. Defaults to 'min'.
            dim (int, optional): Dimension to union on. Defaults to -1.
            keepdim (bool, optional): Whether to keep the dim or not. Defaults to False.

        Raises:
            ValueError: If the union function is invalid
        """
        super().__init__(_functional.smooth_inter_on, dim, keepdim)
        self.a = a

    def forward(self, m: torch.Tensor, dim: int=None) -> torch.Tensor:
        """Calculate the smooth interssection on a dimension 

        Args:
            m (torch.Tensor): The input
            dim (int, optional): The dimension. Defaults to None.

        Returns:
            torch.Tensor: The intersection on the dimension
        """
        return self._f(
            m, a=self.a, keepdim=self.keepdim, 
            dim=self.dim if dim is None else dim
        )


class BoundedInterOn(InterOnBase):
    """Union on a specific dimension
    """

    def __init__(self, dim: int=-1, keepdim: bool=False, g: G=None):
        """Union sets that comprise a fuzzy set on a specified dimension

        Args:
            f (str, optional): The function to use for union. Defaults to 'min'.
            dim (int, optional): Dimension to union on. Defaults to -1.
            keepdim (bool, optional): Whether to keep the dim or not. Defaults to False.

        Raises:
            ValueError: If the union function is invalid
        """
        super().__init__(_functional.bounded_inter_on, dim, keepdim)
        self.g = g

    def forward(self, m: torch.Tensor, dim: int=None) -> torch.Tensor:
        """Compute the intersection on the specified dimension of m

        Args:
            m (torch.Tensor): The membership
            dim (int, optional): The dimension to compute. Defaults to None.

        Returns:
            torch.Tensor: The intersection
        """
        return self._f(m, g=self.g, keepdim=self.keepdim, dim=self.dim if dim is None else dim)


class InterBase(nn.Module):
    """The base class for performing a intersection
    """

    def __init__(self, f: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        """Create an intersection module

        Args:
            f (typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): The function used for intersection

        """
        super().__init__()
        self._f = f

    def forward(self, m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
        
        return self._f(m1, m2)


class UnionBase(nn.Module):
    """The base class for performing a union
    """

    def __init__(self, f: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        """Create a union module using a union funciotn

        Args:
            f (typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): The function to perform the union

        """
        super().__init__()
        self._f = f

    def forward(self, m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
        """Caclculate the union of two sets

        Args:
            m1 (torch.Tensor): Membership tensor 1
            m2 (torch.Tensor): Membership tensor 2

        Returns:
            torch.Tensor: The union
        """
        return self._f(m1, m2)


class Inter(InterBase):
    """The base class for performing an intersection
    """

    def __init__(self, g: G=None):
        """Create a standard intersection using the "hard" intersection

        Args:
            g (G, optional): The gradient estimator. Defaults to None.
        """
        super().__init__(_functional.inter)
        self.g = g

    def forward(self, m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
        """Caclculate the intersection of two sets

        Args:
            m1 (torch.Tensor): Membership tensor 1
            m2 (torch.Tensor): Membership tensor 2

        Returns:
            torch.Tensor: The intersection
        """
        return self._f(m1, m2, g=self.g)


class ProbInter(InterBase):
    """ProbInter uses the multiplication of the two memberships 
    """

    def __init__(self) -> torch.Tensor:
        """Create a probabilistic intersection using multiply for intersection
        """
        super().__init__(_functional.prob_inter)

    def forward(self, m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
        """Caclculate the intersection of two sets

        Args:
            m1 (torch.Tensor): Membership tensor 1
            m2 (torch.Tensor): Membership tensor 2

        Returns:
            torch.Tensor: The intersection
        """
        return self._f(m1, m2)


class SmoothInter(InterBase):
    """SmoothInter uses the softmin function to compute the intersection
    """

    def __init__(self, a: float=None) -> torch.Tensor:
        """Create a standard intersection using the "hard" intersection

        Args:
            g (G, optional): The gradient estimator. Defaults to None.
        """
        super().__init__(_functional.smooth_inter)
        self.a = a

    def forward(self, m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
        """Caclculate the intersection of two sets

        Args:
            m1 (torch.Tensor): Membership tensor 1
            m2 (torch.Tensor): Membership tensor 2

        Returns:
            torch.Tensor: The intersection
        """
        return self._f(m1, m2, a=self.a)


class BoundedInter(InterBase):
    """BoundedInter uses the sum of the two memberships - 1 bounded by 0
    """

    def __init__(self, g: G=None) -> torch.Tensor:
        """Create a bounded intersection using the intersection

        Args:
            g (G, optional): The gradient estimator. Defaults to None.
        """
        super().__init__(_functional.bounded_inter)
        self.g = g

    def forward(self, m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
        """Caclculate the intersection of two sets

        Args:
            m1 (torch.Tensor): Membership tensor 1
            m2 (torch.Tensor): Membership tensor 2

        Returns:
            torch.Tensor: The intersection
        """
        return self._f(m1, m2, g=self.g)


class Union(UnionBase):
    """Union uses the max to compute the union
    """

    def __init__(self, g: G=None) -> torch.Tensor:
        """Create a union using the "hard" union

        Args:
            g (G, optional): The gradient estimator. Defaults to None.
        """
        super().__init__(_functional.union)
        self.g = g

    def forward(self, m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
        """Caclculate the union of two sets

        Args:
            m1 (torch.Tensor): Membership tensor 1
            m2 (torch.Tensor): Membership tensor 2

        Returns:
            torch.Tensor: The union
        """
        return self._f(m1, m2, g=self.g)


class ProbUnion(UnionBase):
    """ProbUnion uses the probabilistic sum to compute the union
    """

    def __init__(self) -> torch.Tensor:
        """Create a union using the probabilistic union

        Args:
            g (G, optional): The gradient estimator. Defaults to None.
        """
        super().__init__(_functional.prob_union)

    def forward(self, m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
        """Caclculate the union of two sets

        Args:
            m1 (torch.Tensor): Membership tensor 1
            m2 (torch.Tensor): Membership tensor 2

        Returns:
            torch.Tensor: The union
        """
        return self._f(m1, m2)


class SmoothUnion(UnionBase):
    """SmoothUnion uses the softmax to compute the union
    """

    def __init__(self, a: float=None) -> torch.Tensor:
        """Create a union using the smooth union

        Args:
            a (a, optional): The smoothness parameter. Defaults to None.
        """
        super().__init__(_functional.smooth_union)
        self.a = a

    def forward(self, m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
        """Caclculate the smooth union of two sets

        Args:
            m1 (torch.Tensor): Membership tensor 1
            m2 (torch.Tensor): Membership tensor 2

        Returns:
            torch.Tensor: The union
        """
        return self._f(m1, m2, a=self.a)


class BoundedUnion(UnionBase):
    """BoundedUnion is the sum of the two memberships bounded by one
    """

    def __init__(self, g: G=None) -> torch.Tensor:
        """Create a union using the bounded union

        Args:
            g (G, optional): The gradient estimator. Defaults to None.
        """
        super().__init__(_functional.bounded_union)
        self.g = g

    def forward(self, m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
        """Caclculate the bounded union of two sets

        Args:
            m1 (torch.Tensor): Membership tensor 1
            m2 (torch.Tensor): Membership tensor 2

        Returns:
            torch.Tensor: The union
        """
        return self._f(m1, m2, g=self.g)


class Else(nn.Module):
    """A module to compute the "else"
    """

    F = EnumFactory(
        else_=set_ops.else_,
        signed_else=set_ops.signed_else_
    )

    def __init__(self, f: typing.Callable='else_', dim=-1, keepdim: bool = False):
        """Calculate else along a certain dimension It calculates the sum of all the membership values along the dimension

        Args:
            f (typing.Callable, optional): The else function. Defaults to 'fuzzy'.
            dim (int, optional): The dimension to take the else on. Defaults to -1.
            keepdim (bool, optional): Whether to keep the dimension after taking the else. Defaults to False.
        """
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        self._f = self.F.f(f)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """Calculate the else for the fuzzy set

        Args:
            m (torch.Tensor): the membership value

        Returns:
            torch.Tensor: the else of the fuzzy set
        """
        return self._f(m, dim=self.dim, keepdim=self.keepdim)


class CatElse(nn.Module):
    """Else accompanied by a concatenation of the input
    """

    def __init__(self, f: typing.Callable='else_', dim=-1):
        """Create an "Else" of the fuzzy set and then concatenate with the input

        Args:
            f (typing.Callable, optional): The else function. Defaults to 'fuzzy'.
            dim (int, optional): The dimension to take the else on. Defaults to -1.
        """
        super().__init__()
        self._else = Else(f, dim, True)

    def forward(self, m: torch.Tensor) -> torch.Tensor:

        else_ = self._else(m)
        return torch.cat(
            [m, else_], dim=self._else.dim
        )


class Complement(nn.Module):

    F = EnumFactory(
        complement = set_ops.complement,
        signed_complement = set_ops.signed_complement,
    )

    def __init__(self, f: typing.Callable='complement'):
        """Take the complement of the set

        Args:
            f (typing.Callable, optional): The complement function. Defaults to 'boolean'.
        """
        super().__init__()
        self._f = self.F.f(f)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """Calculate the complement for the set

        Args:
            m (torch.Tensor): the membership value

        Returns:
            torch.Tensor: the complement of the set
        """
        return self._f(m)


class CatComplement(nn.Module):

    def __init__(self, f: typing.Callable='complement', dim=-1):
        """Take the complement and then concatenate it

        Args:
            f (typing.Callable, optional): The complement function. Defaults to 'boolean'.
            dim (int, optional): The dim to cat on. Defaults to -1.
        """
        super().__init__()
        self._complement = Complement(f)
        self.dim = dim

    def forward(self, m: torch.Tensor) -> torch.Tensor:

        complement = self._complement(m)
        return torch.cat(
            [m, complement], dim=self.dim
        )
