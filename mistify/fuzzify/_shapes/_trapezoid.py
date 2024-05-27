from ._base import Polygon, Coords, replace, replace_slice
from ...utils import unsqueeze
import torch
from ... import _functional as functional
from ..._functional import G


def trapezoid_area(a: torch.Tensor, b: torch.Tensor, height: torch.Tensor) -> torch.Tensor:
    """
    Args:
        a (torch.Tensor): The length of the top
        b (torch.Tensor): The length of b
        height (torch.Tensor): The height

    Returns:
        torch.Tensor: the area
    """
    return (a + b) * height / 2.0


def trapezoid_centroid(a: torch.Tensor, b: torch.Tensor, height: torch.Tensor) -> torch.Tensor:
    """
    Args:
        a (torch.Tensor): The length of the top
        b (torch.Tensor): The length of b
        height (torch.Tensor): The height

    Returns:
        torch.Tensor: the centroid
    """
    return height * (b + 2 * a) / (3 * (a + b))


def trapezoid_mean_core(upper1: torch.Tensor, upper2: torch.Tensor) -> torch.Tensor:
    """
    Args:
        upper1 (torch.Tensor): The first point on the upper
        upper2 (torch.Tensor): The second point on the upper half of the trapezoid

    Returns:
        torch.Tensor: The mean core of the trapezoid
    """
    return (upper1 + upper2) / 2.0


class Trapezoid(Polygon):
    """A general trapezoid consisting of four points
    """

    PT = 4

    def __init__(self, coords: Coords, g: G=None):
        super().__init__(coords)
        self.g = g

    def join(self, x: torch.Tensor) -> torch.Tensor:
        """Join calculates the membership value for each section of trapezoid and uses the maximimum value
        as the value

        Args:
            x (torch.Tensor): The value to calculate the membership for

        Returns:
            torch.Tensor: The membership
        """
        params = self.coords()
        x = unsqueeze(x)
        # m1 = calc_m_linear_increasing(x, params.pt(0), params.pt(1), self._m)
        # m2 = calc_m_flat(x, params.pt(1), params.pt(2), self._m)
        # m3 = calc_m_linear_decreasing(x, params.pt(2), params.pt(3), self._m)

        # return torch.max(torch.max(m1, m2), m3)

        return functional.shape.trapezoid(
            x, params[...,0], params[...,1], params[...,2], params[...,3], g=self.g
        )

    def a(self, params: torch.Tensor=None) -> torch.Tensor:
        """
        Args:
            params (ShapeParams, optional): the params for the trapezoid. Defaults to None.

        Returns:
            torch.Tensor: Calculate the top length
        """

        params = params if params is not None else self.coords()
        return (
            params[...,3] - params[...,0]
        )

    def b(self, params: torch.Tensor=None) -> torch.Tensor:
        """
        Args:
            params (ShapeParams, optional): the params for the trapezoid. Defaults to None.

        Returns:
            torch.Tensor: The bottom length
        """
        params = params if params is not None else self.coords()
        return params[...,2] - params[...,1]

    def truncate(self, m: torch.Tensor) -> torch.Tensor:
        """

        Args:
            m (torch.Tensor): The membership

        Returns:
            torch.Tensor: Truncate the trapezoid
        """
        params = self.coords()
        new_pt1 = (params[...,1] * (1 - m) + params[...,0] * m)
        new_pt2 = params[...,3] * m + params[...,2] * (1 - m)
        params = replace(params, new_pt1, 1, to_unsqueeze=True)
        params = replace(params, new_pt2, 2, to_unsqueeze=True)
        return params

    def areas(self, m: torch.Tensor, truncate: bool = False) -> torch.Tensor:
        """

        Args:
            m (torch.Tensor): The membership
            truncate (bool, optional): Whether to truncate or scale. Defaults to False.

        Returns:
            torch.Tensor: The areas of the trapezoid
        """
        if truncate:
            params = self.truncate(m)
        else:
            params = self.coords()
            
        a = self.a(params)
        b = self.b(params)

        return trapezoid_area(a, b, m)
    
    def mean_cores(self, m: torch.Tensor, truncate: bool = False) -> torch.Tensor:
        """

        Args:
            m (torch.Tensor): The membership
            truncate (bool, optional): Whether to truncate or scale. Defaults to False.

        Returns:
            torch.Tensor: The mean cores of the trapezoid
        """
        params = self._coords()
        return self._resize_to_m(
             trapezoid_mean_core(params[...,1], params[...,2]),
             m
        )

    def centroids(self, m: torch.Tensor, truncate: bool = False) -> torch.Tensor:
        """

        Args:
            m (torch.Tensor): The membership
            truncate (bool, optional): Whether to truncate or scale. Defaults to False.

        Returns:
            torch.Tensor: The centroids of the trapezoid
        """
        # x = (b+2a)/(3(a+b))h
        if truncate:
            params = self.truncate(m)
        else:
            params = self._coords()
        
        a = self.a(params)
        b = self.b(params)
        return trapezoid_centroid(a, b, m)


class IsoscelesTrapezoid(Polygon):

    PT = 3

    def __init__(self, coords: Coords, g: G=None):
        super().__init__(coords)
        self.g = g

    def join(self, x: torch.Tensor) -> 'torch.Tensor':
        """Calculates the membership value for each part of the isosceles
        trapezoid and takes the maximum

        Args:
            x (torch.Tensor): The value to get the membership value for

        Returns:
            torch.Tensor: The membership value of x
        """
        x = x[...,None]
        params = self.coords()
        return functional.shape.isosceles_trapezoid(
            x, params[...,0], params[...,1], params[...,2], g=self.g
        )
    
    def a(self, params: torch.Tensor=None) -> torch.Tensor:

        """
        Args:
            params (ShapeParams, optional): the params for the trapezoid. Defaults to None.

        Returns:
            torch.Tensor: Calculate the top length
        """
        params = params if params is not None else self.coords()

        dx = params[...,1] - params[...,0]
        return (
            params[...,2] + dx - params[...,0]
        )

    def b(self, params: torch.Tensor=None):
        """
        Args:
            params (ShapeParams, optional): the params for the trapezoid. Defaults to None.

        Returns:
            torch.Tensor: Calculate the bottom length
        """
        params = params if params is not None else self.coords()
        return params[...,2] - params[...,1]

    def truncate(self, m: torch.Tensor) -> torch.Tensor:
        """

        Args:
            m (torch.Tensor): The membership

        Returns:
            torch.Tensor: Truncate the trapezoid
        """
        params = self._coords()
        new_pt1 = (params[...,0] * (1 - m) + params[...,1] * m)
        new_pt2 = 2 * params[...,2] - new_pt1
        params = replace(params, new_pt1, 1, to_unsqueeze=True)
        params = replace(params, new_pt2, 2, to_unsqueeze=True)
        return params

    def areas(self, m: torch.Tensor, truncate: bool = False) -> torch.Tensor:
        """

        Args:
            m (torch.Tensor): The membership
            truncate (bool, optional): Whether to truncate or scale. Defaults to False.

        Returns:
            torch.Tensor: The areas of the trapezoid
        """
        if truncate:
            params = self.truncate(m)
        else:
            params = self._coords()
            
        a = self.a(params)
        b = self.b(params)

        return trapezoid_area(a, b, m)
    
    def mean_cores(self, m: torch.Tensor, truncate: bool = False) -> torch.Tensor:
        """

        Args:
            m (torch.Tensor): The membership
            truncate (bool, optional): Whether to truncate or scale. Defaults to False.

        Returns:
            torch.Tensor: The mean cores of the trapezoid
        """
        params = self._coords()
        return self._resize_to_m(
            trapezoid_mean_core(params[...,1], params[...,2]), m
        )

    def centroids(self, m: torch.Tensor, truncate: bool = False) -> torch.Tensor:
        """

        Args:
            m (torch.Tensor): The membership
            truncate (bool, optional): Whether to truncate or scale. Defaults to False.

        Returns:
            torch.Tensor: The centroids of the trapezoid
        """
        # x = (b+2a)/(3(a+b))h
        if truncate:
            params = self.truncate(m)
        else:
            params = self._coords()
        
        a = self.a(params)
        b = self.b(params)
        return self._resize_to_m(
            trapezoid_centroid(a, b, m), m
        )


class RightTrapezoid(Polygon):

    PT = 3

    def __init__(self, params: Coords, increasing: bool=True, g: G=None):
        super().__init__(params)
        self.increasing = increasing
        self.g = g

    def join(self, x: torch.Tensor) -> 'torch.Tensor':
        """Join calculates the membership value for each section of trapezoid and uses the maximimum value
        as the value

        Args:
            x (torch.Tensor): The value to calculate the membership for

        Returns:
            torch.Tensor: The membership
        """
        params = self.coords()
        return functional.shape.right_trapezoid(
            x[...,None], params[...,0], params[...,1], params[...,2], 
            self.increasing, g=self.g
        )
    
    def a(self, params: torch.Tensor=None) -> torch.Tensor:
        """
        Args:
            params (ShapeParams, optional): the params for the trapezoid. Defaults to None.

        Returns:
            torch.Tensor: Calculate the top length
        """
        params = params if params is not None else self.coords()
        return (
            params[...,2] - params[...,0]
        )

    def b(self, params: Coords=None) -> torch.Tensor:
        """
        Args:
            params (ShapeParams, optional): the params for the trapezoid. Defaults to None.

        Returns:
            torch.Tensor: Calculate the bottom length
        """
        params = params if params is not None else self.coords()
        if self.increasing:

            return params[...,2] - params[...,1]
        return params[...,1] - params[...,0]
    
    def triangle_pts(self, params: torch.Tensor=None, to_order: bool=False) -> Coords:
        """

        Args:
            params (ShapeParams, optional): The points for the triangle part. Defaults to None.

        Returns:
            ShapeParams: The triangle points
        """
        params = params if params is not None else self.coords()

        if self.increasing:
            return params[...,[0, 1]]
        if to_order:
            return params[...,[2, 1]]
        return params[...,[1, 2]]

    def square_pts(self, params: torch.Tensor=None) -> Coords:
        """

        Args:
            params (ShapeParams, optional): The points for the square part. Defaults to None.

        Returns:
            ShapeParams: The square points
        """
        params = params if params is not None else self.coords()

        if self.increasing:
            return params[...,[1, 2]]
        return params[...,[0, 1]]
    
    def truncate(self, m: torch.Tensor) -> torch.Tensor:
        """

        Args:
            m (torch.Tensor): The membership

        Returns:
            torch.Tensor: Truncate the trapezoid
        """
        params = self._coords()
        if self.increasing:
            new_pt = params[...,0] * (1 - m)  + params[...,1] * m
        else:
            new_pt = (params[...,1] * m +  params[...,2] * (1 - m))
        return replace(params, new_pt, 1, to_unsqueeze=True)

    def areas(self, m: torch.Tensor, truncate: bool = False) -> torch.Tensor:
        """

        Args:
            m (torch.Tensor): The membership
            truncate (bool, optional): Whether to truncate or scale. Defaults to False.

        Returns:
            torch.Tensor: The areas of the trapezoid
        """
        if truncate:
            params = self.truncate(m)
        else:
            params = self._coords()
        
        assert isinstance(params, torch.Tensor), type(params)
        a = self.a(params)
        b = self.b(params)

        return trapezoid_area(a, b, m)
    
    def mean_cores(self, m: torch.Tensor, truncate: bool = False) -> torch.Tensor:
        """

        Args:
            m (torch.Tensor): The membership
            truncate (bool, optional): Whether to truncate or scale. Defaults to False.

        Returns:
            torch.Tensor: The mean cores of the trapezoid
        """
        if truncate:
            params = self.truncate(m)
        else:
            params = self._coords()

        pts = self.square_pts(params)
        return self._resize_to_m(
            trapezoid_mean_core(pts[...,0], pts[...,1]), m
        )

    def centroids(self, m: torch.Tensor, truncate: bool = False) -> torch.Tensor:
        """

        Args:
            m (torch.Tensor): The membership
            truncate (bool, optional): Whether to truncate or scale. Defaults to False.

        Returns:
            torch.Tensor: The centroids of the trapezoid
        """
        # x = (b+2a)/(3(a+b))h
        if truncate:
            params = self.truncate(m)
        else:
            params = self._coords()
        
        a = self.a(params)
        b = self.b(params)
        return self._resize_to_m(
            trapezoid_centroid(a, b, m), m
        )
