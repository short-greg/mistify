from ._base import Polygon, ShapeParams
from ...utils import unsqueeze
import torch
from ... import _functional as functional


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

    def join(self, x: torch.Tensor) -> torch.Tensor:
        """Join calculates the membership value for each section of trapezoid and uses the maximimum value
        as the value

        Args:
            x (torch.Tensor): The value to calculate the membership for

        Returns:
            torch.Tensor: The membership
        """
        params = self.params()
        x = unsqueeze(x)
        # m1 = calc_m_linear_increasing(x, params.pt(0), params.pt(1), self._m)
        # m2 = calc_m_flat(x, params.pt(1), params.pt(2), self._m)
        # m3 = calc_m_linear_decreasing(x, params.pt(2), params.pt(3), self._m)

        # return torch.max(torch.max(m1, m2), m3)

        return functional.shape.trapezoid(
            x, params.pt(0), params.pt(1), params.pt(2), params.pt(3)
        )

    def a(self, params: ShapeParams=None) -> torch.Tensor:
        """
        Args:
            params (ShapeParams, optional): the params for the trapezoid. Defaults to None.

        Returns:
            torch.Tensor: Calculate the top length
        """

        params = params or self.params()
        return (
            params.pt(3) - params.pt(0)
        )

    def b(self, params: ShapeParams=None) -> torch.Tensor:
        """
        Args:
            params (ShapeParams, optional): the params for the trapezoid. Defaults to None.

        Returns:
            torch.Tensor: The bottom length
        """
        params = params or self.params()
        return params.pt(2) - params.pt(1)

    def truncate(self, m: torch.Tensor) -> ShapeParams:
        """

        Args:
            m (torch.Tensor): The membership

        Returns:
            torch.Tensor: Truncate the trapezoid
        """
        params = self.params()
        new_pt1 = (params.pt(1) * (1 - m) + params.pt(0) * m)
        new_pt2 = params.pt(3) * m + params.pt(2) * (1 - m)
        params = params.replace(new_pt1, 1, to_unsqueeze=True)
        params = params.replace(new_pt2, 2, to_unsqueeze=True)
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
            params = self.params()
            
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
        params = self._params()
        return self._resize_to_m(
             trapezoid_mean_core(params.pt(1), params.pt(2)),
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
            params = self._params()
        
        a = self.a(params)
        b = self.b(params)
        return trapezoid_centroid(a, b, m)


class IsoscelesTrapezoid(Polygon):

    PT = 3

    def join(self, x: torch.Tensor) -> 'torch.Tensor':
        """Calculates the membership value for each part of the isosceles
        trapezoid and takes the maximum

        Args:
            x (torch.Tensor): The value to get the membership value for

        Returns:
            torch.Tensor: The membership value of x
        """
        x = unsqueeze(x)
        params = self.params()
        return functional.shape.isosceles_trapezoid(
            x, params.pt(0), params.pt(1), params.pt(2)
        )
    
    def a(self, params: ShapeParams=None) -> torch.Tensor:

        """
        Args:
            params (ShapeParams, optional): the params for the trapezoid. Defaults to None.

        Returns:
            torch.Tensor: Calculate the top length
        """
        params = params or self.params()

        dx = params.pt(1) - params.pt(0)
        return (
            params.pt(2) + dx - params.pt(0)
        )

    def b(self, params: ShapeParams=None):
        """
        Args:
            params (ShapeParams, optional): the params for the trapezoid. Defaults to None.

        Returns:
            torch.Tensor: Calculate the bottom length
        """
        params = params or self.params()
        return params.pt(2) - params.pt(1)

    def truncate(self, m: torch.Tensor) -> ShapeParams:
        """

        Args:
            m (torch.Tensor): The membership

        Returns:
            torch.Tensor: Truncate the trapezoid
        """
        params = self._params()
        new_pt1 = (params.pt(0) * (1 - m) + params.pt(1) * m)
        new_pt2 = 2 * params.pt(2) - new_pt1
        params = params.replace(new_pt1, 1, to_unsqueeze=True)
        params = params.replace(new_pt2, 2, to_unsqueeze=True)
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
            params = self._params()
            
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
        params = self._params()
        return self._resize_to_m(
            trapezoid_mean_core(params.pt(1), params.pt(2)), m
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
            params = self._params()
        
        a = self.a(params)
        b = self.b(params)
        return self._resize_to_m(
            trapezoid_centroid(a, b, m), m
        )

class RightTrapezoid(Polygon):

    PT = 3

    def __init__(self, params: ShapeParams, increasing: bool=True):
        super().__init__(params)
        self.increasing = increasing

    def join(self, x: torch.Tensor) -> 'torch.Tensor':
        """Join calculates the membership value for each section of trapezoid and uses the maximimum value
        as the value

        Args:
            x (torch.Tensor): The value to calculate the membership for

        Returns:
            torch.Tensor: The membership
        """
        params = self.params()
        return functional.shape.right_trapezoid(
            unsqueeze(x), params.pt(0), params.pt(1), params.pt(2), self.increasing
        )
    
    def a(self, params: ShapeParams=None) -> torch.Tensor:
        """
        Args:
            params (ShapeParams, optional): the params for the trapezoid. Defaults to None.

        Returns:
            torch.Tensor: Calculate the top length
        """
        params = params or self.params()
        return (
            params.pt(2) - params.pt(0)
        )

    def b(self, params: ShapeParams=None) -> torch.Tensor:
        """
        Args:
            params (ShapeParams, optional): the params for the trapezoid. Defaults to None.

        Returns:
            torch.Tensor: Calculate the bottom length
        """
        params = params or self.params()
        if self.increasing:

            return params.pt(2) - params.pt(1)
        return params.pt(1) - params.pt(0)
    
    def triangle_pts(self, params: ShapeParams=None, to_order: bool=False) -> ShapeParams:
        """

        Args:
            params (ShapeParams, optional): The points for the triangle part. Defaults to None.

        Returns:
            ShapeParams: The triangle points
        """
        params = params or self.params()

        if self.increasing:
            return params.sub([0, 1])
        if to_order:
            return params.sub([2, 1])
        return params.sub([1, 2])

    def square_pts(self, params: ShapeParams=None) -> ShapeParams:
        """

        Args:
            params (ShapeParams, optional): The points for the square part. Defaults to None.

        Returns:
            ShapeParams: The square points
        """
        params = params or self.params()

        if self.increasing:
            return params.sub([1, 2])
        return params.sub([0, 1])
    
    def truncate(self, m: torch.Tensor) -> ShapeParams:
        """

        Args:
            m (torch.Tensor): The membership

        Returns:
            torch.Tensor: Truncate the trapezoid
        """
        params = self._params()
        if self.increasing:
            new_pt = params.pt(0) * (1 - m)  + params.pt(1) * m
        else:
            new_pt = (params.pt(1) * m +  params.pt(2) * (1 - m))
        return params.replace(new_pt, 1, to_unsqueeze=True)

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
            params = self._params()
            
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
            params = self._params()

        pts = self.square_pts(params)
        return self._resize_to_m(
            trapezoid_mean_core(pts.pt(0), pts.pt(1)), m
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
            params = self._params()
        
        a = self.a(params)
        b = self.b(params)
        return self._resize_to_m(
            trapezoid_centroid(a, b, m), m
        )
