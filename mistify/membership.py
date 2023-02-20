from abc import abstractmethod, abstractproperty
import typing
import torch
from dataclasses import dataclass
from .fuzzy import intersect, positives

# TODO: 
# Analyze the classes and design an approach to make
# them easier to work with
# Change so that it uses the FuzzySet class

def resize_to(x1: torch.Tensor, x2: torch.Tensor, dim=0):

    if x1.size(dim) == 1 and x2.size(dim) != 1:
        size = [1] * x1.dim()
        size[dim] = x2.size(dim)
        return x1.repeat(*size)
    elif x1.size(dim) != x2.size(dim):
        raise ValueError()
    return x1


class ShapeParams:
    
    # batch, set, index
    param: torch.Tensor

    def __init__(self, x: torch.Tensor):
        if x.dim() == 3:
            x = x[None]
        assert x.dim() == 4
        self._x = x

    def sub(self, index: typing.Union[int, typing.Tuple[int, int]]):
        if isinstance(index, int):
            index = slice(index, index + 1)
        else:
            index = slice(*index)
        return ShapeParams(self._x[:, :, :, index])

    def pt(self, index: int):
        assert isinstance(index, int)
        return self._x[:,:,:,index]

    def sample(self, index: int):
        return self._x[index]

    def samples(self, indices):
        return self._x[indices]
        
    @property
    def x(self) -> torch.Tensor:
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
        
        print(mine[:,:,:,:idx].shape, x.shape, mine[:,:,:,idx:].shape)
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


def check_contains(x: torch.Tensor, pt1: torch.Tensor, pt2: torch.Tensor):
    
    return (x >= pt1) & (x <= pt2)


def calc_m_flat(x, pt1: torch.Tensor, pt2: torch.Tensor, m: torch.Tensor):

    return m * check_contains(x, pt1, pt2).float()


def calc_m_linear_increasing(x: torch.Tensor, pt1: torch.Tensor, pt2: torch.Tensor, m: torch.Tensor):
    return (x - pt1) * (m / (pt2 - pt1)) * check_contains(x, pt1, pt2).float() 


def calc_m_linear_decreasing(x: torch.Tensor, pt1: torch.Tensor, pt2: torch.Tensor, m: torch.Tensor):
    return ((x - pt1) * (-m / (pt2 - pt1)) + m) * check_contains(x, pt1, pt2).float()


def calc_x_linear_increasing(m0: torch.Tensor, pt1: torch.Tensor, pt2: torch.Tensor, m: torch.Tensor):
    # NOTE: To save on computational costs do not perform checks to see
    # if m0 is greater than m

    # TODO: use intersect function
    m0 = torch.min(m0, m)
    # m0 = m0.intersect(m)
    x = m0 * (pt2 - pt1) / m + pt1
    torch.nan_to_num_(x, 0.0, 0.0)
    return x


def calc_x_linear_decreasing(m0: torch.Tensor, pt1, pt2, m: torch.Tensor):

    # m0 = m0.intersect(m)
    m0 = torch.min(m0, m)
    x = -(m0 - 1) * (pt2 - pt1) / m + pt1
    torch.nan_to_num_(x, 0.0, 0.0)
    return x


def calc_m_logistic(x, b, s, m: torch.Tensor):

    z = s * (x - b)
    multiplier = 4 * m
    y = torch.sigmoid(z)
    return multiplier * y * (1 - y)


def calc_x_logistic(y, b, s):

    return -torch.log(1 / y - 1) / s + b


def calc_dx_logistic(m0: torch.Tensor, s: torch.Tensor, m_base: torch.Tensor):
    
    m = m0 / m_base
    dx = -torch.log((-m - 2 * torch.sqrt(1 - m) + 2) / (m)).float()
    dx = dx / s
    return dx


def calc_area_logistic(s: torch.Tensor, m_base: torch.Tensor, left=True):
    
    return 4 * m_base / s


def calc_area_logistic_one_side(x: torch.Tensor, b: torch.Tensor, s: torch.Tensor, m_base: torch.Tensor):
    
    z = s * (x - b)
    left = (z < 0).float()
    a = torch.sigmoid(z)
    # only calculate area of one side so 
    # flip the probability
    a = left * a + (1 - left) * (1 - a)

    return a * m_base * 4 / s

def unsqueeze(x: torch.Tensor):
    return x.unsqueeze(x.dim())


class Shape(object):

    def __init__(self, n_variables: int, n_terms: int):

        super().__init__()
        self._areas = None
        self._mean_cores = None
        self._centroids = None
        self._n_variables = n_variables
        self._n_terms = n_terms

    @property
    def n_terms(self):
        return self._n_terms
    
    @property
    def n_variables(self):
        return self._n_variables

    def join(self, x: torch.Tensor):
        pass

    @abstractmethod
    def _calc_areas(self):
        pass

    @property
    def areas(self) -> torch.Tensor:
        if self._areas is None:
            self._areas = self._calc_areas()
        return self._areas

    @abstractmethod
    def _calc_mean_cores(self):
        pass

    @property
    def mean_cores(self) -> torch.Tensor:
        if self._mean_cores is None:
            self._mean_cores = self._calc_mean_cores()
        return self._mean_cores

    @abstractmethod
    def _calc_centroids(self) -> torch.Tensor:
        pass

    @abstractproperty
    def m(self):
        pass

    @property
    def centroids(self) -> torch.Tensor:
        if self._centroids is None:
            self._centroids = self._calc_centroids()
        return self._centroids
    
    @abstractmethod
    def scale(self, m: torch.Tensor) -> 'Shape':
        pass

    @abstractmethod
    def truncate(self, m: torch.Tensor) -> 'Shape':
        pass

    @abstractmethod
    def join(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def _resize_to_m(self, x: torch.Tensor, m: torch.Tensor):
        if x.size(0) == 1 and m.size(0) != 1:
            return x.repeat(m.size(0), *[1] * (m.dim() - 1))
        return x    


class Polygon(Shape):

    PT = None

    def __init__(self, params: ShapeParams, m: typing.Optional[torch.Tensor]=None):

        assert params.x.size(3) == self.PT, f'Number of points must be {self.PT} not {params.x.size(3)}'
        self._params = params
        # Change to fuzzy.positives
        self._m = m if m is not None else torch.ones(
            self._params.batch_size, self._params.set_size, 
            self._params.n_terms, device=params.x.device,        
        )

        super().__init__(self._params.set_size, self._params.n_terms)


class IncreasingRightTriangle(Polygon):

    PT = 2

    def join(self, x: torch.Tensor) -> torch.Tensor:
        return calc_m_linear_increasing(
            unsqueeze(x), self._params.pt(0), self._params.pt(1), self._m
        )

    def _calc_areas(self):
        
        return self._resize_to_m(
            0.5 * (self._params.pt(1)
            - self._params.pt(0)) * self._m, self._m
        )

    def _calc_mean_cores(self):
        return self._resize_to_m(self._params.pt(1), self._m)

    def _calc_centroids(self):
        
        p1, p2 = 1 / 3, 2 / 3

        return self._resize_to_m(
            p1 * self._params.pt(0) + p2 * self._params.pt(1), self._m
        )
    
    def scale(self, m: torch.Tensor) -> 'IncreasingRightTriangle':

        print(type(m), type(self._m))
        updated_m = intersect(m, self._m)
        print(updated_m)
        
        return IncreasingRightTriangle(
            self._params, updated_m
        )

    def truncate(self, m: torch.Tensor) -> 'IncreasingRightTrapezoid':
        # TODO: FINISH
        updated_m = intersect(self._m, m)

        pt = calc_x_linear_increasing(
            updated_m, self._params.pt(0), self._params.pt(1), self._m
        )
        params = self._params.insert(
            pt, 1, to_unsqueeze=True, equalize_to=updated_m
        )
        return IncreasingRightTrapezoid(
            params, updated_m
        )


class DecreasingRightTriangle(Polygon):

    PT = 2
    
    def join(self, x: torch.Tensor):
    
        return calc_m_linear_decreasing(
            unsqueeze(x), self._params.pt(0), self._params.pt(1), self._m
        )

    def _calc_areas(self):
        
        return self._resize_to_m((
            0.5 * (self._params.pt(1)
            - self._params.pt(0)) * self._m
        ), self._m)

    def _calc_mean_cores(self):
        
        return self._resize_to_m(self._params.pt(0), self._m)

    def _calc_centroids(self):
        return self._resize_to_m(
            2 / 3 * self._params.pt(0) 
            + 1 / 3 * self._params.pt(1), self._m
        )
    
    def scale(self, m: torch.Tensor):
        updated_m = intersect(self._m, m)
        
        return DecreasingRightTriangle(
            self._params, updated_m
        )

    def truncate(self, m: torch.Tensor):
        updated_m = intersect(self._m, m)

        pt = calc_x_linear_decreasing(
            updated_m, self._params.pt(0), self._params.pt(1), self._m
        )

        # print('Truncate Right Triangle: ', pt.size(), updated_m.data.size(), self._params.x.size())
        params = self._params.insert(pt, 1, to_unsqueeze=True, equalize_to=updated_m)
        return DecreasingRightTrapezoid(
            params, updated_m
        )


class Square(Polygon):

    PT = 2

    def join(self, x: torch.Tensor):
        return (
            (x[:,:,None] >= self._params.pt(0)) 
            & (x[:,:,None] <= self._params.pt(1))
        ).type_as(x) * self._m

    def _calc_areas(self):
        
        return self._resize_to_m((
            (self._params.pt(1) 
            - self._params.pt(0)) * self._m
        ), self._m)

    def _calc_mean_cores(self):
        return self._resize_to_m(1 / 2 * (
            self._params.pt(0) + self._params.pt(1)
        ), self._m)

    def _calc_centroids(self):
        return self._resize_to_m(1 / 2 * (
            self._params.pt(0) + self._params.pt(1)
        ), self._m)
    
    def scale(self, m: torch.Tensor) -> 'Square':
        updated_m = intersect(m, self._m)
        
        return Square(
            self._params, updated_m
        )

    def truncate(self, m: torch.Tensor) -> 'Square':
        updated_m = intersect(m, self._m)

        return Square(
            self._params, updated_m
        )


class Triangle(Polygon):

    PT = 3

    def join(self, x: torch.Tensor):
        
        m1 = calc_m_linear_increasing(
            unsqueeze(x), self._params.pt(0), self._params.pt(1), self._m
        )
        m2 = calc_m_linear_decreasing(
            unsqueeze(x), self._params.pt(1), self._params.pt(2), self._m
        )
        return intersect(m1, m2)

    def _calc_areas(self):
        
        return self._resize_to_m((
            0.5 * (self._params.pt(2) 
            - self._params.pt(0)) * self._m
        ), self._m)

    def _calc_mean_cores(self):
        return self._resize_to_m(self._params.pt(1), self._m)

    def _calc_centroids(self):
        return self._resize_to_m(1 / 3 * (
            self._params.pt(0) + self._params.pt(1) + self._params.pt(2)
        ), self._m)
    
    def scale(self, m: torch.Tensor) -> 'Triangle':

        updated_m = intersect(self._m, m)        
        return Triangle(
            self._params, updated_m
        )

    def truncate(self, m: torch.Tensor) -> 'Trapezoid':
        updated_m = intersect(self._m, m)

        pt1 = calc_x_linear_increasing(updated_m, self._params.pt(0), self._params.pt(1), self._m)
        pt2 = calc_x_linear_decreasing(updated_m, self._params.pt(1), self._params.pt(2), self._m)
        to_replace = torch.cat(
            [pt1.unsqueeze(3), pt2.unsqueeze(3)], dim=3
        )
        params= self._params.replace(
            to_replace, 1, False, equalize_to=updated_m
        )

        return Trapezoid(
            params, updated_m
        )


class Trapezoid(Polygon):

    PT = 4

    def join(self, x: torch.Tensor) -> torch.Tensor:

        x = unsqueeze(x)
        m1 = calc_m_linear_increasing(x, self._params.pt(0), self._params.pt(1), self._m)
        m2 = calc_m_flat(x, self._params.pt(1), self._params.pt(2), self._m)
        m3 = calc_m_linear_decreasing(x, self._params.pt(2), self._params.pt(3), self._m)

        return torch.max(torch.max(
            m1, m2
        ), m3)

    def _calc_areas(self):
        
        return self._resize_to_m((
            0.5 * (self._params.pt(2) 
            - self._params.pt(0)) * self._m
        ), self._m)

    def _calc_mean_cores(self):
        return self._resize_to_m(
            0.5 * (self._params.pt(1) + self._params.pt(2)), self._m
        )

    def _calc_centroids(self):
        d1 = 0.5 * (self._params.pt(1) - self._params.pt(0))
        d2 = self._params.pt(2) - self._params.pt(1)
        d3 = 0.5 * (self._params.pt(3) - self._params.pt(2))

        return self._resize_to_m((
            d1 * (2 / 3 * self._params.pt(1) + 1 / 3 * self._params.pt(0)) +
            d2 * (1 / 2 * self._params.pt(2) + 1 / 2 *  self._params.pt(1)) + 
            d3 * (1 / 3 * self._params.pt(3) + 2 / 3 * self._params.pt(2))
        ) / (d1 + d2 + d3), self._m)

    def scale(self, m: torch.Tensor) -> 'Trapezoid':
        updated_m = intersect(self._m, m)
        return Trapezoid(
            self._params, updated_m
        )

    def truncate(self, m: torch.Tensor) -> 'Trapezoid':
        updated_m = intersect(self._m, m)

        # m = ShapeParams(m, True, m.dim() == 3)
        left_x = calc_x_linear_increasing(
            updated_m, self._params.pt(0), self._params.pt(1), self._m
        )

        right_x = calc_x_linear_decreasing(
            updated_m, self._params.pt(2), self._params.pt(3), self._m
        )

        params = self._params.replace(left_x, 1, to_unsqueeze=True, equalize_to=updated_m)
        params = params.replace(right_x, 2, to_unsqueeze=True)

        return Trapezoid(
            params, updated_m, 
        )


class Logistic(Shape):

    def __init__(
        self, biases: ShapeParams, scales: ShapeParams, m: torch.Tensor=None
    ):
        self._biases = biases
        self._scales = scales

        self._m = m if m is not None else positives(
            self._biases.batch_size, self._biases.set_size, 
            self._biases.n_terms, device=biases.x.device
        )

        super().__init__(
            self._biases.n_variables,
            self._biases.n_terms
        )

    @property
    def biases(self):
        return self._biases
    
    @property
    def scales(self):
        return self._scales
    
    @classmethod
    def from_combined(cls, params: ShapeParams, m: torch.Tensor=None):

        #if params.x.dim() == 4:
        return cls(
            ShapeParams(params.sub((0, 1))), 
            ShapeParams(params.sub((1, 2))), m
        )
        # return cls(params[:,:,0], params[:,:,1], m)


class LogisticBell(Logistic):

    def join(self, x: torch.Tensor) -> torch.Tensor:
        z = self._scales.pt(0) * (unsqueeze(x) - self._biases.pt(0))
        sig = torch.sigmoid(z)
        # not 4 / s
        return 4  * (1 - sig) * sig * self._m

    def _calc_areas(self):
        return self._resize_to_m(4 * self._m / self._biases.pt(0), self._m)
        
    def _calc_mean_cores(self):
        return self._resize_to_m(self._biases.pt(0), self._m)

    def _calc_centroids(self):
        return self._resize_to_m(self._biases.pt(0), self._m)

    def scale(self, m: torch.Tensor) -> 'LogisticBell':
        updated_m = intersect(self._m, m)
        return LogisticBell(
            self._biases, self._scales, updated_m
        )

    def truncate(self, m: torch.Tensor) -> 'LogisticTrapezoid':

        return LogisticTrapezoid(
            self._biases, self._scales,  m, self._m 
        )


class LogisticTrapezoid(Logistic):
    
    def __init__(
        self, biases: ShapeParams, scales: ShapeParams, 
        truncated_m: torch.Tensor=None, scaled_m: torch.Tensor=None
    ):
        super().__init__(biases, scales, scaled_m)

        if truncated_m is None:
            truncated_m = positives(*self._m.size(), device=self._m.device)

        self._truncated_m = intersect(truncated_m, self._m)
        
        dx = unsqueeze(calc_dx_logistic(self._truncated_m, self._scales.pt(0), self._m))
        self._dx = ShapeParams(dx)
        self._pts = ShapeParams(torch.concat([
            self._biases.x - self._dx.x,
            self._biases.x + self._dx.x
        ], dim=dx.dim() - 1))

    @property
    def dx(self):
        return self._dx
    
    @property
    def m(self) -> torch.Tensor:
        return self._truncated_m

    def join(self, x: torch.Tensor) -> 'torch.Tensor':
        x = unsqueeze(x)
        inside = check_contains(x, self._pts.pt(0), self._pts.pt(1)).float()
        m1 = calc_m_logistic(x, self._biases.pt(0), self._scales.pt(0), self._m) * (1 - inside)
        m2 = self._truncated_m * inside
        return torch.max(m1, m2)

    def _calc_areas(self):
        # symmetrical so multiply by 2
        return self._resize_to_m(2 * calc_area_logistic_one_side(
            self._pts.pt(0), self._biases.pt(0), self._scales.pt(0), self._m
        ), self._m)
        
    def _calc_mean_cores(self):
        return self._resize_to_m(self._biases.pt(0), self._m)

    def _calc_centroids(self):
        return self._resize_to_m(self._biases.pt(0), self._m)

    def scale(self, m: torch.Tensor) -> 'LogisticTrapezoid':
        updated_m = intersect(self._m, m)
        # TODO: check if multiplication is correct
        truncated_m = self._truncated_m * updated_m

        return LogisticTrapezoid(
            self._biases, self._scales, truncated_m, updated_m
        )

    def truncate(self, m: torch.Tensor) -> 'LogisticTrapezoid':
        truncated_m = intersect(self._truncated_m, m)
        return LogisticTrapezoid(
            self._biases, self._scales, truncated_m, self._m
        )


class RightLogistic(Logistic):
    
    def __init__(
        self, biases: ShapeParams, scales: ShapeParams, is_right: bool=True,
        m: torch.Tensor= None
    ):
        super().__init__(biases, scales, m)
        self._is_right = is_right
        self._direction = is_right * 2 - 1
    
    def _on_side(self, x: torch.Tensor):
        if self._is_right:
            side = x >= self._biases.pt(0)
        else: side = x <= self._biases.pt(0)
        return side

    def join(self, x: torch.Tensor):
        x = unsqueeze(x)
        return calc_m_logistic(
            x, self._biases.pt(0), 
            self._scales.pt(0), self._m
        ) * self._on_side(x).float()

    def _calc_areas(self):
        return self._resize_to_m(2 * self._m / self._biases.pt(0), self._m)

    def _calc_mean_cores(self):
        return self._resize_to_m(self._biases.pt(0), self._m)

    def _calc_centroids(self):
        base_y = 0.75 if self._is_right else 0.25
        x = torch.logit(torch.tensor(base_y, dtype=torch.float, device=self._m.device)) / self._scales.pt(0) + self._biases.pt(0)
        return self._resize_to_m(x, self._m)

    def scale(self, m: torch.Tensor) -> 'RightLogistic':
        updated_m = intersect(self._m, m)
        
        return RightLogistic(
            self._biases, self._scales, self._is_right, updated_m
        )

    def truncate(self, m: torch.Tensor) -> 'LogisticTrapezoid':
        truncated_m = intersect(self._m, m)
        return RightLogisticTrapezoid(
            self._biases, self._scales, self._is_right, truncated_m, self._m
        )

    @classmethod
    def from_combined(cls, params: torch.Tensor, is_right: bool=True,m: torch.Tensor=None):
        # TODO: Check this and confirm
        if params.dim() == 4:
            return cls(params.sub(0), params.sub(1), is_right, m)
        return cls(params.sub(0), params.sub(1), is_right, m)


class RightLogisticTrapezoid(Logistic):

    def __init__(
        self, biases: ShapeParams, scales: ShapeParams, is_right: bool, 
        truncated_m: torch.Tensor=None, scaled_m: torch.Tensor=None
        
    ):
        super().__init__(biases, scales, scaled_m)

        if truncated_m is None:
            truncated_m = positives(self._m.size(), device=self._m.device)

        self._truncated_m = intersect(self._m, truncated_m)
        dx = unsqueeze(calc_dx_logistic(self._truncated_m, self._scales.pt(0), self._m))
        self._dx = ShapeParams(dx)
        self._is_right = is_right
        self._direction = is_right * 2 - 1
        self._pts = ShapeParams(self._biases.x + self._direction * dx)

    @property
    def dx(self):
        return self._dx

    @property
    def m(self):
        return self._truncated_m

    def _contains(self, x: torch.Tensor):
        if self._is_right:
            square_contains = (x >= self._biases.pt(0)) & (x <= self._pts.pt(0))
            logistic_contains = x >= self._pts.pt(0)
        else:
            square_contains = (x <= self._biases.pt(0)) & (x >= self._pts[0])
            logistic_contains = x <= self._pts.pt(0)
        return square_contains.float(), logistic_contains.float()

    def join(self, x: torch.Tensor) -> 'torch.Tensor':
        x = unsqueeze(x)
        
        square_contains, logistic_contains = self._contains(x)
        
        m1 = calc_m_logistic(
            x, self._biases.pt(0), self._scales.pt(0), self._m
        ) * logistic_contains
        m2 = self._m * square_contains
        return torch.max(m1, m2)

    def _calc_areas(self):
        a1 = self._resize_to_m(calc_area_logistic_one_side(
            self._pts.pt(0), self._biases.pt(0), self._scales.pt(0), 
            self._m), self._m)
        a2 = 0.5 * (self._biases.pt(0) + self._pts.pt(0)) * self._m
        return self._resize_to_m(a1 + a2, self._m)

    def _calc_mean_cores(self):
        return self._resize_to_m(0.5 * (self._biases.pt(0) + self._pts.pt(0)), self._m) 

    def _calc_centroids(self):

        # area up to "dx"
        # print('Centroids: ', self._scales.x.size(), self._dx.x.size())
        p = torch.sigmoid(self._scales.pt(0) * (-self._dx.pt(0)))
        centroid_logistic = self._biases.pt(0) + torch.logit(p / 2) / self._scales.pt(0)
        centroid_square = self._biases.pt(0) - self._dx.pt(0) / 2

        centroid = (centroid_logistic * p + centroid_square * self._dx.pt(0)) / (p + self._dx.pt(0))
        if self._is_right:
            return self._biases.pt(0) + self._biases.pt(0) - centroid
        return self._resize_to_m(centroid, self._m)

    def scale(self, m: torch.Tensor) -> 'RightLogisticTrapezoid':

        updated_m = intersect(self._m, m)

        # TODO: Confirm if this is correct
        # I think it should be intersecting rather than multiplying
        truncated_m = self._truncated_m * updated_m

        return RightLogisticTrapezoid(
            self._biases, self._scales, self._is_right, truncated_m, updated_m
        )

    def truncate(self, m: torch.Tensor) -> 'RightLogisticTrapezoid':

        truncated_m = intersect(self._truncated_m, m)
        return RightLogisticTrapezoid(
            self._biases, self._scales, self._is_right, truncated_m, self._m
        )

    @classmethod
    def from_combined(cls, params: torch.Tensor, is_right: bool=True,m: torch.Tensor=None):

        if params.dim() == 4:

            return cls(params.sub(0), params.sub(1), is_right, m)
        return cls(params.sub(0), params.sub(1), is_right, m)


class IsoscelesTriangle(Polygon):

    PT = 2

    def join(self, x: torch.Tensor) -> torch.Tensor:

        left_m = calc_m_linear_increasing(
            unsqueeze(x), self._params.pt(0), self._params.pt(1), self._m
        )
        right_m = calc_m_linear_decreasing(
            unsqueeze(x), self._params.pt(1), 
            self._params.pt(1) + (self._params.pt(1) - self._params.pt(0)), 
            self._m
        )
        return torch.max(left_m, right_m)

    def _calc_areas(self):
        
        return self._resize_to_m(
            0.5 * (self._params.pt(0)
            - self._params.pt(1)) * self._m, self._m
        )

    def _calc_mean_cores(self):
        return self._resize_to_m(self._params.pt(1), self._m)

    def _calc_centroids(self):
        return self._resize_to_m(self._params.pt(1), self._m)

    def scale(self, m: torch.Tensor) -> 'IsoscelesTriangle':
        updated_m = intersect(self._m, m)
        return IsoscelesTriangle(
            self._params, updated_m
        )

    def truncate(self, m: torch.Tensor) -> 'IsoscelesTrapezoid':
        
        updated_m = intersect(self._m, m)
        pt1 = calc_x_linear_increasing(updated_m, self._params.pt(0), self._params.pt(1), self._m)
        pt2 = calc_x_linear_decreasing(
            updated_m, self._params.pt(1), self._params.pt(1) + self._params.pt(1) - self._params.pt(0), self._m)

        to_replace = torch.cat(
            [pt1.unsqueeze(3), pt2.unsqueeze(3)], dim=3
        )

        params = self._params.replace(
            to_replace, 1, False, updated_m
        )
        return IsoscelesTrapezoid(
            params, updated_m
        )


class IsoscelesTrapezoid(Polygon):

    PT = 3

    def join(self, x: torch.Tensor) -> 'torch.Tensor':

        x = unsqueeze(x)
        left_m = calc_m_linear_increasing(
            x, self._params.pt(0), self._params.pt(1), self._m
        )
        middle = calc_m_flat(x, self._params.pt(1), self._params.pt(2), self._m)
        pt3 = self._params.pt(1) - self._params.pt(0) + self._params.pt(2)
        right_m = calc_m_linear_decreasing(
            x, self._params.pt(2), pt3, self._m
        )
        return torch.max(torch.max(left_m, middle), right_m)
    
    @property
    def a(self):
        return (
            self._params.pt(2) - self._params.pt(0) + 
            self._params.pt(1) - self._params.pt(0)
        )

    @property
    def b(self):
        return self._params.pt(2) - self._params.pt(1)

    def _calc_areas(self):
        
        return self._resize_to_m(
            0.5 * (self.a + self.b) * self._m, self._m
        )

    def _calc_mean_cores(self):
        return self._resize_to_m(0.5 * (self._params.pt(2) + self._params.pt(1)), self._m)

    def _calc_centroids(self):
        return self.mean_cores

    def scale(self, m: torch.Tensor) -> 'IsoscelesTrapezoid':
        updated_m = intersect(self._m, m)
        return IsoscelesTrapezoid(self._params, updated_m)

    def truncate(self, m: torch.Tensor) -> 'IsoscelesTrapezoid':
        updated_m = intersect(self._m, m)

        left_x = calc_x_linear_increasing(
            updated_m, self._params.pt(0), self._params.pt(1), self._m
        )

        right_x = self._params.pt(2) + self._params.pt(1) - left_x

        params = self._params.replace(
            left_x, 1, True, updated_m
        )
        params = params.replace(
            right_x, 2, True
        )
        return IsoscelesTrapezoid(params, updated_m)


class IncreasingRightTrapezoid(Polygon):

    PT = 3

    def join(self, x: torch.Tensor) -> 'torch.Tensor':
        m = calc_m_linear_increasing(
            unsqueeze(x), self._params.pt(0), self._params.pt(1), self._m
        )
        m2 = calc_m_flat(unsqueeze(x), self._params.pt(1), self._params.pt(2), self._m)

        return torch.max(m, m2)
    
    @property
    def a(self):
        return (
            self._params.pt(2) - self._params.pt(0)
        )

    @property
    def b(self):
        return self._params.pt(2) - self._params.pt(1)

    def _calc_areas(self):
        
        return self._resize_to_m(
            0.5 * (self.a + self.b) * self._m, self._m
        )

    def _calc_mean_cores(self):
        return self._resize_to_m(
            0.5 * (self._params.pt(2) + self._params.pt(1)), self._m
        )

    def _calc_centroids(self):
        
        d1 = 0.5 * (self._params.pt(1) - self._params.pt(0))
        d2 = self._params.pt(2) - self._params.pt(1)

        return self._resize_to_m((
            d1 * (2 / 3 * self._params.pt(1) + 1 / 3 * self._params.pt(0)) +
            d2 * (1 / 2 * self._params.pt(2) + 1 / 2 * self._params.pt(1))
        ) / (d1 + d2), self._m)

    def scale(self, m: torch.Tensor) -> 'IncreasingRightTrapezoid':
        return IncreasingRightTrapezoid(self._params, intersect(m, self._m))

    def truncate(self, m: torch.Tensor) -> 'IncreasingRightTrapezoid':
        updated_m = intersect(m, self._m)
        
        x = calc_x_linear_increasing(
            updated_m, self._params.pt(0), self._params.pt(1), self._m
        )
        params = self._params.replace(x, 1, True, updated_m)
        return IncreasingRightTrapezoid(params, updated_m)


class DecreasingRightTrapezoid(Polygon):

    PT = 3

    def join(self, x: torch.Tensor) -> 'torch.Tensor':

        m = calc_m_linear_decreasing(
            unsqueeze(x), self._params.pt(0), self._params.pt(1), self._m
        )
        m2 = calc_m_flat(unsqueeze(x), self._params.pt(1), self._params.pt(2), self._m)

        return torch.max(m, m2)
    
    @property
    def a(self):
        return (
            self._params.pt(2) - self._params.pt(0)
        )

    @property
    def b(self):
        return self._params.pt(1) - self._params.pt(0)

    def _calc_areas(self):
        
        return self._resize_to_m((
            0.5 * (self.a + self.b) * self._m
        ), self._m)

    def _calc_mean_cores(self):
        return self._resize_to_m(
            0.5 * (self._params.pt(0) + self._params.pt(1)), self._m
        )

    def _calc_centroids(self):
        d1 = self._params.pt(1) - self._params.pt(0)
        d2 = 0.5 * (self._params.pt(2) - self._params.pt(1))
        
        return self._resize_to_m((
            d1 * (1 / 2 * self._params.pt(1) + 1 / 2 * self._params.pt(0)) +
            d2 * (1 / 3 * self._params.pt(2) + 2 / 3 * self._params.pt(1))
        ) / (d1 + d2), self._m)

    def scale(self, m: torch.Tensor) -> 'DecreasingRightTrapezoid':
        return DecreasingRightTrapezoid(self._params, intersect(m, self._m))

    def truncate(self, m: torch.Tensor) -> 'DecreasingRightTrapezoid':
        updated_m = intersect(m, self._m)
        
        x = calc_x_linear_decreasing(
            updated_m, self._params.pt(0), self._params.pt(1), self._m
        )
        params = self._params.replace(x, 1, True, updated_m)
        return DecreasingRightTrapezoid(params, updated_m)
