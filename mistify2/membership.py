from abc import abstractmethod, abstractproperty
import typing
import torch
from dataclasses import dataclass
from .fuzzy import FuzzySet

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

    def sub(self, index: typing.Union[int, slice]):
        if isinstance(index, int):
            index = slice(index, index + 1)
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

    def insert(self, x: torch.Tensor, idx: int, to_unsqueeze: bool=False):
        x = x if not to_unsqueeze else unsqueeze(x)
        mine = resize_to(self.x, x)
        if not (0 <= idx <= mine.size(3)):
            raise ValueError(f'Argument idx must be in range of [0, {mine.size(3)}] not {idx}')
        
        return ShapeParams(
            torch.concat([mine[:,:,:,:idx], x, mine[:,:,:,idx:]], dim=3)
        )

    def replace(self, x: torch.Tensor, idx: int, to_unsqueeze: bool=False):
        x = x if not to_unsqueeze else unsqueeze(x)
        mine = resize_to(self.x, x)
        if not (0 <= idx < self._x.size(3)):
            raise ValueError(f'Argument idx must be in range of [0, {mine.size(3)}) not {idx}')
        return ShapeParams(
            torch.concat([mine[:,:,:,:idx], x, mine[:,:,:,idx+1:]], dim=3)
        )

    def replace_slice(self, x: torch.Tensor, pt_range: typing.Tuple[int, int], to_unsqueeze: bool=False):
        x = x if not to_unsqueeze else unsqueeze(x)
        
        return ShapeParams(
            torch.concat([self._x[:,:,:,:pt_range[0]], x, self._x[:,:,:,pt_range[1]+1:]], dim=3)
        )

    @classmethod
    def from_sub(cls, *sub: 'ShapeParams'):
        
        return ShapeParams(
            torch.cat([sub_i._x for sub_i in sub], dim=3)
        )


def check_contains(x: torch.Tensor, pt1: torch.Tensor, pt2: torch.Tensor):
    
    return (x >= pt1) & (x <= pt2)


def calc_m_flat(x, pt1: torch.Tensor, pt2: torch.Tensor, m: FuzzySet):

    return m.data * check_contains(x, pt1, pt2).float()


def calc_m_linear_increasing(x: torch.Tensor, pt1: torch.Tensor, pt2: torch.Tensor, m: FuzzySet):
    return (x - pt1) * (m.data / (pt2 - pt1)) * check_contains(x, pt1, pt2).float() 


def calc_m_linear_decreasing(x: torch.Tensor, pt1: torch.Tensor, pt2: torch.Tensor, m: FuzzySet):
    return ((x - pt1) * (-m.data / (pt2 - pt1)) + m.data) * check_contains(x, pt1, pt2).float()


def calc_x_linear_increasing(m0: FuzzySet, pt1: torch.Tensor, pt2: torch.Tensor, m: FuzzySet):
    # NOTE: To save on computational costs do not perform checks to see
    # if m0 is greater than m

    m0 = m0.intersect(m)
    x = m0.data * (pt2 - pt1) / m.data + pt1
    torch.nan_to_num_(x, 0.0, 0.0)
    return x


def calc_x_linear_decreasing(m0: FuzzySet, pt1, pt2, m: FuzzySet):

    m0 = m0.intersect(m)
    x = -(m0.data - 1) * (pt2 - pt1) / m.data + pt1
    torch.nan_to_num_(x, 0.0, 0.0)
    return x


def calc_m_logistic(x, b, s, m: FuzzySet):

    z = s * (x - b)
    multiplier = 4 * m.data
    y = torch.sigmoid(z)
    return multiplier * y * (1 - y)


def calc_dx_logistic(m0: FuzzySet, s: torch.Tensor, m_base: FuzzySet):
    
    m = m0.data / m_base.data
    dx = -torch.log((-m.data - 2 * torch.sqrt(1 - m) + 2) / (m.data)).float()
    dx = dx / s
    return dx


def calc_area_logistic(s: torch.Tensor, m_base: FuzzySet, left=True):
    
    return 4 * m_base / s


def calc_area_logistic_one_side(x: torch.Tensor, b: torch.Tensor, s: torch.Tensor, m_base: FuzzySet):
    
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
    def scale(self, m: FuzzySet) -> 'Shape':
        pass

    @abstractmethod
    def truncate(self, m: FuzzySet) -> 'Shape':
        pass

    @abstractmethod
    def join(self, x: torch.Tensor) -> FuzzySet:
        pass

    def _resize_to_m(self, x: torch.Tensor, m: FuzzySet):
        if m.is_batch and x.size(0) == 1 and m.data.size(0) != 1:
            return x.repeat(m.data.size(0), *[1] * (m.data.dim() - 1))
        return x    


class ConvexPolygon(Shape):

    PT = None

    def __init__(self, params: ShapeParams, m: typing.Optional[torch.Tensor]=None):

        assert params.x.size(3) == self.PT

        self._params = params
        self._m = m or FuzzySet.positives(
            self._params.batch_size, self._params.set_size, 
            self._params.n_terms, device=params.x.device, is_batch=True
        )

        super().__init__(self._params.set_size, self._params.n_terms)


class IncreasingRightTriangle(ConvexPolygon):

    PT = 2

    def join(self, x: torch.Tensor) -> FuzzySet:
        return FuzzySet(calc_m_linear_increasing(
            unsqueeze(x), self._params.pt(0), self._params.pt(1), self._m.data
        ), x.dim() == 2)

    def _calc_areas(self):
        
        return self._resize_to_m(
            0.5 * (self._params.pt(1)
            - self._params.pt(0)) * self._m.data, self._m
        )

    def _calc_mean_cores(self):
        return self._resize_to_m(self._params.pt(1), self._m)

    def _calc_centroids(self):
        
        p1, p2 = 1 / 3, 2 / 3

        return self._resize_to_m(
            p1 * self._params.pt(0) + p2 * self._params.pt(1), self._m
        )
    
    def scale(self, m: FuzzySet):

        updated_m = m.intersect(self._m)
        
        return IncreasingRightTriangle(
            self._params, updated_m
        )

    def truncate(self, m: FuzzySet):
        # TODO: FINISH
        updated_m = m.intersect(self._m)

        pt = calc_x_linear_increasing(
            updated_m, self._params.pt(0), self._params.pt(1), self._m
        )
        
        params = self._params.insert(pt, 1, to_unsqueeze=True)
        return IncreasingRightTrapezoid(
            params, updated_m
        )


# class DecreasingRightTriangle(ConvexPolygon):
    
#     def join(self, x: torch.Tensor):
        
#         return calc_m_linear_decreasing(
#             un_x(x), self._params[0], self._params[1], self._m[0]
#         )

#     def _calc_areas(self):
        
#         return (
#             0.5 * (self._params[1]
#             - self._params[0]) * self._m[0]
#         )

#     def _calc_mean_cores(self):
        
#         return self._params[0]

#     def _calc_centroids(self):
#         return 2 / 3 * self._params[0] + 1 / 3 * self._params[1]
    
#     def scale(self, m: FuzzySet):
#         updated_m = self._intersect_m(m)
        
#         return DecreasingRightTriangle(
#             self._params, updated_m
#         )

#     def truncate(self, m: FuzzySet):
        
#         # TODO: FINISH
#         updated_m = self._intersect_m(m)
#         pt = ShapeParams(
#             calc_x_linear_decreasing(
#                 updated_m, self._params.pt(0), self._params.pt(1), self._m
#             )
#         )

#         params = self._params.insert(pt, 1)

#         return DecreasingRightTrapezoid(
#             params.x, updated_m.x
#         )

# # need to simplify this

# # self._points
# # self._scales <- logistic
# # etc

# # 

# class Square(ConvexPolygon):

#     def join(self, x: torch.Tensor):
#         return (x >= self._params[0] & x <= self._params[1]).type_as(x) * self._m

#     def _calc_areas(self):
        
#         return (
#             (self._params[2] 
#             - self._params[0]) * self._m[0]
#         )

#     def _calc_mean_cores(self):
#         return 1 / 2 * (
#             self._params.pt(0) + self._params.pt(1)
#         )

#     def _calc_centroids(self):
#         return 1 / 2 * (
#             self._params.pt(0) + self._params.pt(1)
#         )
    
#     def scale(self, m: FuzzySet):

#         updated_m = m.intersect(self._m)
        
#         return Square(
#             self._params, updated_m
#         )

#     def truncate(self, m: FuzzySet):
#         # TODO: FINISH
        
#         updated_m = m.intersect(self._m)

#         return Square(
#             self._params, updated_m
#         )


# class Triangle(ConvexPolygon):

#     def join(self, x: torch.Tensor):

#         m = self._m[0][0]
        
#         m1 = calc_m_linear_increasing(
#             un_x(x), self._params[0][0], self._params[1][0], m
#         )
#         m2 = calc_m_linear_decreasing(
#             un_x(x), self._params[1][0], self._params[2][0], m
#         )
#         return torch.max(m1, m2)

#     def _calc_areas(self):
        
#         return (
#             0.5 * (self._params[2] 
#             - self._params[0]) * self._m[0]
#         )

#     def _calc_mean_cores(self):
#         return self._params[1]

#     def _calc_centroids(self):
#         return 1 / 3 * (
#             self._params[0] + self._params[1] + self._params[2]
#         )
    
#     def scale(self, m: torch.Tensor):

#         updated_m = self._intersect_m(m)
        
#         return Triangle(
#             self._params.x, updated_m.x
#         )

#     def truncate(self, m: torch.Tensor):
#         # TODO: FINISH
        
#         updated_m = self._intersect_m(m)

#         pt1 = calc_x_linear_increasing(updated_m[0], self._params[0], self._params[1], self._m[0])
#         pt2 = calc_x_linear_decreasing(updated_m[0], self._params[1], self._params[2], self._m[0])

#         params = self._params.replace(
#             ShapeParams(torch.cat(
#                 [pt1.unsqueeze(3), pt2.unsqueeze(3)], dim=3
#             ), False, updated_m.is_batch or self._params.is_batch), 1)

#         return Trapezoid(
#             params.x, updated_m.x
#         )


# class IsoscelesTriangle(ConvexPolygon):

#     def join(self, x: torch.Tensor) -> FuzzySet:

#         left_m = calc_m_linear_increasing(
#             un_x(x), self._params[0], self._params[1], self._m[0]
#         )
#         right_m = calc_m_linear_decreasing(
#             un_x(x), self._params[1], self._params[1] + (self._params[1] - self._params[0]), 
#             self._m[0]
#         )
#         return torch.max(left_m, right_m)

#     def _calc_areas(self):
        
#         return (
#             0.5 * (self._params[0] 
#             - self._params[1]) * self._m[0]
#         )

#     def _calc_mean_cores(self):
#         return self._params[1]

#     def _calc_centroids(self):
#         return self._params[1]

#     def scale(self, m: torch.Tensor) -> 'IsoscelesTriangle':
#         updated_m = self._intersect_m(m)
        
#         return IsoscelesTriangle(
#             self._params.x, updated_m.x
#         )

#     def truncate(self, m: FuzzySet) -> 'IsoscelesTrapezoid':
        
#         updated_m = self._intersect_m(m)
        
#         pt1 = calc_x_linear_increasing(updated_m[0], self._params[0], self._params[1], self._m[0])
#         pt2 = calc_x_linear_decreasing(updated_m[0], self._params[1], self._params[1] + self._params[1] - self._params[0], self._m[0])

#         params = self._params.replace(
#             ShapeParams(torch.cat(
#                 [pt1.unsqueeze(3), pt2.unsqueeze(3)], dim=3
#             ), False, updated_m.is_batch or self._params.is_batch), 1)

#         return IsoscelesTrapezoid(
#             params.x, updated_m.x
#         )


# class Logistic(SimpleShape):

#     def __init__(
#         self, biases: torch.Tensor, scales: torch.Tensor, m: torch.Tensor=None
#     ):
#         self._biases = ShapeParams(biases, True, biases.dim() == 3)
#         self._scales = ShapeParams(scales, True, scales.dim() == 3)
#         if m is None:
#             m = torch.ones(
#                 *scales.size(), device=scales.device, dtype=scales.dtype
#             )

#         super().__init__(
#             self._biases.n_features, 
#             self._biases.n_categories,
#             ShapeParams(m, True, m.dim() == 3)
#         )

#     @property
#     def biases(self):
#         return self._biases
    
#     @property
#     def scales(self):
#         return self._scales
    
#     @classmethod
#     def from_combined(cls, params: torch.Tensor, m: torch.Tensor=None):

#         if params.dim() == 4:

#             return cls(params[:,:,:,0], params[:,:,:,1], m)
#         return cls(params[:,:,0], params[:,:,1], m)


# class LogisticBell(Logistic):

#     def join(self, x: torch.Tensor) -> FuzzySet:
#         z = self._scales[0] * (x[:,:,None] - self._biases[0])
#         sig = torch.sigmoid(z)
#         # not 4 / s
#         return 4  * (1 - sig) * sig * self._m[0]

#     def _calc_areas(self):
#         return 4 * self._m[0] / self._biases[0]
        
#     def _calc_mean_cores(self):
#         return self._biases[0]

#     def _calc_centroids(self):
#         return self._biases[0]

#     def scale(self, m: torch.Tensor) -> 'LogisticBell':
#         updated_m = self._intersect_m(m)
#         return LogisticBell(
#             self._biases.x, self._scales.x, updated_m.x
#         )

#     def truncate(self, m: torch.Tensor) -> 'LogisticTrapezoid':

#         return LogisticTrapezoid(
#             self._biases.x, self._scales.x,  m, self._m.x 
#         )


# class LogisticTrapezoid(Logistic):
    
#     def __init__(
#         self, biases: torch.Tensor, scales: torch.Tensor, 
#         truncated_m: torch.Tensor=None, scaled_m: torch.Tensor=None
#     ):
#         super().__init__(biases, scales, scaled_m)

#         if truncated_m is None:
#             truncated_m = torch.ones(self._m.x.size(), device=self._m.x.device)

#         self._truncated_m = self._intersect_m(truncated_m)
        
#         dx = calc_dx_logistic(self._truncated_m[0], self._scales[0], self._m[0])
#         self._dx = ShapeParams(dx, True, dx.dim() == 3)
#         self._pts = ShapeParams(torch.stack([
#             self._biases.x - self._dx.x,
#             self._biases.x + self._dx.x
#         ], dim=dx.dim()), False, dx.dim() == 3)

#     @property
#     def dx(self):
#         return self._dx
    
#     @property
#     def m(self):
#         return self._truncated_m[0]

#     def join(self, x: torch.Tensor) -> 'FuzzySet':
        
#         inside = check_contains(x, self._pts[0], self._pts[1]).float()
#         m1 = calc_m_logistic(un_x(x), self._biases[0], self._scales[0], self._m[0]) * (1 - inside)
#         m2 = self._truncated_m[0] * inside
#         return torch.max(m1, m2)

#     def _calc_areas(self):
#         # symmetrical so multiply by 2
#         return 2 * calc_area_logistic_one_side(
#             self._pts[0], self._biases[0], self._scales[0], self._m[0]
#         )
        
#     def _calc_mean_cores(self):
#         return self._biases[0]

#     def _calc_centroids(self):
#         return self._biases[0]

#     def scale(self, m: torch.Tensor) -> 'LogisticTrapezoid':
#         updated_m = self._intersect_m(m)
#         truncated_m = self._truncated_m.x * updated_m.x

#         return LogisticTrapezoid(
#             self._biases.x, self._scales.x, truncated_m, updated_m.x 
#         )

#     def truncate(self, m: torch.Tensor) -> 'LogisticTrapezoid':
#         truncated_m = self._intersect_m(m, self._truncated_m)
#         return LogisticTrapezoid(
#             self._biases.x, self._scales.x, truncated_m.x, self._m.x 
#         )


# class RightLogistic(Logistic):
    
#     def __init__(
#         self, biases: torch.Tensor, scales: torch.Tensor, is_right: bool=True,
#         m: torch.Tensor= None
#     ):
#         super().__init__(biases, scales, m)
#         self._is_right = is_right
#         self._direction = is_right * 2 - 1
    
#     def _on_side(self, x: torch.Tensor):
#         if self._is_right:
#             side = x >= self._biases[0][:,:,0]
#         else: side = x <= self._biases[0][:,:,0]
#         return side.unsqueeze(2)

#     def join(self, x: torch.Tensor):
#         return calc_m_logistic(
#             un_x(x), self._biases[0], 
#             self._scales[0], self._m[0]
#         ) * self._on_side(x).float()

#     def _calc_areas(self):
#         return 2 * self._m[0] / self._biases[0]

#     def _calc_mean_cores(self):
#         return self._biases[0]

#     def _calc_centroids(self):
#         dx = calc_dx_logistic(torch.tensor(
#             0.75, device=self._biases.x.device
#         ), self._scales[0])
#         return self._biases[0] + self._direction * dx

#     def scale(self, m: torch.Tensor) -> 'RightLogistic':
#         updated_m = self._intersect_m(m)
        
#         return RightLogistic(
#             self._biases.x, self._scales.x, self._is_right, updated_m.x
#         )

#     def truncate(self, m: torch.Tensor) -> 'LogisticTrapezoid':
#         truncated_m = self._intersect_m(m)
#         return LogisticTrapezoid(
#             self._biases.x, self._scales.x, truncated_m.x, self._m.x 
#         )

#     @classmethod
#     def from_combined(cls, params: torch.Tensor, is_right: bool=True,m: torch.Tensor=None):

#         if params.dim() == 4:

#             return cls(params[:,:,:,0], params[:,:,:,1], is_right, m)
#         return cls(params[:,:,0], params[:,:,1], is_right, m)


# class RightLogisticTrapezoid(Logistic):

#     # TODO: Think about the "logistic trapezoids" more

#     def __init__(
#         self, biases: torch.Tensor, scales: torch.Tensor, is_right: bool, 
#         truncated_m: torch.Tensor=None, scaled_m: torch.Tensor=None
        
#     ):
#         super().__init__(biases, scales, scaled_m)

#         if truncated_m is None:
#             truncated_m = torch.ones(self._m.x.size(), device=self._m.x.device)

#         self._truncated_m = self._intersect_m(truncated_m)
        
#         dx = calc_dx_logistic(self._truncated_m[0], self._scales[0], self._m[0])
#         self._dx = ShapeParams(dx, True, dx.dim() == 3)
#         self._is_right = is_right
#         self._direction = is_right * 2 - 1
#         self._pts = ShapeParams(self._biases.x + self._direction * dx, True, dx.dim() == 3)

#     @property
#     def dx(self):
#         return self._dx

#     @property
#     def m(self):
#         return self._truncated_m[0]

#     def _contains(self, x: torch.Tensor):
#         if self._is_right:
#             square_contains = (x >= self._biases[0]) & (x <= self._pts[0])
#             logistic_contains = x >= self._pts[0]
#         else:
#             square_contains = (x <= self._biases[0]) & (x >= self._pts[0])
#             logistic_contains = x <= self._pts[0]
#         return square_contains.float(), logistic_contains.float()

#     def join(self, x: torch.Tensor) -> 'FuzzySet':
        
#         square_contains, logistic_contains = self._contains(x)
        
#         m1 = calc_m_logistic(
#             un_x(x), self._biases[0], self._scales[0], self._m[0]
#         ) * logistic_contains
#         m2 = self._m[0] * square_contains
#         return torch.max(m1, m2)

#     def _calc_areas(self):
#         a1 = calc_area_logistic_one_side(
#             self._pts[0], self._biases[0], self._scales[0], 
#             self._m[0], left=not self._is_right)
#         a2 = 0.5 * (self._biases[0] + self._pts[0]) * self._m[0]
#         return a1 + a2

#     def _calc_mean_cores(self):
#         return 0.5 * (self._biases[0] + self._pts[0])    

#     def _calc_centroids(self):

#         # area up to "dx"
#         p = torch.sigmoid(self._scales[0] * (-self._dx[0])) # check
#         centroid_logistic = self._biases[0] + torch.logit(p / 2) / self._scales[0]
#         centroid_square = self._biases[0] - self._dx[0] / 2

#         centroid = (centroid_logistic * p + centroid_square * self._dx[0]) / (p + self._dx[0])
#         if self._is_right:
#             return self._biases[0] + self._biases[0] - centroid
#         return centroid

#     def scale(self, m: torch.Tensor) -> 'RightLogisticTrapezoid':
#         updated_m = self._intersect_m(m)
#         truncated_m = self._truncated_m.x * updated_m.x

#         return RightLogisticTrapezoid(
#             self._biases.x, self._scales.x, self._is_right, truncated_m, updated_m.x 
#         )

#     def truncate(self, m: torch.Tensor) -> 'RightLogisticTrapezoid':

#         truncated_m = self._intersect_m(m, self._truncated_m)
#         return RightLogisticTrapezoid(
#             self._biases.x, self._scales.x, self._is_right, truncated_m.x, self._m.x 
#         )

#     @classmethod
#     def from_combined(cls, params: torch.Tensor, is_right: bool=True,m: torch.Tensor=None):

#         if params.dim() == 4:

#             return cls(params[:,:,:,0], params[:,:,:,1], is_right, m)
#         return cls(params[:,:,0], params[:,:,1], is_right, m)


# class Trapezoid(ConvexPolygon):

#     P = 4

#     def join(self, x: torch.Tensor) -> FuzzySet:

#         m1 = calc_m_linear_increasing(un_x(x), self._params[0], self._params[1], self._m[0])
#         m2 = calc_m_flat(un_x(x), self._params[1], self._params[2], self._m[0])
#         m3 = calc_m_linear_decreasing(un_x(x), self._params[2], self._params[3], self._m[0])

#         return torch.max(torch.max(
#             m1, m2
#         ), m3)

#     def _calc_areas(self):
        
#         return (
#             0.5 * (self._params[2] 
#             - self._params[0]) * self._m[0]
#         )

#     def _calc_mean_cores(self):
#         return 0.5 * (self._params[1] + self._params[2])

#     def _calc_centroids(self):
#         d1 = 0.5 * (self._params[1] - self._params[0])
#         d2 = self._params[2] - self._params[1]
#         d3 = 0.5 * (self._params[3] - self._params[2])

#         return (
#             d1 * (2 / 3 * self._params[1] + 1 / 3 * self._params[0]) +
#             d2 * (1 / 2 * self._params[2] + 1 / 2 *  self._params[1]) + 
#             d3 * (1 / 3 * self._params[3] + 2 / 3 * self._params[2])
#         ) / (d1 + d2 + d3)

#     def scale(self, m: torch.Tensor) -> 'Trapezoid':
#         updated_m = self._intersect_m(m)
#         return Trapezoid(
#             self._params.x, updated_m.x
#         )

#     def truncate(self, m: torch.Tensor) -> 'Trapezoid':
#         updated_m = self._intersect_m(m)

#         # m = ShapeParams(m, True, m.dim() == 3)
#         left_x = ShapeParams(calc_x_linear_increasing(
#             updated_m.x, self._params[0], self._params[1], self._m[0]
#         ), True, updated_m.is_batch or self._params.is_batch)

#         right_x = ShapeParams(calc_x_linear_decreasing(
#             updated_m.x, self._params[2], self._params[3], self._m[0]
#         ), True, updated_m.is_batch or self._params.is_batch)        
        
#         params = self._params.replace(left_x, 1)
#         params = params.replace(right_x, 2)

#         return Trapezoid(
#             params.x, updated_m.x, 
#         )


# class IsoscelesTrapezoid(ConvexPolygon):

#     P = 3

#     def join(self, x: torch.Tensor) -> 'FuzzySet':

#         left_m = calc_m_linear_increasing(
#             un_x(x), self._params[0], self._params[1], self._m[0]
#         )
#         middle = calc_m_flat(un_x(x), self._params[1], self._params[2], self._m[0])
#         pt3 = self._params[1] - self._params[0] + self._params[2]
#         right_m = calc_m_linear_decreasing(
#             un_x(x), self._params[2], pt3, self._m[0]
#         )
#         return torch.max(torch.max(left_m, middle), right_m)
    
#     @property
#     def a(self):
#         return (
#             self._params[2] - self._params[0] + 
#             self._params[1] - self._params[0]
#         )

#     @property
#     def b(self):
#         return self._params[2] - self._params[1]

#     def _calc_areas(self):
        
#         return (
#             0.5 * (self.a + self.b) * self._m[0]
#         )

#     def _calc_mean_cores(self):
#         return 0.5 * (self._params[2] + self._params[1])

#     def _calc_centroids(self):
#         return self.mean_cores

#     def scale(self, m: torch.Tensor) -> 'IsoscelesTrapezoid':
#         updated_m = self._intersect_m(m)
#         return IsoscelesTrapezoid(self._params.x, updated_m.x)

#     def truncate(self, m: torch.Tensor) -> 'IsoscelesTrapezoid':
#         updated_m = self._intersect_m(m)

#         left_x = ShapeParams(calc_x_linear_increasing(
#             updated_m.x, self._params[0], self._params[1], self._m[0]
#         ), True, updated_m.is_batch or self._params.is_batch)

#         right_x = ShapeParams(
#             self._params[2] + self._params[1] - left_x[0],
#             True, updated_m.is_batch or self._params.is_batch
#         )
#         params = self._params.replace(
#             left_x, 1
#         )
#         params = params.replace(
#             right_x, 2
#         )
#         return IsoscelesTrapezoid(params.x, updated_m.x)


class IncreasingRightTrapezoid(ConvexPolygon):

    PT = 3

    def join(self, x: torch.Tensor) -> 'FuzzySet':
        m = calc_m_linear_increasing(
            unsqueeze(x), self._params.pt(0), self._params.pt(1), self._m.data
        )
        m2 = calc_m_flat(unsqueeze(x), self._params.pt(1), self._params.pt(2), self._m.data)

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
            0.5 * (self.a + self.b) * self._m.data, self._m
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

    def scale(self, m: FuzzySet) -> 'IncreasingRightTrapezoid':
        return IncreasingRightTrapezoid(self._params, m.intersect(self._m))

    def truncate(self, m: FuzzySet) -> 'IncreasingRightTrapezoid':
        updated_m = m.intersect(self._m)
        
        x = calc_x_linear_increasing(
            updated_m, self._params.pt(0), self._params.pt(1), self._m
        )
        params = self._params.replace(x, 1, True)
        return IncreasingRightTrapezoid(params, updated_m)


# class DecreasingRightTrapezoid(ConvexPolygon):

#     P = 3

#     def join(self, x: torch.Tensor) -> 'FuzzySet':

#         m = calc_m_linear_decreasing(
#             un_x(x), self._params[1], self._params[2], self._m[0]
#         )
#         m2 = calc_m_flat(un_x(x), self._params[0], self._params[1], self._m[0])
    
#         return torch.max(m, m2)
    
#     @property
#     def a(self):
#         return (
#             self._params[2] - self._params[0]
#         )

#     @property
#     def b(self):
#         return self._params[1] - self._params[0]

#     def _calc_areas(self):
        
#         return (
#             0.5 * (self.a + self.b) * self._m[0]
#         )

#     def _calc_mean_cores(self):
#         return 0.5 * (self._params[0] + self._params[1])

#     def _calc_centroids(self):
#         d1 = self._params[1] - self._params[0]
#         d2 = 0.5 * (self._params[2] - self._params[1])
        
#         return (
#             d1 * (1 / 2 * self._params[1] + 1 / 2 * self._params[0]) +
#             d2 * (1 / 3 * self._params[2] + 2 / 3 * self._params[1])
#         ) / (d1 + d2)

#     def scale(self, m: torch.Tensor) -> 'DecreasingRightTrapezoid':
#         updated_m = self._intersect_m(m)
#         return DecreasingRightTrapezoid(self._params.x, updated_m.x)

#     def truncate(self, m: torch.Tensor) -> 'DecreasingRightTrapezoid':
#         updated_m = self._intersect_m(m)
        
#         x = ShapeParams(calc_x_linear_decreasing(
#             updated_m[0], self._params[1], self._params[2], self._m[0]
#         ), True, updated_m.is_batch or self._params.is_batch)
#         params = self._params.replace(x, 1)
#         return DecreasingRightTrapezoid(params.x, updated_m.x)


# # class Params(object):

# #     def __init__(self, x: torch.Tensor, is_singular: bool, is_batch: bool=True):

# #         self._is_singular = is_singular
# #         self._is_batch = is_batch
# #         if is_singular and is_batch:
# #             if x.dim() != 3:
# #                 raise ValueError("Dimension of x must be 3 not {}".format(x.dim()))
# #             x = x[:, :, :, None]
# #         elif is_singular:
# #             if x.dim() != 2:
# #                 raise ValueError("Dimension of x must be 2 not {}".format(x.dim()))
# #             x = x[None,:,:,None]
# #         elif not is_batch:
# #             if x.dim() != 3:
# #                 raise ValueError("Dimension of x must be 3 not {}".format(x.dim()))
# #             x = x[None,:,:,:]
# #         else:
# #             if x.dim() != 4:
# #                 raise ValueError("Dimension of x must be 4 not {}".format(x.dim()))

# #         self._n_features = x.size(1)
# #         self._n_categories = x.size(2)
# #         self._batch_size = x.size(0) if self._is_batch else None
# #         self._n_params = x.size(3)
# #         self._x = x

# #     def is_aligned_with(self, other) -> bool:
        
# #         if self._is_batch and other._is_batch:
# #             return self._x.size()[0:3] == other._x.size()[0:3]
        
# #         return self._x.size()[1:3] == other._x.size()[1:3]
    
# #     @property
# #     def is_batch(self) -> bool:
# #         return self._is_batch
    
# #     @property
# #     def is_singular(self) -> bool:
# #         return self._is_singular
    
# #     def _expand(self, other):
# #         '''
# #         Helper method to expand self or other if one is a batch and the other isn't
# #         '''

# #         if self.is_batch is other.is_batch:
# #             return self, other
# #         if self.is_batch:
# #             # expand other
# #             x = other._x.repeat(self._x.size(0), 1, 1, 1)
# #             if other.is_singular:
# #                 x = x[:,:,:,0]
# #             return self, Params(x, other.is_singular, True)
        
# #         x = self._x.repeat(other._x.size(0), 1, 1, 1)
# #         if self.is_singular:
# #             x = x[:,:,:,0]
            
# #         return Params(x, self.is_singular, True), other

# #     def insert(self, other, position: int):
        
# #         self, other = self._expand(other)
# #         sz = list(self._x.size())
# #         other_sz = other._x.size(3)
# #         sz[3] += other_sz
# #         x = torch.empty(*sz, dtype=self._x.dtype, device=self._x.device)
# #         if position > 0:
# #             x[:,:,:,:position] = self._x[:,:,:,:position]
# #         x[:,:,:,position:other_sz + position] = other._x[:,:,:,:]
# #         if position < self._x.size(3):
# #             x[:,:,:,other_sz + position:] = self._x[:,:,:,position:]
# #         if self._is_batch:
# #             return Params(
# #                 x, False, True
# #             )
# #         else:
# #             return Params(
# #                 x[0], False, False
# #             )

# #     def replace(self, other, position: typing.Union[typing.Tuple[int, int], int]):

# #         self, other = self._expand(other)
# #         if isinstance(position, tuple):
# #             position = range(*position)
# #         else:
# #             position = [position, position + 1]
        
# #         lower_bound = position[0]
# #         upper_bound = position[-1]
        
# #         sz = list(self._x.size())
# #         other_sz = other._x.size(3)
# #         sz[3] = sz[3] - len(position) + other_sz + 1

# #         x = torch.empty(*sz, dtype=self._x.dtype, device=self._x.device)
# #         if lower_bound > 0:
# #             x[:,:,:,:lower_bound] = self._x[:,:,:,:lower_bound]
# #         x[:,:,:,lower_bound:lower_bound + other_sz] = other._x
# #         if upper_bound < self._x.size(3):
# #             x[:,:,:,lower_bound + other_sz:] = self._x[:,:,:,upper_bound:]
        
# #         if self._is_batch:
# #             return Params(
# #                 x, False, True
# #             )
# #         else:
# #             return Params(
# #                 x[0], False, False
# #             )
    
# #     def clone(self):
# #         return Params(torch.clone(self.x), self._is_singular, self._is_batch)

# #     @property
# #     def x(self):
# #         if self._is_singular and self._is_batch:
# #             return self._x[:,:,:,0]
# #         elif self._is_singular:
# #             return self._x[0,:,:,0]
# #         elif not self._is_batch:
# #             return self._x[0,:,:,:]
# #         return self._x
    
# #     def __getitem__(self, idx: typing.Union[int, slice]):
# #         return self._x[:,:,:,idx]

# #     @property
# #     def n_features(self):
# #         return self._n_features

# #     @property
# #     def n_categories(self):
# #         return self._n_categories

# #     @property
# #     def batch_size(self):
# #         return self._batch_size
    
# #     @property
# #     def n_params(self):
# #         return self._n_params


# class SimpleShape(Shape):

#     def __init__(self, n_features: int, n_categories: int, m: ShapeParams):

#         super().__init__(n_features, n_categories)
#         self._m = m

#     @property
#     def m(self):
#         return self._m[0]

#     def _intersect_m(self, m: FuzzySet, other: ShapeParams=None):

#         other = other or self._m

#         if other.x.dim() == 3 and m.dim() == 2:
#             m = m.unsqueeze(0)

#         updated_m = torch.min(m, other_x)

#         return ShapeParams(
#             updated_m, True, updated_m.dim() == 3
#         )