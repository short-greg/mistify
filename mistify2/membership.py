from abc import abstractmethod, abstractproperty
import typing
import torch

# TODO: Change so that it uses the FuzzySet class

def check_contains(x: torch.Tensor, pt1: torch.Tensor, pt2: torch.Tensor):
    
    return (x >= pt1) & (x <= pt2)


def calc_m_flat(x, pt1, pt2, m):

    return m * check_contains(x, pt1, pt2).float()


def calc_m_linear_increasing(x: torch.Tensor, pt1: torch.Tensor, pt2: torch.Tensor, m=1.0):
    return (x - pt1) * (m / (pt2 - pt1)) * check_contains(x, pt1, pt2).float() 


def calc_m_linear_decreasing(x: torch.Tensor, pt1: torch.Tensor, pt2: torch.Tensor, m=1.0):
    return ((x - pt1) * (-m / (pt2 - pt1)) + m) * check_contains(x, pt1, pt2).float()


def calc_x_linear_increasing(m0, pt1, pt2, m):
    # NOTE: To save on computational costs do not perform checks to see
    # if m0 is greater than m

    m0 = torch.min(m0, m)
    x = m0 * (pt2 - pt1) / m + pt1
    torch.nan_to_num_(x, 0.0, 0.0)
    return x


def calc_x_linear_decreasing(m0, pt1, pt2, m=1):

    m0 = torch.min(m0, m)
    x = -(m0 - 1) * (pt2 - pt1) / m + pt1
    torch.nan_to_num_(x, 0.0, 0.0)
    return x


def calc_m_logistic(x, b, s, m):

    z = s * (x - b)
    multiplier = 4 * m
    y = torch.sigmoid(z)
    return multiplier * y * (1 - y)


def calc_dx_logistic(m0, s, m_base=1):
    
    m = m0 / m_base
    dx = -torch.log((-m - 2 * torch.sqrt(1 - m) + 2) / (m)).float()
    dx = dx / s
    return dx


def calc_area_logistic(s, m_base=1, left=True):
    
    return 4 * m_base / s


def calc_area_logistic_one_side(x, b, s, m_base=1):
    
    z = s * (x - b)
    left = (z < 0).float()
    a = torch.sigmoid(z)
    # only calculate area of one side so 
    # flip the probability
    a = left * a + (1 - left) * (1 - a)

    return a * m_base * 4 / s

def un_x(x):
    return x[:,:,None]


class Params(object):

    def __init__(self, x: torch.Tensor, is_singular: bool, is_batch: bool=True):

        self._is_singular = is_singular
        self._is_batch = is_batch
        if is_singular and is_batch:
            if x.dim() != 3:
                raise ValueError("Dimension of x must be 3 not {}".format(x.dim()))
            x = x[:, :, :, None]
        elif is_singular:
            if x.dim() != 2:
                raise ValueError("Dimension of x must be 2 not {}".format(x.dim()))
            x = x[None,:,:,None]
        elif not is_batch:
            if x.dim() != 3:
                raise ValueError("Dimension of x must be 3 not {}".format(x.dim()))
            x = x[None,:,:,:]
        else:
            if x.dim() != 4:
                raise ValueError("Dimension of x must be 4 not {}".format(x.dim()))

        self._n_features = x.size(1)
        self._n_categories = x.size(2)
        self._batch_size = x.size(0) if self._is_batch else None
        self._n_params = x.size(3)
        self._x = x

    def is_aligned_with(self, other) -> bool:
        
        if self._is_batch and other._is_batch:
            return self._x.size()[0:3] == other._x.size()[0:3]
        
        return self._x.size()[1:3] == other._x.size()[1:3]
    
    @property
    def is_batch(self) -> bool:
        return self._is_batch
    
    @property
    def is_singular(self) -> bool:
        return self._is_singular
    
    def _expand(self, other):
        '''
        Helper method to expand self or other if one is a batch and the other isn't
        '''

        if self.is_batch is other.is_batch:
            return self, other
        if self.is_batch:
            # expand other
            x = other._x.repeat(self._x.size(0), 1, 1, 1)
            if other.is_singular:
                x = x[:,:,:,0]
            return self, Params(x, other.is_singular, True)
        
        x = self._x.repeat(other._x.size(0), 1, 1, 1)
        if self.is_singular:
            x = x[:,:,:,0]
            
        return Params(x, self.is_singular, True), other

    def insert(self, other, position: int):
        
        self, other = self._expand(other)
        sz = list(self._x.size())
        other_sz = other._x.size(3)
        sz[3] += other_sz
        x = torch.empty(*sz, dtype=self._x.dtype, device=self._x.device)
        if position > 0:
            x[:,:,:,:position] = self._x[:,:,:,:position]
        x[:,:,:,position:other_sz + position] = other._x[:,:,:,:]
        if position < self._x.size(3):
            x[:,:,:,other_sz + position:] = self._x[:,:,:,position:]
        if self._is_batch:
            return Params(
                x, False, True
            )
        else:
            return Params(
                x[0], False, False
            )

    def replace(self, other, position: typing.Union[typing.Tuple[int, int], int]):

        self, other = self._expand(other)
        if isinstance(position, tuple):
            position = range(*position)
        else:
            position = [position, position + 1]
        
        lower_bound = position[0]
        upper_bound = position[-1]
        
        sz = list(self._x.size())
        other_sz = other._x.size(3)
        sz[3] = sz[3] - len(position) + other_sz + 1

        x = torch.empty(*sz, dtype=self._x.dtype, device=self._x.device)
        if lower_bound > 0:
            x[:,:,:,:lower_bound] = self._x[:,:,:,:lower_bound]
        x[:,:,:,lower_bound:lower_bound + other_sz] = other._x
        if upper_bound < self._x.size(3):
            x[:,:,:,lower_bound + other_sz:] = self._x[:,:,:,upper_bound:]
        
        if self._is_batch:
            return Params(
                x, False, True
            )
        else:
            return Params(
                x[0], False, False
            )
    
    def clone(self):
        return Params(torch.clone(self.x), self._is_singular, self._is_batch)

    @property
    def x(self):
        if self._is_singular and self._is_batch:
            return self._x[:,:,:,0]
        elif self._is_singular:
            return self._x[0,:,:,0]
        elif not self._is_batch:
            return self._x[0,:,:,:]
        return self._x
    
    def __getitem__(self, idx: typing.Union[int, slice]):
        return self._x[:,:,:,idx]

    @property
    def n_features(self):
        return self._n_features

    @property
    def n_categories(self):
        return self._n_categories

    @property
    def batch_size(self):
        return self._batch_size
    
    @property
    def n_params(self):
        return self._n_params


class Shape(object):

    def __init__(self, n_features: int, n_categories: int):

        super().__init__()
        self._areas = None
        self._modes = None
        self._centroids = None
        self._n_features = n_features
        self._n_categories = n_categories

    @property
    def n_features(self):
        return self._n_features
    
    @property
    def n_categories(self):
        return self._n_categories

    def join(self, x: torch.Tensor):
        pass

    @abstractmethod
    def _calc_areas(self):
        pass

    @property
    def areas(self):
        if self._areas is None:
            self._areas = self._calc_areas()
        return self._areas

    @abstractmethod
    def _calc_mean_cores(self):
        pass

    @property
    def mean_cores(self):
        if self._modes is None:
            self._modes = self._calc_mean_cores()
        return self._modes

    @abstractmethod
    def _calc_centroids(self):
        pass

    @abstractproperty
    def m(self):
        pass

    @property
    def centroids(self):
        if self._centroids is None:
            self._centroids = self._calc_centroids()
        return self._centroids
    
    @abstractmethod
    def scale(self, m: torch.Tensor):
        pass

    @abstractmethod
    def truncate(self, m: torch.Tensor):
        pass

    @abstractmethod
    def join(self, x: torch.Tensor):
        pass


class SimpleShape(Shape):

    def __init__(self, n_features: int, n_categories: int, m: Params):

        super().__init__(n_features, n_categories)
        self._m = m

    @property
    def m(self):
        return self._m[0]

    def _intersect_m(self, m: torch.Tensor, other: Params=None):

        other = other or self._m
        if other.x.dim() == 2 and m.dim() == 3:
            other_x = other.x.unsqueeze(0)
        else:
            other_x = other.x

        if other.x.dim() == 3 and m.dim() == 2:
            m = m.unsqueeze(0)

        updated_m = torch.min(m, other_x)

        return Params(
            updated_m, True, updated_m.dim() == 3
        )


class ConvexPolygon(SimpleShape):

    def __init__(self, params: torch.Tensor, m: typing.Optional[torch.Tensor]=None):

        self._params = Params(params, False, params.dim() == 4)
        
        if m is None and params.dim() == 4:
            m = torch.ones(
                self._params.batch_size, self._params.n_features, self._params.n_categories, device=params.device
            )
        elif m is None:
            m = torch.ones(self._params.n_features, self._params.n_categories, device=params.device)

        super().__init__(self._params.n_features, self._params.n_categories, Params(m, True, m.dim() == 3))

        if not self._params.is_aligned_with(self._m):
            raise ValueError("Membership size does not match with params size{} {} ".format())

    @abstractmethod
    def to_mesh(self):
        pass


class IncreasingRightTriangle(ConvexPolygon):

    def join(self, x: torch.Tensor):
        
        return calc_m_linear_increasing(
            un_x(x), self._params[0], self._params[1], self._m[0]
        )

    def _calc_areas(self):
        
        return (
            0.5 * (self._params[1]
            - self._params[0]) * self._m[0]
        )

    def _calc_mean_cores(self):
        
        return self._params[1]

    def _calc_centroids(self):
        
        p1, p2 = 1 / 3, 2 / 3

        return p1 * self._params[0] + p2 * self._params[1]
    
    def scale(self, m: torch.Tensor):
        updated_m = self._intersect_m(m)
        
        return IncreasingRightTriangle(
            self._params.x, updated_m.x
        )

    def truncate(self, m: torch.Tensor):
        # TODO: FINISH
        updated_m = self._intersect_m(m)

        pt = Params(calc_x_linear_increasing(
            updated_m[0], self._params[0], self._params[1], self._m[0]),
            True, m.dim() == 3)
        params = self._params.insert(pt, 1)
        return IncreasingRightTrapezoid(
            params.x, updated_m.x
        )


class DecreasingRightTriangle(ConvexPolygon):
    
    def join(self, x: torch.Tensor):
        
        return calc_m_linear_decreasing(
            un_x(x), self._params[0], self._params[1], self._m[0]
        )

    def _calc_areas(self):
        
        return (
            0.5 * (self._params[1]
            - self._params[0]) * self._m[0]
        )

    def _calc_mean_cores(self):
        
        return self._params[0]

    def _calc_centroids(self):
        return 2 / 3 * self._params[0] + 1 / 3 * self._params[1]
    
    def scale(self, m: torch.Tensor):
        updated_m = self._intersect_m(m)
        
        return DecreasingRightTriangle(
            self._params.x, updated_m.x
        )

    def truncate(self, m: torch.Tensor):
        
        # TODO: FINISH
        updated_m = self._intersect_m(m)
        pt = Params(
            calc_x_linear_decreasing(
                updated_m[0], self._params[0], self._params[1], self._m[0]
            ), True, m.dim() == 3
        )

        params = self._params.insert(pt, 1)

        return DecreasingRightTrapezoid(
            params.x, updated_m.x
        )


class Triangle(ConvexPolygon):

    def join(self, x: torch.Tensor):

        m = self._m[0][0]
        
        m1 = calc_m_linear_increasing(
            un_x(x), self._params[0][0], self._params[1][0], m
        )
        m2 = calc_m_linear_decreasing(
            un_x(x), self._params[1][0], self._params[2][0], m
        )
        return torch.max(m1, m2)

    def _calc_areas(self):
        
        return (
            0.5 * (self._params[2] 
            - self._params[0]) * self._m[0]
        )

    def _calc_mean_cores(self):
        return self._params[1]

    def _calc_centroids(self):
        return 1 / 3 * (
            self._params[0] + self._params[1] + self._params[2]
        )
    
    def scale(self, m: torch.Tensor):

        updated_m = self._intersect_m(m)
        
        return Triangle(
            self._params.x, updated_m.x
        )

    def truncate(self, m: torch.Tensor):
        # TODO: FINISH
        
        updated_m = self._intersect_m(m)

        pt1 = calc_x_linear_increasing(updated_m[0], self._params[0], self._params[1], self._m[0])
        pt2 = calc_x_linear_decreasing(updated_m[0], self._params[1], self._params[2], self._m[0])

        params = self._params.replace(
            Params(torch.cat(
                [pt1.unsqueeze(3), pt2.unsqueeze(3)], dim=3
            ), False, updated_m.is_batch or self._params.is_batch), 1)

        return Trapezoid(
            params.x, updated_m.x
        )


class IsoscelesTriangle(ConvexPolygon):

    def join(self, x: torch.Tensor):

        left_m = calc_m_linear_increasing(
            un_x(x), self._params[0], self._params[1], self._m[0]
        )
        right_m = calc_m_linear_decreasing(
            un_x(x), self._params[1], self._params[1] + (self._params[1] - self._params[0]), 
            self._m[0]
        )
        return torch.max(left_m, right_m)

    def _calc_areas(self):
        
        return (
            0.5 * (self._params[0] 
            - self._params[1]) * self._m[0]
        )

    def _calc_mean_cores(self):
        return self._params[1]

    def _calc_centroids(self):
        return self._params[1]

    def scale(self, m: torch.Tensor):
        updated_m = self._intersect_m(m)
        
        return IsoscelesTriangle(
            self._params.x, updated_m.x
        )

    def truncate(self, m: torch.Tensor):
        
        updated_m = self._intersect_m(m)

        pt1 = calc_x_linear_increasing(updated_m[0], self._params[0], self._params[1], self._m[0])
        pt2 = calc_x_linear_decreasing(updated_m[0], self._params[1], self._params[1] + self._params[1] - self._params[0], self._m[0])

        params = self._params.replace(
            Params(torch.cat(
                [pt1.unsqueeze(3), pt2.unsqueeze(3)], dim=3
            ), False, updated_m.is_batch or self._params.is_batch), 1)

        return IsoscelesTrapezoid(
            params.x, updated_m.x
        )


class Logistic(SimpleShape):

    def __init__(
        self, biases: torch.Tensor, scales: torch.Tensor, m: torch.Tensor=None
    ):
        self._biases = Params(biases, True, biases.dim() == 3)
        self._scales = Params(scales, True, scales.dim() == 3)
        if m is None:
            m = torch.ones(
                *scales.size(), device=scales.device, dtype=scales.dtype
            )

        super().__init__(
            self._biases.n_features, 
            self._biases.n_categories,
            Params(m, True, m.dim() == 3)
        )

    @property
    def biases(self):
        return self._biases
    
    @property
    def scales(self):
        return self._scales
    
    @classmethod
    def from_combined(cls, params: torch.Tensor, m: torch.Tensor=None):

        if params.dim() == 4:

            return cls(params[:,:,:,0], params[:,:,:,1], m)
        return cls(params[:,:,0], params[:,:,1], m)


class LogisticBell(Logistic):

    def join(self, x: torch.Tensor):
        z = self._scales[0] * (x[:,:,None] - self._biases[0])
        sig = torch.sigmoid(z)
        # not 4 / s
        return 4  * (1 - sig) * sig * self._m[0]

    def _calc_areas(self):
        return 4 * self._m[0] / self._biases[0]
        
    def _calc_mean_cores(self):
        return self._biases[0]

    def _calc_centroids(self):
        return self._biases[0]

    def scale(self, m: torch.Tensor):
        updated_m = self._intersect_m(m)
        return LogisticBell(
            self._biases.x, self._scales.x, updated_m.x
        )

    def truncate(self, m: torch.Tensor):

        return LogisticTrapezoid(
            self._biases.x, self._scales.x,  m, self._m.x 
        )


class LogisticTrapezoid(Logistic):
    
    def __init__(
        self, biases: torch.Tensor, scales: torch.Tensor, 
        truncated_m: torch.Tensor=None, scaled_m: torch.Tensor=None
    ):
        super().__init__(biases, scales, scaled_m)

        if truncated_m is None:
            truncated_m = torch.ones(self._m.x.size(), device=self._m.x.device)

        self._truncated_m = self._intersect_m(truncated_m)
        
        dx = calc_dx_logistic(self._truncated_m[0], self._scales[0], self._m[0])
        self._dx = Params(dx, True, dx.dim() == 3)
        self._pts = Params(torch.stack([
            self._biases.x - self._dx.x,
            self._biases.x + self._dx.x
        ], dim=dx.dim()), False, dx.dim() == 3)

    @property
    def dx(self):
        return self._dx
    
    @property
    def m(self):
        return self._truncated_m[0]

    def join(self, x: torch.Tensor):
        
        inside = check_contains(x, self._pts[0], self._pts[1]).float()
        m1 = calc_m_logistic(un_x(x), self._biases[0], self._scales[0], self._m[0]) * (1 - inside)
        m2 = self._truncated_m[0] * inside
        return torch.max(m1, m2)

    def _calc_areas(self):
        # symmetrical so multiply by 2
        return 2 * calc_area_logistic_one_side(
            self._pts[0], self._biases[0], self._scales[0], self._m[0]
        )
        
    def _calc_mean_cores(self):
        return self._biases[0]

    def _calc_centroids(self):
        return self._biases[0]

    def scale(self, m: torch.Tensor):
        updated_m = self._intersect_m(m)
        truncated_m = self._truncated_m.x * updated_m.x

        return LogisticTrapezoid(
            self._biases.x, self._scales.x, truncated_m, updated_m.x 
        )

    def truncate(self, m: torch.Tensor):
        truncated_m = self._intersect_m(m, self._truncated_m)
        return LogisticTrapezoid(
            self._biases.x, self._scales.x, truncated_m.x, self._m.x 
        )


class RightLogistic(Logistic):
    
    def __init__(
        self, biases: torch.Tensor, scales: torch.Tensor, is_right: bool=True,
        m: torch.Tensor= None
    ):
        super().__init__(biases, scales, m)
        self._is_right = is_right
        self._direction = is_right * 2 - 1
    
    def _on_side(self, x: torch.Tensor):
        if self._is_right:
            side = x >= self._biases[0][:,:,0]
        else: side = x <= self._biases[0][:,:,0]
        return side.unsqueeze(2)

    def join(self, x: torch.Tensor):
        return calc_m_logistic(
            un_x(x), self._biases[0], 
            self._scales[0], self._m[0]
        ) * self._on_side(x).float()

    def _calc_areas(self):
        return 2 * self._m[0] / self._biases[0]

    def _calc_mean_cores(self):
        return self._biases[0]

    def _calc_centroids(self):
        dx = calc_dx_logistic(torch.tensor(
            0.75, device=self._biases.x.device
        ), self._scales[0])
        return self._biases[0] + self._direction * dx

    def scale(self, m: torch.Tensor):
        updated_m = self._intersect_m(m)
        
        return RightLogistic(
            self._biases.x, self._scales.x, self._is_right, updated_m.x
        )

    def truncate(self, m: torch.Tensor):
        truncated_m = self._intersect_m(m)
        return LogisticTrapezoid(
            self._biases.x, self._scales.x, truncated_m.x, self._m.x 
        )

    @classmethod
    def from_combined(cls, params: torch.Tensor, is_right: bool=True,m: torch.Tensor=None):

        if params.dim() == 4:

            return cls(params[:,:,:,0], params[:,:,:,1], is_right, m)
        return cls(params[:,:,0], params[:,:,1], is_right, m)


class RightLogisticTrapezoid(Logistic):

    # TODO: Think about the "logistic trapezoids" more

    def __init__(
        self, biases: torch.Tensor, scales: torch.Tensor, is_right: bool, 
        truncated_m: torch.Tensor=None, scaled_m: torch.Tensor=None
        
    ):
        super().__init__(biases, scales, scaled_m)

        if truncated_m is None:
            truncated_m = torch.ones(self._m.x.size(), device=self._m.x.device)

        self._truncated_m = self._intersect_m(truncated_m)
        
        dx = calc_dx_logistic(self._truncated_m[0], self._scales[0], self._m[0])
        self._dx = Params(dx, True, dx.dim() == 3)
        self._is_right = is_right
        self._direction = is_right * 2 - 1
        self._pts = Params(self._biases.x + self._direction * dx, True, dx.dim() == 3)

    @property
    def dx(self):
        return self._dx

    @property
    def m(self):
        return self._truncated_m[0]

    def _contains(self, x: torch.Tensor):
        if self._is_right:
            square_contains = (x >= self._biases[0]) & (x <= self._pts[0])
            logistic_contains = x >= self._pts[0]
        else:
            square_contains = (x <= self._biases[0]) & (x >= self._pts[0])
            logistic_contains = x <= self._pts[0]
        return square_contains.float(), logistic_contains.float()

    def join(self, x: torch.Tensor):
        
        square_contains, logistic_contains = self._contains(x)
        
        m1 = calc_m_logistic(
            un_x(x), self._biases[0], self._scales[0], self._m[0]
        ) * logistic_contains
        m2 = self._m[0] * square_contains
        return torch.max(m1, m2)

    def _calc_areas(self):
        a1 = calc_area_logistic_one_side(
            self._pts[0], self._biases[0], self._scales[0], 
            self._m[0], left=not self._is_right)
        a2 = 0.5 * (self._biases[0] + self._pts[0]) * self._m[0]
        return a1 + a2

    def _calc_mean_cores(self):
        return 0.5 * (self._biases[0] + self._pts[0])    

    def _calc_centroids(self):

        # area up to "dx"
        p = torch.sigmoid(self._scales[0] * (-self._dx[0])) # check
        centroid_logistic = self._biases[0] + torch.logit(p / 2) / self._scales[0]
        centroid_square = self._biases[0] - self._dx[0] / 2

        centroid = (centroid_logistic * p + centroid_square * self._dx[0]) / (p + self._dx[0])
        if self._is_right:
            return self._biases[0] + self._biases[0] - centroid
        return centroid

    def scale(self, m: torch.Tensor):
        updated_m = self._intersect_m(m)
        truncated_m = self._truncated_m.x * updated_m.x

        return RightLogisticTrapezoid(
            self._biases.x, self._scales.x, self._is_right, truncated_m, updated_m.x 
        )

    def truncate(self, m: torch.Tensor):

        truncated_m = self._intersect_m(m, self._truncated_m)
        return RightLogisticTrapezoid(
            self._biases.x, self._scales.x, self._is_right, truncated_m.x, self._m.x 
        )

    @classmethod
    def from_combined(cls, params: torch.Tensor, is_right: bool=True,m: torch.Tensor=None):

        if params.dim() == 4:

            return cls(params[:,:,:,0], params[:,:,:,1], is_right, m)
        return cls(params[:,:,0], params[:,:,1], is_right, m)


class Trapezoid(ConvexPolygon):

    P = 4

    def join(self, x: torch.Tensor):

        m1 = calc_m_linear_increasing(un_x(x), self._params[0], self._params[1], self._m[0])
        m2 = calc_m_flat(un_x(x), self._params[1], self._params[2], self._m[0])
        m3 = calc_m_linear_decreasing(un_x(x), self._params[2], self._params[3], self._m[0])

        return torch.max(torch.max(
            m1, m2
        ), m3)

    def _calc_areas(self):
        
        return (
            0.5 * (self._params[2] 
            - self._params[0]) * self._m[0]
        )

    def _calc_mean_cores(self):
        return 0.5 * (self._params[1] + self._params[2])

    def _calc_centroids(self):
        d1 = 0.5 * (self._params[1] - self._params[0])
        d2 = self._params[2] - self._params[1]
        d3 = 0.5 * (self._params[3] - self._params[2])

        return (
            d1 * (2 / 3 * self._params[1] + 1 / 3 * self._params[0]) +
            d2 * (1 / 2 * self._params[2] + 1 / 2 *  self._params[1]) + 
            d3 * (1 / 3 * self._params[3] + 2 / 3 * self._params[2])
        ) / (d1 + d2 + d3)

    def scale(self, m: torch.Tensor):
        updated_m = self._intersect_m(m)
        return Trapezoid(
            self._params.x, updated_m.x
        )

    def truncate(self, m: torch.Tensor):
        updated_m = self._intersect_m(m)

        # m = Params(m, True, m.dim() == 3)
        left_x = Params(calc_x_linear_increasing(
            updated_m.x, self._params[0], self._params[1], self._m[0]
        ), True, updated_m.is_batch or self._params.is_batch)

        right_x = Params(calc_x_linear_decreasing(
            updated_m.x, self._params[2], self._params[3], self._m[0]
        ), True, updated_m.is_batch or self._params.is_batch)        
        
        params = self._params.replace(left_x, 1)
        params = params.replace(right_x, 2)

        return Trapezoid(
            params.x, updated_m.x, 
        )


class IsoscelesTrapezoid(ConvexPolygon):

    P = 3

    def join(self, x: torch.Tensor):

        left_m = calc_m_linear_increasing(
            un_x(x), self._params[0], self._params[1], self._m[0]
        )
        middle = calc_m_flat(un_x(x), self._params[1], self._params[2], self._m[0])
        pt3 = self._params[1] - self._params[0] + self._params[2]
        right_m = calc_m_linear_decreasing(
            un_x(x), self._params[2], pt3, self._m[0]
        )
        return torch.max(torch.max(left_m, middle), right_m)
    
    @property
    def a(self):
        return (
            self._params[2] - self._params[0] + 
            self._params[1] - self._params[0]
        )

    @property
    def b(self):
        return self._params[2] - self._params[1]

    def _calc_areas(self):
        
        return (
            0.5 * (self.a + self.b) * self._m[0]
        )

    def _calc_mean_cores(self):
        return 0.5 * (self._params[2] + self._params[1])

    def _calc_centroids(self):
        return self.mean_cores

    def scale(self, m: torch.Tensor):
        updated_m = self._intersect_m(m)
        return IsoscelesTrapezoid(self._params.x, updated_m.x)

    def truncate(self, m: torch.Tensor):
        updated_m = self._intersect_m(m)

        left_x = Params(calc_x_linear_increasing(
            updated_m.x, self._params[0], self._params[1], self._m[0]
        ), True, updated_m.is_batch or self._params.is_batch)

        right_x = Params(
            self._params[2] + self._params[1] - left_x[0],
            True, updated_m.is_batch or self._params.is_batch
        )
        params = self._params.replace(
            left_x, 1
        )
        params = params.replace(
            right_x, 2
        )
        return IsoscelesTrapezoid(params.x, updated_m.x)


class IncreasingRightTrapezoid(ConvexPolygon):

    P = 3

    def join(self, x: torch.Tensor):
        m = calc_m_linear_increasing(
            un_x(x), self._params[0], self._params[1], self._m[0]
        )
        m2 = calc_m_flat(un_x(x), self._params[1], self._params[2], self._m[0])

        return torch.max(m, m2)
    
    @property
    def a(self):
        return (
            self._params[2] - self._params[0]
        )

    @property
    def b(self):
        return self._params[2] - self._params[1]

    def _calc_areas(self):
        
        return (
            0.5 * (self.a + self.b) * self._m[0]
        )

    def _calc_mean_cores(self):
        return 0.5 * (self._params[2] + self._params[1])

    def _calc_centroids(self):
        
        d1 = 0.5 * (self._params[1] - self._params[0])
        d2 = self._params[2] - self._params[1]

        return (
            d1 * (2 / 3 * self._params[1] + 1 / 3 * self._params[0]) +
            d2 * (1 / 2 * self._params[2] + 1 / 2 * self._params[1])
        ) / (d1 + d2)

    def scale(self, m: torch.Tensor):
        updated_m = self._intersect_m(m)
        return IncreasingRightTrapezoid(self._params.x, updated_m.x)

    def truncate(self, m: torch.Tensor):
        updated_m = self._intersect_m(m)
        
        x = Params(calc_x_linear_increasing(
            updated_m.x, self._params[0], self._params[1], self._m[0]
        ), True, updated_m.is_batch or self._params.is_batch)
        params = self._params.replace(x, 1)
        return IncreasingRightTrapezoid(params.x, updated_m.x)


class DecreasingRightTrapezoid(ConvexPolygon):

    P = 3

    def join(self, x: torch.Tensor):

        m = calc_m_linear_decreasing(
            un_x(x), self._params[1], self._params[2], self._m[0]
        )
        m2 = calc_m_flat(un_x(x), self._params[0], self._params[1], self._m[0])
    
        return torch.max(m, m2)
    
    @property
    def a(self):
        return (
            self._params[2] - self._params[0]
        )

    @property
    def b(self):
        return self._params[1] - self._params[0]

    def _calc_areas(self):
        
        return (
            0.5 * (self.a + self.b) * self._m[0]
        )

    def _calc_mean_cores(self):
        return 0.5 * (self._params[0] + self._params[1])

    def _calc_centroids(self):
        d1 = self._params[1] - self._params[0]
        d2 = 0.5 * (self._params[2] - self._params[1])
        
        return (
            d1 * (1 / 2 * self._params[1] + 1 / 2 * self._params[0]) +
            d2 * (1 / 3 * self._params[2] + 2 / 3 * self._params[1])
        ) / (d1 + d2)

    def scale(self, m: torch.Tensor):
        updated_m = self._intersect_m(m)
        return DecreasingRightTrapezoid(self._params.x, updated_m.x)

    def truncate(self, m: torch.Tensor):
        updated_m = self._intersect_m(m)
        
        x = Params(calc_x_linear_decreasing(
            updated_m[0], self._params[1], self._params[2], self._m[0]
        ), True, updated_m.is_batch or self._params.is_batch)
        params = self._params.replace(x, 1)
        return DecreasingRightTrapezoid(params.x, updated_m.x)
