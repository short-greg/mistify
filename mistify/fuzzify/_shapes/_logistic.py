# 1st party

# 3rd party
import torch

# local
from ._base import ShapeParams, Concave
from ...utils import unsqueeze, check_contains
from ._utils import calc_dx_logistic, calc_area_logistic_one_side, calc_m_logistic, calc_x_logistic


intersect = torch.min


class Logistic(Concave):

    def __init__(
        self, biases: ShapeParams, scales: ShapeParams, m: torch.Tensor=None
    ):
        self._biases = biases
        self._scales = scales

        self._m = m if m is not None else torch.ones(
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
            params.sub((0, 1)), 
            params.sub((1, 2)), m
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
            truncated_m = torch.ones(*self._m.size(), device=self._m.device)

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
            truncated_m = torch.ones(self._m.size(), device=self._m.device)

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
