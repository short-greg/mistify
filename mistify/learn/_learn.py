from abc import abstractmethod
import typing

import torch
from zenkai.kaku import IO, State

from zenkai.kikai import GradLearner
from zenkai import OptimFactory, ThLoss
from ..infer import Or, And, WEIGHT_FACTORY
from ._fuzzy_assess import MaxMinLoss3, MinMaxLoss3, NeuronMSELoss


class PostFit(object):

    @abstractmethod
    def fit_postprocessor(self, X: IO, t: IO=None):
        pass


class PreFit(object):
    
    @abstractmethod
    def fit_preprocessor(self, X: IO, t: IO=None):
        pass


class OrLearner(GradLearner):

    def __init__(
        self, in_features: int, out_features: int, 
        optim_factory: OptimFactory=None, n_terms: int=None, 
        f: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]="max_min",
        wf: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]="clamp",
        reduction: str='mean', x_lr: float=None, loss_weight: float=None,
        not_chosen_x_weight: float=0.01, not_chosen_theta_weight: float=0.01,
        weight_update_f=None
    ):
        criterion = ThLoss('MSELoss', reduction=reduction, weight=loss_weight)
        or_ = Or(
            in_features, out_features, n_terms, f, wf
        )
        # learn_criterion = MaxMinLoss3(
        #     or_, reduction=reduction, not_chosen_theta_weight=0.01,
        #     not_chosen_x_weight=0.01
        # )
        learn_criterion = NeuronMSELoss(
            or_, reduction, not_chosen_x_weight, not_chosen_theta_weight
        )
        self._weight_update_f = WEIGHT_FACTORY.factory(weight_update_f)
        super().__init__(
            or_, criterion, optim_factory, False, reduction, x_lr, learn_criterion
        )
    
    def step_x(self, x: IO, t: IO, state: State) -> IO:
        x_prime = super().step_x(x, t, state)
        return x_prime

    def step(self, x: IO, t: IO, state: State):
        super().step(x, t, state)
        self._net.weight.data = self._weight_update_f(self._net.weight.data).detach()


class AndLearner(GradLearner):

    def __init__(
        self, in_features: int, out_features: int, 
        optim_factory: OptimFactory=None, n_terms: int=None, 
        f: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]="min_max",
        wf: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]="clamp",
        reduction: str='mean', x_lr: float=None, loss_weight: float=None,
        not_chosen_x_weight: float=0.01, not_chosen_theta_weight: float=0.01,
        weight_update_f=None
    ):
        criterion = ThLoss('MSELoss', reduction=reduction, weight=loss_weight)
        and_ = And(
            in_features, out_features, n_terms, f, wf
        )
        # learn_criterion = MinMaxLoss3(
        #     and_, reduction=reduction, not_chosen_theta_weight=0.01,
        #     not_chosen_x_weight=0.01
        # )
        learn_criterion = NeuronMSELoss(
            and_, reduction, not_chosen_x_weight, not_chosen_theta_weight
        )
        self._weight_update_f = WEIGHT_FACTORY.factory(weight_update_f)
        super().__init__(
            and_, criterion, optim_factory, False, reduction, x_lr, learn_criterion
        )

    def step_x(self, x: IO, t: IO, state: State) -> IO:
        x_prime = super().step_x(x, t, state)
        return x_prime
    
    def step(self, x: IO, t: IO, state: State):
        super().step(x, t, state)
        self._net.weight.data = self._weight_update_f(self._net.weight.data).detach()