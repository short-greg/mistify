from abc import abstractmethod
import typing

import torch

from zenkai.kikai import GradLearner
from zenkai import OptimFactory, ThLoss
from ..infer import Or, And
from ._fuzzy_assess import MaxMinLoss3, MinMaxLoss3, NeuronMSELoss


class PostFit(object):

    @abstractmethod
    def fit_postprocessor(self):
        pass


class PreFit(object):
    
    @abstractmethod
    def fit_preprocessor(self):
        pass


class OrLearner(GradLearner):

    def __init__(
        self, in_features: int, out_features: int, 
        optim_factory: OptimFactory=None, n_terms: int=None, 
        f: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]="max_min",
        wf: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]="clamp",
        reduction: str='mean', x_lr: float=None,
        not_chosen_x_weight: float=0.01, not_chosen_theta_weight: float=0.01,
    ):
        criterion = ThLoss('MSELoss', reduction=reduction)
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
        super().__init__(
            or_, criterion, optim_factory, False, reduction, x_lr, learn_criterion
        )


class AndLearner(GradLearner):

    def __init__(
        self, in_features: int, out_features: int, 
        optim_factory: OptimFactory=None, n_terms: int=None, 
        f: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]="min_max",
        wf: typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]="clamp",
        reduction: str='mean', x_lr: float=None,
        not_chosen_x_weight: float=0.01, not_chosen_theta_weight: float=0.01,
    ):
        criterion = ThLoss('MSELoss', reduction=reduction)
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
        super().__init__(
            and_, criterion, optim_factory, False, reduction, x_lr, learn_criterion
        )
