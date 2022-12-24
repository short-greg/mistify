import typing
import torch
import torch.nn as nn
import typing
from .base import Set, SetParam, CompositionBase
from .utils import get_comp_weight_size

# Add in TernarySet as a type of crisp set
# with


class BinarySet(Set):

    def differ(self, other: 'BinarySet') -> 'BinarySet':
        return BinarySet((self.data - other._data).clamp(0.0, 1.0))
    
    def unify(self, other: 'BinarySet') -> 'BinarySet':
        return BinarySet(torch.max(self.data, other.data))

    def intersect(self, other: 'BinarySet') -> 'BinarySet':
        return BinarySet(torch.min(self.data, other.data))

    def inclusion(self, other: 'BinarySet') -> 'BinarySet':
        return BinarySet(
            (1 - other.data) + torch.min(self.data, other.data), self._is_batch
        )

    def exclusion(self, other: 'BinarySet') -> 'BinarySet':
        return BinarySet(
            (1 - self.data) + torch.min(self.data, other.data), self._is_batch
        )

    def __sub__(self, other: 'BinarySet') -> 'BinarySet':
        return self.differ(other)

    def __mul__(self, other: 'BinarySet') -> 'BinarySet':
        return self.intersect(other)

    def __add__(self, other: 'BinarySet') -> 'BinarySet':
        return self.unify(other)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def convert_variables(self, *size_after: int):
        
        if self._is_batch:
            return BinarySet(
                self._data.view(self._data.size(0), *size_after, -1), True
            )
        return self.__class__(
            self._data.view(*size_after, -1), False
        )

    @classmethod
    def negatives(cls, *size: int, is_batch: bool=None, dtype=torch.float32, device='cpu'):

        return BinarySet(
            torch.zeros(*size, dtype=dtype, device=device), is_batch
        )
    
    @classmethod
    def positives(cls, *size: int, dtype=torch.float32, is_batch: bool=None, device='cpu'):

        return BinarySet(
            torch.ones(*size, dtype=dtype, device=device), 
            is_batch
        )

    @classmethod
    def rand(cls, *size: int, is_batch: bool=None, dtype=torch.float32, device='cpu'):

        return BinarySet(
            (torch.rand(*size, device=device)).round(), 
            is_batch
        )

    def transpose(self, first_dim, second_dim) -> 'BinarySet':
        assert self._value_size is not None
        return BinarySet(self._data.transpose(first_dim, second_dim), self._is_batch)


class TernarySet(Set):

    def differ(self, other: 'TernarySet') -> 'TernarySet':
        return TernarySet((self.data - other._data).clamp(-1.0, 1.0))
    
    def unify(self, other: 'TernarySet') -> 'TernarySet':
        return TernarySet(torch.max(self.data, other.data))

    def intersect(self, other: 'TernarySet') -> 'TernarySet':
        return TernarySet(torch.min(self.data, other.data))

    def inclusion(self, other: 'TernarySet') -> 'TernarySet':
        return TernarySet(
            (1 - other.data) + torch.min(self.data, other.data), self._is_batch
        )

    def exclusion(self, other: 'TernarySet') -> 'TernarySet':
        return TernarySet(
            (1 - self.data) + torch.min(self.data, other.data), self._is_batch
        )

    def __sub__(self, other: 'TernarySet') -> 'TernarySet':
        return self.differ(other)

    def __mul__(self, other: 'TernarySet') -> 'TernarySet':
        return self.intersect(other)

    def __add__(self, other: 'TernarySet') -> 'TernarySet':
        return self.unify(other)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def convert_variables(self, *size_after: int):
        
        if self._is_batch:
            return TernarySet(
                self._data.view(self._data.size(0), *size_after, -1), True
            )
        return self.__class__(
            self._data.view(*size_after, -1), False
        )

    @classmethod
    def negatives(cls, *size: int, is_batch: bool=None, dtype=torch.float32, device='cpu'):

        return TernarySet(
            -torch.ones(*size, dtype=dtype, device=device), is_batch
        )
    
    @classmethod
    def positives(cls, *size: int, dtype=torch.float32, is_batch: bool=None, device='cpu'):

        return TernarySet(
            torch.ones(*size, dtype=dtype, device=device), 
            is_batch
        )

    @classmethod
    def unknowns(cls, *size: int, dtype=torch.float32, is_batch: bool=None, device='cpu'):

        return TernarySet(
            torch.zeros(*size, dtype=dtype, device=device), 
            is_batch
        )

    @classmethod
    def rand(cls, *size: int, is_batch: bool=None, dtype=torch.float32, device='cpu'):

        return TernarySet(
            ((torch.rand(*size, device=device)) * 2 - 1).round(), 
            is_batch
        )

    def transpose(self, first_dim, second_dim) -> 'TernarySet':
        assert self._value_size is not None
        return TernarySet(self._data.transpose(first_dim, second_dim), self._is_batch)


class BinaryComposition(CompositionBase):

    def init_weight(self, in_features: int, out_features: int, in_variables: int = None) -> SetParam:
        return SetParam(BinarySet.ones(get_comp_weight_size(in_features, out_features, in_variables)))

    def forward(self, m: BinarySet):
        return BinarySet(torch.max(
            torch.min(self.prepare_inputs(m), self.weight.data[None]), dim=-2
        )[0], True)


class TernaryComposition(CompositionBase):

    def init_weight(self, in_features: int, out_features: int, in_variables: int = None) -> SetParam:
        return SetParam(TernarySet.ones(get_comp_weight_size(in_features, out_features, in_variables)))

    def forward(self, m: BinarySet):
        return TernarySet(torch.max(
            torch.min(self.prepare_inputs(m), self.weight.data[None]), dim=-2
        )[0], True)



# from .base_machinery import ThetaOptim, ThetaUpdate, XOptim, XUpdate
# import torch.nn as nn
# import torch
# from .assessment import ScalarAssessment, BatchAssessment
# import typing
# from .base_machinery import Machine, Result, T0


# class BinaryRelation(nn.Module):

#     def __init__(self, in_features: int, out_features: int, complement_inputs: bool=False):

#         super().__init__()
#         self._in_features = in_features
#         self._out_features = out_features
#         self._complement_inputs = complement_inputs
#         if complement_inputs:
#             in_features = in_features * 2
#         # store weights as values between 0 and 1
#         self.weight = nn.parameter.Parameter(
#             torch.rand(in_features, self._out_features)
#         )
    
#     @property
#     def to_complement(self) -> bool:
#         return self._complement_inputs
    
#     def forward(self, m: torch.Tensor):

#         # assume inputs are binary
#         # binarize the weights
#         if self._complement_inputs:
#             m = torch.cat([m, 1 - m], dim=1)
#         weights = (self.weight > 0.5).type_as(self.weight)
#         return torch.max(
#             torch.min(m[:,:,None], weights[None]), dim=1
#         )[0]


# class ToBinary(nn.Module):

#     def __init__(self, in_features: int, out_categories: int, same: bool=False):
#         super().__init__()
#         if same:
#             self.threshold = nn.parameter.Parameter(
#                 torch.rand(1, 1, out_categories)
#             )
#         else:
#             self.threshold = nn.parameter.Parameter(
#                 torch.rand(1, in_features, out_categories)
#             )
#         self.same = same

#     def update_threshold(self, threshold: torch.Tensor):

#         assert threshold.size() == self.threshold.size()
#         self.threshold = nn.parameter.Parameter(
#             threshold
#         )

#     def forward(self, x: torch.Tensor):
#         return ((x[:,:,None] - self.threshold) >= 0.0).type_as(x)


# class BinaryThetaOptim(ThetaOptim):

#     def __init__(self, lr: float=0.5):
#         """initializer

#         Args:
#             binary (Binary): Composition layer to optimize
#             lr (float, optional): learning rate value between 0 and 1. Defaults to 0.5.
#         """
#         self.lr = lr
#         # only works if weight between 0 and 1
#         # self._binary.weight = nn.parameter.Parameter(self._binary.weight.clamp(0.0, 1.0))

#     def _calculate_positives(self, x: torch.Tensor, t: torch.Tensor):
#         positive = (t == 1)
#         return (
#             (x[:,:,None] == t[:,None]) & positive[:,None]
#         ).type_as(x).sum(dim=0)

#     def _calculate_negatives(self, x: torch.Tensor, t: torch.Tensor):
#         negative = (t != 1)
#         return (
#             (x[:,:,None] != t[:,None]) & negative[:,None]
#         ).type_as(x).sum(dim=0)
    
#     def _update_score(self, score, positives: torch.Tensor, negatives: torch.Tensor):
#         cur_score = positives / (negatives + positives)
#         cur_score[cur_score.isnan() | cur_score.isinf()] = 0.0
#         if score is not None and self.lr is not None:
#             return (1 - self.lr) * score + self.lr * cur_score
#         return cur_score
    
#     def _calculate_maximums(self, x: torch.Tensor, t: torch.Tensor, score: torch.Tensor):
#         positive = (t == 1)
#         y: torch.Tensor = torch.max(torch.min(x[:,:,None], score[None]), dim=1)[0]
#         return ((score[None] == y[:,None]) & positive[:,None]).type_as(x).sum(dim=0)
    
#     def _update_weight(self, binary_relation: BinaryRelation, maximums: torch.Tensor, negatives: torch.Tensor):
#         cur_weight = maximums / (maximums + negatives)
#         cur_weight[cur_weight.isnan() | cur_weight.isinf()] = 0.0
#         if self.lr is not None:
#             binary_relation.weight = nn.parameter.Parameter(
#                 (1 - self.lr) * binary_relation.weight + self.lr * cur_weight
#             )
#         else:
#             binary_relation.weight = cur_weight


#     def optimize(self, machine: Machine) -> ThetaUpdate:
#         binary_relation = machine.module
#         assert isinstance(binary_relation, BinaryRelation)
#         return ThetaUpdate(self, machine, binary_relation=binary_relation, score=None)
    
#     def spawn(self, lr: float=None) -> 'ThetaOptim':
#         return BinaryThetaOptim(lr=self.lr)

#     def step(self, theta_update: ThetaUpdate, x: torch.Tensor, out: 'XUpdate', result: Result) -> typing.Tuple[T0, BatchAssessment]:
#         # TODO: Ensure doesn't need to be mean
#         binary_relation = theta_update['binary_relation']
#         score = theta_update['score']
    
#         positives = self._calculate_positives(x, out.x)
#         negatives = self._calculate_negatives(x, out.x)
#         score = self._update_score(score, positives, negatives)
#         theta_update['score'] = score
#         maximums = self._calculate_maximums(x, out.x, score)
#         self._update_weight(binary_relation, maximums, negatives)
#         assessment = theta_update.machine.assess(x, out.x)
#         # assessment2 = objective.assess(x, t)
#         # print('Binary Theta After /Before', assessment2.mean().regularized, assessment.mean().regularized)
#         return assessment


# class BinaryXOptim(XOptim):

#     def __init__(self, lr: float=None):
#         """initializer

#         Args:
#             lr (float, optional): learning rate value between 0 and 1. Defaults to 0.5.
#         """
#         self.lr = lr

#     def spawn(self) -> 'XOptim':
#         return super().spawn()
    
#     def _calculate_positives(self, w: torch.Tensor, t: torch.Tensor):
#         positive = (t == 1)
#         return (
#             (w[None] == t[:,None]) & positive[:,None]
#         ).type_as(w).sum(dim=2)

#     def _calculate_negatives(self, w: torch.Tensor, t: torch.Tensor):
#         negative = (t != 1)
#         return (
#             (w[None] != t[:,None]) & negative[:,None]
#         ).type_as(w).sum(dim=2)
    
#     def _calculate_score(self, positives: torch.Tensor, negatives: torch.Tensor):
#         cur_score = positives / (negatives + positives)
#         cur_score[cur_score.isnan()] == 0.0
#         return cur_score
    
#     def _calculate_maximums(self, score: torch.Tensor, w: torch.Tensor, t: torch.Tensor):
#         positive = (t == 1)
        
#         y: torch.Tensor = torch.max(torch.min(w[None,], score[:,:,None]), dim=1)[0]
#         return ((score[:,:,None] == y[:,None]) & positive[:,None]).type_as(score).sum(dim=2)
    
#     def _update_base_inputs(self, binary: BinaryRelation, maximums: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor):

#         if binary.to_complement:
#             maximums.view(maximums.size(0), 2, -1)
#             negatives.view(negatives.size(0), 2, -1)
#             # only use the maximums + negatives
#             # is okay for other positives to be 1 since it is an
#             # "or" neuron
#             base = (
#                 maximums[:,0] / (maximums[:,0] + negatives[:,0])
#             )
#             # negatives for the complements must have 1 as the input 
#             complements = (
#                 negatives[:,1] / (positives[:,1] + negatives[:,1])
#             )

#             return ((0.5 * complements + 0.5 * base))

#         cur_inputs = (maximums / (maximums + negatives))
#         cur_inputs[cur_inputs.isnan()] = 0.0
#         return cur_inputs
    
#     def _update_inputs(self, x_update: XUpdate, base_inputs: torch.Tensor):
#         if self.lr is not None:
#             base_inputs = (1 - self.lr) * x_update['base_inputs'] + self.lr * base_inputs
        
#         return (base_inputs >= 0.5).type_as(base_inputs)

#     def optimize(self, machine: Machine, x: torch.Tensor, t: torch.Tensor) ->    XUpdate:
#         binary_relation = machine.module
#         assert isinstance(binary_relation, BinaryRelation)
#         return XUpdate(machine, self, x=x, t=t, binary_relation=binary_relation, base_inputs=None)

#     def step(self, x_update: XUpdate, x_prime: torch.Tensor, indices: typing.Optional[torch.LongTensor] = None) -> typing.Tuple[torch.Tensor, BatchAssessment]:

#         # t = self._binary.activation.reverse(t)
#         binary: BinaryRelation = x_update['binary_relation']
#         # cur_inputs = x_update['cur_inputs']
#         w = binary.weight
#         t = x_update.t
#         positives = self._calculate_positives(w, t)
#         negatives = self._calculate_negatives(w, t)
#         score = self._calculate_score(positives, negatives)
#         maximums = self._calculate_maximums(score, w, t)
#         base_inputs = self._update_base_inputs(binary, maximums, positives, negatives)
#         x_update['base_inputs'] = base_inputs
#         x_prime = self._update_inputs(x_update, base_inputs)
#         assessment2, y, _ = x_update.assess(x_prime, full_output=True)

#         return x_prime, assessment2


# class ToBinaryThetaOptim(ThetaOptim):

#     def __init__(self, lr: float=1e-2):
#         """initialzier

#         Args:
#             linear (nn.Linear): Linear layer to optimize
#             act_inverse (Reversible): The invertable activation of the layer
#         """
#         self.lr = lr

#     def optimize(self, machine: Machine) -> ThetaUpdate:
#         to_binary = machine.module
#         assert isinstance(to_binary, ToBinary)
#         return ThetaUpdate(self, machine, to_binary=to_binary)

#     def spawn(self) -> 'ThetaOptim':
#         return ToBinaryThetaOptim(self.lr)

#     def step(self, theta_update: ThetaUpdate, x: torch.Tensor, out: 'XUpdate', result: Result) -> BatchAssessment:

#         # assessment, y, result = get_y_and_assessment(objective, x, t, result)
#         to_binary: ToBinary = theta_update['to_binary']
#         assessment, y, result = theta_update.machine.assess(x, out.x, True, True)
#         change = (y != out.x).type_as(y)
#         threshold = torch.clone(to_binary.threshold)

#         dthreshold = (change * (x[:,:,None] - threshold)).sum(dim=0) / change.sum(dim=0)
#         dthreshold[dthreshold.isnan()] = 0.0
#         #  x_prime = x + self._lr * ((t - y) @ weight.T)
#         # aggregate
#         if to_binary.same:
#             threshold += self.lr * dthreshold[None].mean(dim=1, keepdim=True)
#         else:
#             threshold += self.lr * dthreshold[None]

#         to_binary.threshold = nn.parameter.Parameter(threshold)
#         return assessment


# class ToBinaryXOptim(XOptim):

#     def __init__(self, lr: float=1e-2):
#         """initialzier

#         Args:
#             linear (nn.Linear): Linear layer to optimize
#             act_inverse (Reversible): The invertable activation of the layer
#         """
#         self.lr = lr

#     def spawn(self) -> 'XOptim':
#         return ToBinaryXOptim(self.lr)

#     def optimize(self, machine: Machine, x: torch.Tensor, t: torch.Tensor) -> XUpdate:
#         to_binary = machine.module
#         assert isinstance(to_binary, ToBinary)
#         return XUpdate(machine, self, x=x, t=t, to_binary=to_binary)

#     def step(self, x_update: XUpdate, x_prime: torch.Tensor, indices: typing.Optional[torch.LongTensor]=None) -> typing.Tuple[torch.Tensor, BatchAssessment]:
        
#         # TODO: simplify this.... This code is duplicated quite a bit
#         # assessment, y, result = get_y_and_assessment(objective, x, t, result)
#         to_binary = x_update['to_binary']
#         assessment, y, result = x_update.assess(full_output=True)
#         # TODO: check calculation of assessment
#         change = (y != x_update.t).type_as(y)
#         # change = self._to_binary.activation.reverse(change)
#         threshold = torch.clone(to_binary.threshold)
#         dx = (change * (threshold - x_update.x[:,:,None])).sum(dim=2) / change.sum(dim=2)
#         dx[dx.isnan()] = 0.0
#         x_prime = x_update.x + self.lr * dx
#         return x_prime, assessment

