import typing
import torch
import torch.nn as nn
import typing
from .base import Set, SetParam, CompositionBase, MistifyLoss
from .utils import get_comp_weight_size, maxmin, minmax, maxprod

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

    def reshape(self, *size: int):
        return BinarySet(
            self.data.reshape(*size), self.is_batch
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

    @property
    def shape(self) -> torch.Size:
        return self._data.shape


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
                self._data.reshape(self._data.size(0), *size_after, -1), True
            )
        return self.__class__(
            self._data.reshape(*size_after, -1), False
        )

    def reshape(self, *size: int):
        return BinarySet(
            self.data.reshape(*size), self.is_batch
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

    @property
    def shape(self) -> torch.Size:
        return self._data.shape

class BinaryComposition(CompositionBase):

    def init_weight(self, in_features: int, out_features: int, in_variables: int = None) -> SetParam:
        return SetParam(BinarySet.positives(get_comp_weight_size(in_features, out_features, in_variables)))

    def forward(self, m: BinarySet):
        return BinarySet(
            maxmin(self.prepare_inputs(m), self.weight.data[None]).round(), True
        )


class TernaryComposition(CompositionBase):

    def init_weight(self, in_features: int, out_features: int, in_variables: int = None) -> SetParam:
        return SetParam(
            TernarySet.positives(get_comp_weight_size(in_features, out_features, in_variables))
        )

    def forward(self, m: BinarySet):
        return TernarySet(
            maxmin(self.prepare_inputs(m), self.weight.data[None]).round(), True
        )


class BinaryThetaLoss(MistifyLoss):

    def __init__(self, lr: float=0.5):
        """initializer

        Args:
            binary (Binary): Composition layer to optimize
            lr (float, optional): learning rate value between 0 and 1. Defaults to 0.5.
        """
        self.lr = lr

    def _calculate_positives(self, x: torch.Tensor, t: torch.Tensor):
        positive = (t == 1)
        return (
            (x[:,:,None] == t[:,None]) & positive[:,None]
        ).type_as(x).sum(dim=0)

    def _calculate_negatives(self, x: torch.Tensor, t: torch.Tensor):
        negative = (t != 1)
        return (
            (x[:,:,None] != t[:,None]) & negative[:,None]
        ).type_as(x).sum(dim=0)
    
    def _update_score(self, score, positives: torch.Tensor, negatives: torch.Tensor):
        cur_score = positives / (negatives + positives)
        cur_score[cur_score.isnan() | cur_score.isinf()] = 0.0
        if score is not None and self.lr is not None:
            return (1 - self.lr) * score + self.lr * cur_score
        return cur_score
    
    def _calculate_maximums(self, x: torch.Tensor, t: torch.Tensor, score: torch.Tensor):
        positive = (t == 1)
        y: torch.Tensor = torch.max(torch.min(x[:,:,None], score[None]), dim=1)[0]
        return ((score[None] == y[:,None]) & positive[:,None]).type_as(x).sum(dim=0)
    
    def _update_weight(self, relation: BinaryComposition, maximums: torch.Tensor, negatives: torch.Tensor):
        cur_weight = maximums / (maximums + negatives)
        cur_weight[cur_weight.isnan() | cur_weight.isinf()] = 0.0
        return cur_weight
        # if self.lr is not None:
        #     relation.weight = nn.parameter.Parameter(
        #         (1 - self.lr) * relation.weight + self.lr * cur_weight
        #     )
        # else:
        #     relation.weight = cur_weight

    def forward(self, relation: BinaryComposition, x: torch.Tensor, t: torch.Tensor, state: dict):
        # TODO: Ensure doesn't need to be mean
        score = state['score']
        with torch.no_grad():
            positives = self._calculate_positives(x, t)
            negatives = self._calculate_negatives(x, t)
            score = self._update_score(score, positives, negatives)
            state['score'] = score
            maximums = self._calculate_maximums(x, t, score)
            target_weight = self._update_weight(relation, maximums, negatives)
        return self._reduction((target_weight - relation.weight).abs())


class BinaryXLoss(MistifyLoss):

    def __init__(self, lr: float=None):
        """initializer

        Args:
            lr (float, optional): learning rate value between 0 and 1. Defaults to 0.5.
        """
        self.lr = lr
    
    def _calculate_positives(self, w: torch.Tensor, t: torch.Tensor):
        positive = (t == 1)
        return (
            (w[None] == t[:,None]) & positive[:,None]
        ).type_as(w).sum(dim=2)

    def _calculate_negatives(self, w: torch.Tensor, t: torch.Tensor):
        negative = (t != 1)
        return (
            (w[None] != t[:,None]) & negative[:,None]
        ).type_as(w).sum(dim=2)
    
    def _calculate_score(self, positives: torch.Tensor, negatives: torch.Tensor):
        cur_score = positives / (negatives + positives)
        cur_score[cur_score.isnan()] == 0.0
        return cur_score
    
    def _calculate_maximums(self, score: torch.Tensor, w: torch.Tensor, t: torch.Tensor):
        positive = (t == 1)
        
        y: torch.Tensor = torch.max(torch.min(w[None,], score[:,:,None]), dim=1)[0]
        return ((score[:,:,None] == y[:,None]) & positive[:,None]).type_as(score).sum(dim=2)
    
    def _update_base_inputs(self, binary: BinaryComposition, maximums: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor):

        if binary.to_complement:
            maximums.view(maximums.size(0), 2, -1)
            negatives.view(negatives.size(0), 2, -1)
            # only use the maximums + negatives
            # is okay for other positives to be 1 since it is an
            # "or" neuron
            base = (
                maximums[:,0] / (maximums[:,0] + negatives[:,0])
            )
            # negatives for the complements must have 1 as the input 
            complements = (
                negatives[:,1] / (positives[:,1] + negatives[:,1])
            )

            return ((0.5 * complements + 0.5 * base))

        cur_inputs = (maximums / (maximums + negatives))
        cur_inputs[cur_inputs.isnan()] = 0.0
        return cur_inputs
    
    def _update_inputs(self, state: dict, base_inputs: torch.Tensor):
        if self.lr is not None:
            base_inputs = (1 - self.lr) * state['base_inputs'] + self.lr * base_inputs        
        return (base_inputs >= 0.5).type_as(base_inputs)

    def forward(self, binary: BinaryComposition, x: torch.Tensor, t: torch.Tensor, state: dict):

        # TODO: Update so it is a "loss"
        w = binary.weight
        with torch.no_grad():
            positives = self._calculate_positives(w, t)
            negatives = self._calculate_negatives(w, t)
            score = self._calculate_score(positives, negatives)
            maximums = self._calculate_maximums(score, w, t)
            base_inputs = self._update_base_inputs(binary, maximums, positives, negatives)
            x_prime = self._update_inputs(state, base_inputs)
        return self._reduction((x_prime - x).abs())
