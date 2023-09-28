
# class BinaryWeightLoss(nn.Module):

#     def __init__(self, to_binary: conversion.StepCrispConverter):
#         """initialzier

#         Args:
#             linear (nn.Linear): Linear layer to optimize
#             act_inverse (Reversible): The invertable activation of the layer
#         """
#         self._to_binary = to_binary

#     def step(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor):

#         # assessment, y, result = get_y_and_assessment(objective, x, t, result)
#         # y = to_binary.forward(x)
#         change = (y != t).type_as(y)
#         if self._to_binary.same:
#             loss = (self._to_binary.weight[None,None,:] * change) ** 2
#         else:
#             loss = (self._to_binary.weight[None,:,:] * change) ** 2

#         # TODO: Reduce the loss
#         return loss


# class BinaryXLoss(nn.Module):

#     def __init__(self, to_binary: conversion.StepCrispConverter):
#         """initialzier

#         Args:
#             linear (nn.Linear): Linear layer to optimize
#             act_inverse (Reversible): The invertable activation of the layer
#         """
#         self._to_binary = to_binary

#     def step(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor):

#         # assessment, y, result = get_y_and_assessment(objective, x, t, result)
#         # y = to_binary.forward(x)
#         change = (y != t).type_as(y)
#         loss = (x[:,:,None] * change) ** 2

#         # TODO: Reduce the loss
#         return loss


# class BinaryXLoss(MistifyLoss):

#     def __init__(self, lr: float=None):
#         """initializer

#         Args:
#             lr (float, optional): learning rate value between 0 and 1. Defaults to 0.5.
#         """
#         self.lr = lr
    
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
    
#     def _update_base_inputs(self, binary: BinaryComposition, maximums: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor):

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
    
#     def _update_inputs(self, state: dict, base_inputs: torch.Tensor):
#         if self.lr is not None:
#             base_inputs = (1 - self.lr) * state['base_inputs'] + self.lr * base_inputs        
#         return (base_inputs >= 0.5).type_as(base_inputs)

#     def forward(self, binary: BinaryComposition, x: torch.Tensor, t: torch.Tensor, state: dict):

#         # TODO: Update so it is a "loss"
#         w = binary.weight
#         with torch.no_grad():
#             positives = self._calculate_positives(w, t)
#             negatives = self._calculate_negatives(w, t)
#             score = self._calculate_score(positives, negatives)
#             maximums = self._calculate_maximums(score, w, t)
#             base_inputs = self._update_base_inputs(binary, maximums, positives, negatives)
#             x_prime = self._update_inputs(state, base_inputs)
#         return self._reduction((x_prime - x).abs())


# class BinaryThetaLoss(MistifyLoss):

#     def __init__(self, lr: float=0.5):
#         """initializer

#         Args:
#             binary (Binary): Composition layer to optimize
#             lr (float, optional): learning rate value between 0 and 1. Defaults to 0.5.
#         """
#         self.lr = lr

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
    
#     def _update_weight(self, relation: BinaryComposition, maximums: torch.Tensor, negatives: torch.Tensor):
#         cur_weight = maximums / (maximums + negatives)
#         cur_weight[cur_weight.isnan() | cur_weight.isinf()] = 0.0
#         return cur_weight
#         # if self.lr is not None:
#         #     relation.weight = nn.parameter.Parameter(
#         #         (1 - self.lr) * relation.weight + self.lr * cur_weight
#         #     )
#         # else:
#         #     relation.weight = cur_weight

#     def forward(self, relation: BinaryComposition, x: torch.Tensor, t: torch.Tensor, state: dict):
#         # TODO: Ensure doesn't need to be mean
#         score = state['score']
#         with torch.no_grad():
#             positives = self._calculate_positives(x, t)
#             negatives = self._calculate_negatives(x, t)
#             score = self._update_score(score, positives, negatives)
#             state['score'] = score
#             maximums = self._calculate_maximums(x, t, score)
#             target_weight = self._update_weight(relation, maximums, negatives)
#         return self._reduction((target_weight - relation.weight).abs())



