# # 1st party
# from abc import abstractmethod
# from dataclasses import dataclass
# import typing

# # 3rd party
# import torch
# import torch.nn as nn
# import torch.nn.functional

# # local
# from .._base import ValueWeight, Accumulator, MaxValueAcc, Converter
# from .._base._modules import binary_ste


# class Crispifier(nn.Module):

#     @abstractmethod
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         pass


# class Decrispifier(nn.Module):

#     @abstractmethod
#     def imply(self, m: torch.Tensor) -> ValueWeight:
#         pass

#     @abstractmethod
#     def accumulate(self, value_weight: ValueWeight) -> torch.Tensor:
#         pass

#     def forward(self, m: torch.Tensor) -> torch.Tensor:
#         return self.accumulate(self.imply(m))


# class CrispConverter(Converter):
#     """Convert tensor to crisp set
#     """

#     @abstractmethod
#     def crispify(self, x: torch.Tensor) -> torch.Tensor:
#         pass

#     @abstractmethod
#     def imply(self, m: torch.Tensor) -> ValueWeight:
#         pass

#     @abstractmethod
#     def accumulate(self, value_weight: ValueWeight) -> torch.Tensor:
#         pass

#     def decrispify(self, m: torch.Tensor) -> torch.Tensor:
#         return self.accumulate(self.imply(m))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.crispify(x)

#     def reverse(self, m: torch.Tensor) -> torch.Tensor:
#         return self.decrispify(m)

#     def to_crispifier(self) -> 'Crispifier':
#         return ConverterCrispifier(self)

#     def to_decrispifier(self) -> 'Crispifier':
#         return ConverterDecrispifier(self)


# class StepCrispConverter(CrispConverter):

#     def __init__(
#         self, out_variables: int, out_terms: int, 
#         accumulator: Accumulator=None,
#         threshold_f: typing.Callable[[torch.Tensor, typing.Any], torch.Tensor]=None
#     ):
#         super().__init__()

#         self.threshold = nn.parameter.Parameter(
#             torch.randn(out_variables, out_terms)
#         )
#         self._threshold_f = threshold_f
#         self._accumulator = accumulator or MaxValueAcc()

#     def crispify(self, x: torch.Tensor) -> torch.Tensor:
#         if self._threshold_f is not None:
#             return self._threshold_f(x, self.threshold)
#         return (x[:,:,None] >= self.threshold[None]).type_as(x)

#     def imply(self, m: torch.Tensor) -> ValueWeight:
        
#         return ValueWeight(
#             m * self.threshold[None], m
#         )

#     def accumulate(self, value_weight: ValueWeight) -> torch.Tensor:
#         return self._accumulator.forward(value_weight)


# class EmbeddingCrispifier(Crispifier):

#     def __init__(
#         self, out_variables: int, terms: int
#     ):
#         super().__init__()
#         self._terms = terms
#         self._embedding = nn.Embedding(
#             terms, out_variables
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if x.dim() != 2:
#             raise ValueError('Embedding crispifier only works for two dimensional tensors')
#         return binary_ste(self._embedding(x))


# class ConverterDecrispifier(Decrispifier):

#     def __init__(self, crisp_converter: CrispConverter):

#         super().__init__()
#         self.crisp_converter = crisp_converter

#     def imply(self, m: torch.Tensor) -> ValueWeight:
#         return self.crisp_converter.imply(m)

#     def accumulate(self, value_weight: ValueWeight) -> torch.Tensor:
#         return self.crisp_converter.accumulate(value_weight)


# class ConverterCrispifier(Crispifier):

#     def __init__(self, crisp_converter: CrispConverter):

#         super().__init__()
#         self.crisp_converter = crisp_converter

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.crisp_converter.crispify(x)
