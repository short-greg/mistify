
# TODO: Add in functionality to see if it should be -1
import torch

from mistify._base import ValueWeight
from ..boolean import Crispifier, CrispConverter
from .._base import functional


class SignedCrispifier(Crispifier):

    def __init__(self, boolean_crispifier: Crispifier):
        super().__init__()
        self._crispifier = boolean_crispifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return functional.to_signed(self._crispifier(x))


class SignedConverter(CrispConverter):

    def __init__(self, boolean_converter: CrispConverter):

        super().__init__()
        self._converter = boolean_converter

    def crispify(self, x: torch.Tensor) -> torch.Tensor:
        return functional.to_signed(super().crispify(x))

    def accumulate(self, value_weight: ValueWeight) -> torch.Tensor:
        return self._converter.accumulate(value_weight)
    
    def imply(self, m: torch.Tensor) -> ValueWeight:
        m = functional.to_binary(m)
        return self._converter.imply(m)
