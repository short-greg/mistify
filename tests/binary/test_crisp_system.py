from mistify import binary
import torch
from torch import nn
from mistify import fuzzy
import typing


# TODO: Fix bugs

class TestBasicCrispSystem2:

    class BasicCrispSystem(nn.Module):

        def __init__(self, in_features: int, in_terms: int, hidden_variables: typing.List[int], out_features: typing.List[int]):
            super().__init__()
            print(in_terms, in_features)
            self.converter = binary.StepCrispConverter(in_features, in_terms)

            variables = [in_terms * in_features, *hidden_variables]
            self.fuzzy_layers = nn.ModuleList()
            for in_i, out_i in zip(variables[:-1], variables[1:]):
                self.fuzzy_layers.append(fuzzy.FuzzyOr(in_i, out_i))
            self.implication = nn.Sequential(*self.fuzzy_layers)
            self._out_features = out_features
            self.out_converter = binary.StepCrispConverter(out_features, hidden_variables[-1] // out_features)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            m = self.converter.forward(x)
            m = m.reshape(m.shape[0], -1)
            m = self.implication.forward(m)
            m = m.reshape(m.shape[0], self._out_features, -1)
            x = self.out_converter.decrispify(m)
            return x

    def test_crisp_system_with_one_layer_outputs_correct_size(self):
        
        in_features = 4
        in_terms = 3
        hidden_variables = 4
        out_features = 2
        batch_size = 4
        x = torch.randn(batch_size, in_features)
        system = self.BasicCrispSystem(in_features, in_terms, [hidden_variables], out_features)
        assert system.forward(x).size() == torch.Size([batch_size, out_features])

    def test_crisp_system_with_two_layers_outputs_correct_size(self):
        
        in_features = 4
        in_terms = 3
        hidden_variables = [4, 6]
        out_features = 3
        batch_size = 4
        x = torch.randn(batch_size, in_features)
        system = self.BasicCrispSystem(in_features, in_terms, hidden_variables, out_features)
        assert system.forward(x).size() == torch.Size([batch_size, out_features])
