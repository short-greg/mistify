import typing
from torch import nn
import torch
from mistify import fuzzify
from mistify.infer import _neurons as neurons


class TestBasicSigmoidFuzzySytem:

    class BasicSigmoidFuzzySystem(nn.Module):

        def __init__(self, in_features: int, in_terms: int, hidden_terms: typing.List[int]):
            """

            Args:
                in_features (int): 
                in_terms (int): 
                hidden_terms (typing.List[int]): 
            """
            super().__init__()
            self.converter = fuzzify.SigmoidFuzzyConverter.from_linspace(in_terms)

            terms = [in_terms, *hidden_terms]
            self.fuzzy_layers = nn.ModuleList()
            for in_i, out_i in zip(terms[:-1], terms[1:]):
                self.fuzzy_layers.append(neurons.Or(in_i, out_i, n_terms=in_features))
            self.hypothesis = nn.Sequential(*self.fuzzy_layers)
            self.out_converter = fuzzify.SigmoidFuzzyConverter.from_linspace(hidden_terms[-1])
            self.defuzzifier = fuzzify.ConverterDefuzzifier(self.out_converter)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.converter.forward(x)
            x = self.hypothesis.forward(x)
            x = self.defuzzifier.forward(x)
            return x

    def test_fuzzy_system_with_one_layer_outputs_correct_size(self):
        
        in_features = 4
        in_terms = 3
        hidden_terms = 2
        batch_size = 4
        x = torch.randn(batch_size, in_features)
        system = self.BasicSigmoidFuzzySystem(in_features, in_terms, [hidden_terms])
        assert system.forward(x).size() == torch.Size([batch_size, in_features])

    def test_fuzzy_system_with_two_layers_outputs_correct_size(self):
        
        in_features = 4
        in_terms = 3
        hidden_terms = [2, 4]
        batch_size = 4
        x = torch.randn(batch_size, in_features)
        system = self.BasicSigmoidFuzzySystem(in_features, in_terms, hidden_terms)
        assert system.forward(x).size() == torch.Size([batch_size, in_features])


class TestBasicSigmoidFuzzySytem2:

    class BasicSigmoidFuzzySystem2(nn.Module):

        def __init__(self, in_features: int, in_terms: int, hidden_variables: typing.List[int], out_features: typing.List[int]):
            super().__init__()
            self.converter = fuzzify.SigmoidFuzzyConverter.from_linspace(in_terms)

            variables = [in_terms * in_features, *hidden_variables]
            self.fuzzy_layers = nn.ModuleList()
            for in_i, out_i in zip(variables[:-1], variables[1:]):
                self.fuzzy_layers.append(neurons.Or(in_i, out_i))
            self.hypothesis = nn.Sequential(*self.fuzzy_layers)
            self._out_features = out_features
            self.out_converter = fuzzify.SigmoidFuzzyConverter.from_linspace(hidden_variables[-1] // out_features)
            self.defuzzifier = fuzzify.ConverterDefuzzifier(self.out_converter)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            m = self.converter.forward(x)
            m = m.reshape(m.shape[0], -1)
            m = self.hypothesis.forward(m)
            m = m.reshape(m.shape[0], self._out_features, -1)
            x = self.defuzzifier.forward(m)
            return x

    def test_fuzzy_system_with_one_layer_outputs_correct_size(self):
        
        in_features = 4
        in_terms = 3
        hidden_variables = 4
        out_features = 2
        batch_size = 4
        x = torch.randn(batch_size, in_features)
        system = self.BasicSigmoidFuzzySystem2(in_features, in_terms, [hidden_variables], out_features)
        assert system.forward(x).size() == torch.Size([batch_size, out_features])

    def test_fuzzy_system_with_two_layers_outputs_correct_size(self):
        
        in_features = 4
        in_terms = 3
        hidden_variables = [4, 6]
        out_features = 3
        batch_size = 4
        x = torch.randn(batch_size, in_features)
        system = self.BasicSigmoidFuzzySystem2(in_features, in_terms, hidden_variables, out_features)
        assert system.forward(x).size() == torch.Size([batch_size, out_features])

class TestBasicTriangularFuzzySytem:

    class BasicTriangularFuzzySystem(nn.Module):

        def __init__(self, in_features: int, in_terms: int, hidden_variables: typing.List[int], out_features: typing.List[int]):
            super().__init__()
            self.converter = fuzzify.TriangleFuzzyConverter(in_features, in_terms)

            variables = [in_terms * in_features, *hidden_variables]
            self.fuzzy_layers = nn.ModuleList()
            for in_i, out_i in zip(variables[:-1], variables[1:]):
                self.fuzzy_layers.append(neurons.Or(in_i, out_i))
            self.hypothesis = nn.Sequential(*self.fuzzy_layers)
            self._out_features = out_features
            self.out_converter = fuzzify.TriangleFuzzyConverter(out_features, hidden_variables[-1] // out_features)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            m = self.converter.forward(x)
            m = m.reshape(m.shape[0], -1)
            m = self.hypothesis.forward(m)
            m = m.reshape(m.shape[0], self._out_features, -1)
            x = self.out_converter.defuzzify(m)
            return x

    # def test_fuzzy_system_with_one_layer_outputs_correct_size(self):
        
    #     in_features = 4
    #     in_terms = 3
    #     hidden_variables = 4
    #     out_features = 2
    #     batch_size = 4
    #     x = torch.randn(batch_size, in_features)
    #     system = self.BasicTriangularFuzzySystem(in_features, in_terms, [hidden_variables], out_features)
    #     assert system.forward(x).size() == torch.Size([batch_size, out_features])

    # def test_fuzzy_system_with_two_layers_outputs_correct_size(self):
        
    #     in_features = 4
    #     in_terms = 3
    #     hidden_variables = [4, 6]
    #     out_features = 3
    #     batch_size = 4
    #     x = torch.randn(batch_size, in_features)
    #     system = self.BasicTriangularFuzzySystem(in_features, in_terms, hidden_variables, out_features)
    #     assert system.forward(x).size() == torch.Size([batch_size, out_features])


from mistify import fuzzify
import torch
from torch import nn
from mistify.infer import fuzzy
import typing


class TestBasicCrispSystem2:

    class BasicCrispSystem(nn.Module):

        def __init__(self, in_features: int, in_terms: int, hidden_variables: typing.List[int], out_features: typing.List[int]):
            super().__init__()
            self.converter = fuzzify.StepFuzzyConverter.from_linspace(in_terms)

            variables = [in_terms * in_features, *hidden_variables]
            self.fuzzy_layers = nn.ModuleList()
            for in_i, out_i in zip(variables[:-1], variables[1:]):
                self.fuzzy_layers.append(neurons.Or(in_i, out_i))
            self.hypothesis = nn.Sequential(*self.fuzzy_layers)
            self._out_features = out_features
            self.out_converter = fuzzify.StepFuzzyConverter.from_linspace(hidden_variables[-1] // out_features)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            m = self.converter.forward(x)
            m = m.reshape(m.shape[0], -1)
            m = self.hypothesis.forward(m)
            m = m.reshape(m.shape[0], self._out_features, -1)
            x = self.out_converter.defuzzify(m)
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
