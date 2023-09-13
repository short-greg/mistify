from mistify import membership, fuzzy, conversion
import typing
from torch import nn
import torch


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
            print(in_terms, in_features)
            self.converter = conversion.SigmoidFuzzyConverter(in_features, in_terms)

            terms = [in_terms, *hidden_terms]
            self.fuzzy_layers = nn.ModuleList()
            for in_i, out_i in zip(terms[:-1], terms[1:]):
                self.fuzzy_layers.append(fuzzy.MaxMin(in_i, out_i, in_variables=in_features))
            self.implication = nn.Sequential(*self.fuzzy_layers)
            self.out_converter = conversion.SigmoidFuzzyConverter(in_features, hidden_terms[-1])
            self.defuzzifier = conversion.SigmoidDefuzzifier(self.out_converter)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.converter.forward(x)
            x = self.implication.forward(x)
            print(x.shape)
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
            print(in_terms, in_features)
            self.converter = conversion.SigmoidFuzzyConverter(in_features, in_terms)

            variables = [in_terms * in_features, *hidden_variables]
            self.fuzzy_layers = nn.ModuleList()
            for in_i, out_i in zip(variables[:-1], variables[1:]):
                self.fuzzy_layers.append(fuzzy.MaxMin(in_i, out_i))
            self.implication = nn.Sequential(*self.fuzzy_layers)
            self._out_features = out_features
            self.out_converter = conversion.SigmoidFuzzyConverter(out_features, hidden_variables[-1] // out_features)
            self.defuzzifier = conversion.SigmoidDefuzzifier(self.out_converter)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            m = self.converter.forward(x)
            m = m.reshape(m.shape[0], -1)
            m = self.implication.forward(m)
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
            print(in_terms, in_features)
            self.converter = conversion.TriangleFuzzyConverter(in_features, in_terms)

            variables = [in_terms * in_features, *hidden_variables]
            self.fuzzy_layers = nn.ModuleList()
            for in_i, out_i in zip(variables[:-1], variables[1:]):
                self.fuzzy_layers.append(fuzzy.MaxMin(in_i, out_i))
            self.implication = nn.Sequential(*self.fuzzy_layers)
            self._out_features = out_features
            self.out_converter = conversion.TriangleFuzzyConverter(out_features, hidden_variables[-1] // out_features)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            m = self.converter.forward(x)
            m = m.reshape(m.shape[0], -1)
            m = self.implication.forward(m)
            m = m.reshape(m.shape[0], self._out_features, -1)
            x = self.out_converter.defuzzify(m)
            return x

    def test_fuzzy_system_with_one_layer_outputs_correct_size(self):
        
        in_features = 4
        in_terms = 3
        hidden_variables = 4
        out_features = 2
        batch_size = 4
        x = torch.randn(batch_size, in_features)
        system = self.BasicTriangularFuzzySystem(in_features, in_terms, [hidden_variables], out_features)
        assert system.forward(x).size() == torch.Size([batch_size, out_features])

    def test_fuzzy_system_with_two_layers_outputs_correct_size(self):
        
        in_features = 4
        in_terms = 3
        hidden_variables = [4, 6]
        out_features = 3
        batch_size = 4
        x = torch.randn(batch_size, in_features)
        system = self.BasicTriangularFuzzySystem(in_features, in_terms, hidden_variables, out_features)
        assert system.forward(x).size() == torch.Size([batch_size, out_features])


class TestBasicSigmoidFuzzySytem2:

    class BasicCrispSystem(nn.Module):

        def __init__(self, in_features: int, in_terms: int, hidden_variables: typing.List[int], out_features: typing.List[int]):
            super().__init__()
            print(in_terms, in_features)
            self.converter = conversion.StepCrispConverter(in_features, in_terms)

            variables = [in_terms * in_features, *hidden_variables]
            self.fuzzy_layers = nn.ModuleList()
            for in_i, out_i in zip(variables[:-1], variables[1:]):
                self.fuzzy_layers.append(fuzzy.MaxMin(in_i, out_i))
            self.implication = nn.Sequential(*self.fuzzy_layers)
            self._out_features = out_features
            self.out_converter = conversion.StepCrispConverter(out_features, hidden_variables[-1] // out_features)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            m = self.converter.forward(x)
            m = m.reshape(m.shape[0], -1)
            m = self.implication.forward(m)
            m = m.reshape(m.shape[0], self._out_features, -1)
            x = self.out_converter.decrispify(m)
            return x

    def test_fuzzy_system_with_one_layer_outputs_correct_size(self):
        
        in_features = 4
        in_terms = 3
        hidden_variables = 4
        out_features = 2
        batch_size = 4
        x = torch.randn(batch_size, in_features)
        system = self.BasicCrispSystem(in_features, in_terms, [hidden_variables], out_features)
        assert system.forward(x).size() == torch.Size([batch_size, out_features])

    def test_fuzzy_system_with_two_layers_outputs_correct_size(self):
        
        in_features = 4
        in_terms = 3
        hidden_variables = [4, 6]
        out_features = 3
        batch_size = 4
        x = torch.randn(batch_size, in_features)
        system = self.BasicCrispSystem(in_features, in_terms, hidden_variables, out_features)
        assert system.forward(x).size() == torch.Size([batch_size, out_features])
