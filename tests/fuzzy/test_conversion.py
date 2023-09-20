import pytest
from mistify import fuzzy
import torch



class TestSigmoidFuzzyConverter:

    def test_fuzzify_converts_to_binary_set(self):

        converter = fuzzy.SigmoidFuzzyConverter(2, 3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert isinstance(fuzzy_set, torch.Tensor)

    def test_fuzzify_converts_to_fuzzy_set_with_correct_size(self):

        converter = fuzzy.SigmoidFuzzyConverter(2, 3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert fuzzy_set.data.size() == torch.Size([3, 2, 3])

    def test_fuzzify_converts_to_binary_set_with_correct_size_when_using_same(self):

        converter = fuzzy.SigmoidFuzzyConverter(1, 3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert fuzzy_set.data.size() == torch.Size([3, 2, 3])

    def test_imply_returns_value_weight_with_correct_size(self):

        converter = fuzzy.SigmoidFuzzyConverter(1, 3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        value_weight = converter.imply(fuzzy_set)
        assert value_weight.weight.size() == torch.Size([3, 2, 3])
        assert value_weight.value.size() == torch.Size([3, 2, 3])

    def test_accumulate_returns_tensor_of_correct_size(self):

        converter = fuzzy.SigmoidFuzzyConverter(1, 3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        value_weight = converter.imply(fuzzy_set)
        result = converter.accumulate(value_weight)
        assert result.size() == torch.Size([3, 2])


class TestSigmoidDefuzzifier:

    def test_defuzzifier_defuzzifies(self):

        converter = fuzzy.SigmoidFuzzyConverter(2, 3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        defuzzifier = fuzzy.SigmoidDefuzzifier(converter)        
        result = defuzzifier.forward(fuzzy_set)
        assert result.size() == torch.Size([3, 2])


class TestIsoscelesFuzzyConverter:
    
    def test_fuzzify_converts_to_fuzzy_set(self):

        converter = fuzzy.IsoscelesFuzzyConverter(2, 3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert isinstance(fuzzy_set, torch.Tensor)

    def test_fuzzify_converts_to_fuzzy_set_with_correct_size(self):

        converter = fuzzy.IsoscelesFuzzyConverter(2, 3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert fuzzy_set.data.size() == torch.Size([3, 2, 3])

    def test_fuzzify_converts_to_binary_set_with_correct_size_when_using_same(self):

        converter = fuzzy.IsoscelesFuzzyConverter(1, 3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert fuzzy_set.data.size() == torch.Size([3, 2, 3])

    def test_imply_returns_value_weight_with_correct_size(self):

        converter = fuzzy.IsoscelesFuzzyConverter(1, 3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        value_weight = converter.imply(fuzzy_set)
        assert value_weight.weight.size() == torch.Size([3, 2, 3])
        assert value_weight.value.size() == torch.Size([3, 2, 3])

    def test_accumulate_returns_tensor_of_correct_size(self):

        converter = fuzzy.IsoscelesFuzzyConverter(1, 3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        value_weight = converter.imply(fuzzy_set)
        result = converter.accumulate(value_weight)
        assert result.size() == torch.Size([3, 2])


class TestTriangleFuzzyConverter:
    
    def test_fuzzify_converts_to_fuzzy_set(self):

        converter = fuzzy.TriangleFuzzyConverter(2, 3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert isinstance(fuzzy_set, torch.Tensor)

    def test_fuzzify_converts_to_fuzzy_set_with_correct_size(self):

        converter = fuzzy.TriangleFuzzyConverter(2, 3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert fuzzy_set.data.size() == torch.Size([3, 2, 3])

    def test_fuzzify_converts_to_binary_set_with_correct_size_when_using_same(self):

        converter = fuzzy.TriangleFuzzyConverter(1, 3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert fuzzy_set.data.size() == torch.Size([3, 2, 3])

    def test_fuzzify_converts_to_fuzzy_set_with_correct_size_and_four_terms(self):

        converter = fuzzy.TriangleFuzzyConverter(3, 4)
        fuzzy_set = converter.fuzzify(torch.rand(3, 3))
        assert fuzzy_set.data.size() == torch.Size([3, 3, 4])

    def test_imply_returns_value_weight_with_correct_size(self):

        converter = fuzzy.TriangleFuzzyConverter(1, 3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        value_weight = converter.imply(fuzzy_set)
        assert value_weight.weight.size() == torch.Size([3, 2, 3])
        assert value_weight.value.size() == torch.Size([3, 2, 3])

    def test_accumulate_returns_tensor_of_correct_size(self):

        converter = fuzzy.TriangleFuzzyConverter(1, 3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        value_weight = converter.imply(fuzzy_set)
        result = converter.accumulate(value_weight)
        assert result.size() == torch.Size([3, 2])

    def test_params_are_different_after_calling_fuzzify_and_optimizing(self):

        converter = fuzzy.TriangleFuzzyConverter(1, 3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        
        before = torch.nn.utils.parameters_to_vector(converter.parameters())
        optim = torch.optim.SGD(converter.parameters(), lr=1e-2)
        torch.nn.MSELoss()(torch.rand(fuzzy_set.size(), device=fuzzy_set.device), fuzzy_set).backward()
        optim.step()
        after = torch.nn.utils.parameters_to_vector(converter.parameters())
        assert (after != before).any()


class TestTrapezoidFuzzyConverter:
    
    def test_fuzzify_converts_to_fuzzy_set(self):

        converter = fuzzy.TriangleFuzzyConverter(2, 3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert isinstance(fuzzy_set, torch.Tensor)

    def test_fuzzify_converts_to_fuzzy_set_with_correct_size(self):

        converter = fuzzy.TriangleFuzzyConverter(2, 3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert fuzzy_set.data.size() == torch.Size([3, 2, 3])

    def test_fuzzify_converts_to_binary_set_with_correct_size_when_using_same(self):

        converter = fuzzy.TriangleFuzzyConverter(1, 3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert fuzzy_set.data.size() == torch.Size([3, 2, 3])

    def test_imply_returns_value_weight_with_correct_size(self):

        converter = fuzzy.TriangleFuzzyConverter(1, 3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        value_weight = converter.imply(fuzzy_set)
        assert value_weight.weight.size() == torch.Size([3, 2, 3])
        assert value_weight.value.size() == torch.Size([3, 2, 3])

    def test_accumulate_returns_tensor_of_correct_size(self):

        converter = fuzzy.TriangleFuzzyConverter(1, 3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        value_weight = converter.imply(fuzzy_set)
        result = converter.accumulate(value_weight)
        assert result.size() == torch.Size([3, 2])


class TestLogisticFuzzyConverter:
    
    def test_fuzzify_converts_to_fuzzy_set(self):

        converter = fuzzy.LogisticFuzzyConverter(2, 3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert isinstance(fuzzy_set, torch.Tensor)

    def test_fuzzify_converts_to_fuzzy_set_with_correct_size(self):

        converter = fuzzy.LogisticFuzzyConverter(2, 3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert fuzzy_set.data.size() == torch.Size([3, 2, 3])

    def test_fuzzify_converts_to_binary_set_with_correct_size_when_using_same(self):

        converter = fuzzy.LogisticFuzzyConverter(1, 3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert fuzzy_set.data.size() == torch.Size([3, 2, 3])

    def test_imply_returns_value_weight_with_correct_size(self):

        converter = fuzzy.LogisticFuzzyConverter(1, 3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        value_weight = converter.imply(fuzzy_set)
        assert value_weight.weight.size() == torch.Size([3, 2, 3])
        assert value_weight.value.size() == torch.Size([3, 2, 3])

    def test_accumulate_returns_tensor_of_correct_size(self):

        converter = fuzzy.LogisticFuzzyConverter(1, 3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        value_weight = converter.imply(fuzzy_set)
        result = converter.accumulate(value_weight)
        assert result.size() == torch.Size([3, 2])
