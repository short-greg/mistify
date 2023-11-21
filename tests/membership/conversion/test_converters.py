import pytest
from mistify import membership
import torch

import torch


class TestIsoscelesFuzzyConverter:
    
    def test_fuzzify_converts_to_fuzzy_set(self):

        converter = membership.IsoscelesFuzzyConverter.from_linspace(2)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert isinstance(fuzzy_set, torch.Tensor)

    def test_fuzzify_converts_to_fuzzy_set_with_correct_size(self):

        converter = membership.IsoscelesFuzzyConverter.from_linspace(2)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert fuzzy_set.data.size() == torch.Size([3, 2, 2])

    def test_fuzzify_converts_to_binary_set_with_correct_size_when_using_same(self):

        converter = membership.IsoscelesFuzzyConverter.from_linspace(4)
        fuzzy_set = converter.fuzzify(torch.rand(3, 3))
        assert fuzzy_set.data.size() == torch.Size([3, 3, 4])

    def test_imply_returns_value_weight_with_correct_size(self):

        converter = membership.IsoscelesFuzzyConverter.from_linspace(3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        value_weight = converter.imply(fuzzy_set)
        assert value_weight.weight.size() == torch.Size([3, 2, 3])
        assert value_weight.value.size() == torch.Size([3, 2, 3])

    def test_accumulate_returns_tensor_of_correct_size(self):

        converter = membership.IsoscelesFuzzyConverter.from_linspace(3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        value_weight = converter.imply(fuzzy_set)
        result = converter.accumulate(value_weight)
        assert result.size() == torch.Size([3, 2])


class TestTriangleFuzzyConverter:
    
    def test_fuzzify_converts_to_fuzzy_set(self):

        converter = membership.TriangleFuzzyConverter.from_linspace(3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert isinstance(fuzzy_set, torch.Tensor)

    def test_fuzzify_converts_to_fuzzy_set_with_correct_size(self):

        converter = membership.TriangleFuzzyConverter.from_linspace(3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert fuzzy_set.data.size() == torch.Size([3, 2, 3])

    def test_fuzzify_converts_to_binary_set_with_correct_size_when_using_same(self):

        converter = membership.TriangleFuzzyConverter.from_linspace(3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert fuzzy_set.data.size() == torch.Size([3, 2, 3])

    def test_fuzzify_converts_to_fuzzy_set_with_correct_size_and_four_terms(self):

        converter = membership.TriangleFuzzyConverter.from_linspace(4)
        fuzzy_set = converter.fuzzify(torch.rand(3, 3))
        assert fuzzy_set.data.size() == torch.Size([3, 3, 4])

    def test_imply_returns_value_weight_with_correct_size(self):

        converter = membership.TriangleFuzzyConverter.from_linspace(3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        value_weight = converter.imply(fuzzy_set)
        assert value_weight.weight.size() == torch.Size([3, 2, 3])
        assert value_weight.value.size() == torch.Size([3, 2, 3])

    def test_accumulate_returns_tensor_of_correct_size(self):

        converter = membership.TriangleFuzzyConverter.from_linspace(3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        value_weight = converter.imply(fuzzy_set)
        result = converter.accumulate(value_weight)
        assert result.size() == torch.Size([3, 2])

    # def test_params_are_different_after_calling_fuzzify_and_optimizing(self):

    #     converter = membership.TriangleFuzzyConverter.from_linspace(3)
    #     fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        
    #     before = torch.nn.utils.parameters_to_vector(converter.parameters())
    #     optim = torch.optim.SGD(converter.parameters(), lr=1e-2)
    #     torch.nn.MSELoss()(torch.rand(fuzzy_set.size(), device=fuzzy_set.device), fuzzy_set).backward()
    #     optim.step()
    #     after = torch.nn.utils.parameters_to_vector(converter.parameters())
    #     assert (after != before).any()


class TestIsoscelesTrapezoidFuzzyConverter:
    
    def test_fuzzify_converts_to_fuzzy_set(self):

        converter = membership.IsoscelesTrapezoidFuzzyConverter.from_linspace(3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert isinstance(fuzzy_set, torch.Tensor)

    def test_fuzzify_converts_to_fuzzy_set_with_correct_size(self):

        converter = membership.IsoscelesTrapezoidFuzzyConverter.from_linspace(3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert fuzzy_set.data.size() == torch.Size([3, 2, 3])

    def test_fuzzify_converts_to_fuzzy_set_with_correct_size_with_two_terms(self):

        converter = membership.IsoscelesTrapezoidFuzzyConverter.from_linspace(2)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert fuzzy_set.data.size() == torch.Size([3, 2, 2])

    def test_fuzzify_converts_to_binary_set_with_correct_size_when_using_same(self):

        converter = membership.IsoscelesTrapezoidFuzzyConverter.from_linspace(3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert fuzzy_set.data.size() == torch.Size([3, 2, 3])

    def test_imply_returns_value_weight_with_correct_size(self):

        converter = membership.IsoscelesTrapezoidFuzzyConverter.from_linspace(3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        value_weight = converter.imply(fuzzy_set)
        assert value_weight.weight.size() == torch.Size([3, 2, 3])
        assert value_weight.value.size() == torch.Size([3, 2, 3])

    def test_accumulate_returns_tensor_of_correct_size(self):

        converter = membership.IsoscelesTrapezoidFuzzyConverter.from_linspace(3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        value_weight = converter.imply(fuzzy_set)
        result = converter.accumulate(value_weight)
        assert result.size() == torch.Size([3, 2])


class TestTrapezoidFuzzyConverter:
    
    def test_fuzzify_converts_to_fuzzy_set(self):

        converter = membership.TrapezoidFuzzyConverter.from_linspace(3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert isinstance(fuzzy_set, torch.Tensor)

    def test_fuzzify_converts_to_fuzzy_set_with_correct_size(self):

        converter = membership.TrapezoidFuzzyConverter.from_linspace(3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert fuzzy_set.data.size() == torch.Size([3, 2, 3])

    def test_fuzzify_converts_to_fuzzy_set_with_correct_size_with_two_terms(self):

        converter = membership.TrapezoidFuzzyConverter.from_linspace(2)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert fuzzy_set.data.size() == torch.Size([3, 2, 2])

    def test_fuzzify_converts_to_binary_set_with_correct_size_when_using_same(self):

        converter = membership.TrapezoidFuzzyConverter.from_linspace(3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert fuzzy_set.data.size() == torch.Size([3, 2, 3])

    def test_imply_returns_value_weight_with_correct_size(self):

        converter = membership.TrapezoidFuzzyConverter.from_linspace(3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        value_weight = converter.imply(fuzzy_set)
        assert value_weight.weight.size() == torch.Size([3, 2, 3])
        assert value_weight.value.size() == torch.Size([3, 2, 3])

    def test_accumulate_returns_tensor_of_correct_size(self):

        converter = membership.TrapezoidFuzzyConverter.from_linspace(3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        value_weight = converter.imply(fuzzy_set)
        result = converter.accumulate(value_weight)
        assert result.size() == torch.Size([3, 2])


class TestLogisticFuzzyConverter:
    
    def test_fuzzify_converts_to_fuzzy_set(self):

        converter = membership.LogisticFuzzyConverter.from_linspace(3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert isinstance(fuzzy_set, torch.Tensor)

    def test_fuzzify_converts_to_fuzzy_set_with_correct_size(self):

        converter = membership.LogisticFuzzyConverter.from_linspace(3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert fuzzy_set.data.size() == torch.Size([3, 2, 3])

    def test_fuzzify_converts_to_binary_set_with_correct_size_when_using_same(self):

        converter = membership.LogisticFuzzyConverter.from_linspace(3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert fuzzy_set.data.size() == torch.Size([3, 2, 3])

    def test_imply_returns_value_weight_with_correct_size(self):

        converter = membership.LogisticFuzzyConverter.from_linspace(3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        value_weight = converter.imply(fuzzy_set)
        assert value_weight.weight.size() == torch.Size([3, 2, 3])
        assert value_weight.value.size() == torch.Size([3, 2, 3])

    def test_accumulate_returns_tensor_of_correct_size(self):

        converter = membership.LogisticFuzzyConverter.from_linspace(3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        value_weight = converter.imply(fuzzy_set)
        result = converter.accumulate(value_weight)
        assert result.size() == torch.Size([3, 2])


class TestSigmoidFuzzyConverter:
    
    def test_fuzzify_converts_to_fuzzy_set(self):

        converter = membership.SigmoidFuzzyConverter.from_linspace(3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert isinstance(fuzzy_set, torch.Tensor)

    def test_fuzzify_converts_to_fuzzy_set_with_correct_size(self):

        converter = membership.SigmoidFuzzyConverter.from_linspace(3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert fuzzy_set.data.size() == torch.Size([3, 2, 3])

    def test_fuzzify_converts_to_binary_set_with_correct_size_when_using_same(self):

        converter = membership.SigmoidFuzzyConverter.from_linspace(3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert fuzzy_set.data.size() == torch.Size([3, 2, 3])

    def test_imply_returns_value_weight_with_correct_size(self):

        converter = membership.SigmoidFuzzyConverter.from_linspace(3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        value_weight = converter.imply(fuzzy_set)
        assert value_weight.weight.size() == torch.Size([3, 2, 3])
        assert value_weight.value.size() == torch.Size([3, 2, 3])

    def test_accumulate_returns_tensor_of_correct_size(self):

        converter = membership.SigmoidFuzzyConverter.from_linspace(3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        value_weight = converter.imply(fuzzy_set)
        result = converter.accumulate(value_weight)
        assert result.size() == torch.Size([3, 2])


class TestRampFuzzyConverter:
    
    def test_fuzzify_converts_to_fuzzy_set(self):

        converter = membership.RampFuzzyConverter.from_linspace(3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert isinstance(fuzzy_set, torch.Tensor)

    def test_fuzzify_converts_to_fuzzy_set_with_correct_size(self):

        converter = membership.RampFuzzyConverter.from_linspace(3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert fuzzy_set.data.size() == torch.Size([3, 2, 3])

    def test_fuzzify_converts_to_binary_set_with_correct_size_when_using_same(self):

        converter = membership.SigmoidFuzzyConverter.from_linspace(3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert fuzzy_set.data.size() == torch.Size([3, 2, 3])

    def test_imply_returns_value_weight_with_correct_size(self):

        converter = membership.SigmoidFuzzyConverter.from_linspace(3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        value_weight = converter.imply(fuzzy_set)
        assert value_weight.weight.size() == torch.Size([3, 2, 3])
        assert value_weight.value.size() == torch.Size([3, 2, 3])

    def test_accumulate_returns_tensor_of_correct_size(self):

        converter = membership.SigmoidFuzzyConverter.from_linspace(3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        value_weight = converter.imply(fuzzy_set)
        result = converter.accumulate(value_weight)
        assert result.size() == torch.Size([3, 2])


class TestStepFuzzyConverter:
    
    def test_fuzzify_converts_to_fuzzy_set(self):

        converter = membership.StepFuzzyConverter.from_linspace(3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert isinstance(fuzzy_set, torch.Tensor)

    def test_fuzzify_converts_to_fuzzy_set_with_correct_size(self):

        converter = membership.StepFuzzyConverter.from_linspace(3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert fuzzy_set.data.size() == torch.Size([3, 2, 3])

    def test_fuzzify_converts_to_binary_set_with_correct_size_when_using_same(self):

        converter = membership.StepFuzzyConverter.from_linspace(3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert fuzzy_set.data.size() == torch.Size([3, 2, 3])

    def test_imply_returns_value_weight_with_correct_size(self):

        converter = membership.StepFuzzyConverter.from_linspace(3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        value_weight = converter.imply(fuzzy_set)
        assert value_weight.weight.size() == torch.Size([3, 2, 3])
        assert value_weight.value.size() == torch.Size([3, 2, 3])

    def test_accumulate_returns_tensor_of_correct_size(self):

        converter = membership.StepFuzzyConverter.from_linspace(3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        value_weight = converter.imply(fuzzy_set)
        result = converter.accumulate(value_weight)
        assert result.size() == torch.Size([3, 2])
