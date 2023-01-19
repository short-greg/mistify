import pytest
from . import conversion
from . import crisp, fuzzy
import torch


class TestStepCrispConverter:

    def test_crispify_converts_to_binary_set(self):

        converter = conversion.StepCrispConverter(2, 3)
        crisp_set = converter.crispify(torch.rand(3, 2))
        assert isinstance(crisp_set, crisp.BinarySet)

    def test_crispify_converts_to_binary_set_with_correct_size(self):

        converter = conversion.StepCrispConverter(2, 3)
        crisp_set = converter.crispify(torch.rand(3, 2))
        assert crisp_set.data.size() == torch.Size([3, 2, 3])

    def test_crispify_converts_to_binary_set_with_correct_size_when_using_same(self):

        converter = conversion.StepCrispConverter(1, 3)
        crisp_set = converter.crispify(torch.rand(3, 2))
        assert crisp_set.data.size() == torch.Size([3, 2, 3])

    def test_imply_returns_value_weight_with_correct_size(self):

        converter = conversion.StepCrispConverter(1, 3)
        crisp_set = converter.crispify(torch.rand(3, 2))
        value_weight = converter.imply(crisp_set)
        assert value_weight.weight.size() == torch.Size([3, 2, 3])
        assert value_weight.value.size() == torch.Size([3, 2, 3])

    def test_accumulate_returns_tensor_of_correct_size(self):

        converter = conversion.StepCrispConverter(1, 3)
        crisp_set = converter.crispify(torch.rand(3, 2))
        value_weight = converter.imply(crisp_set)
        result = converter.accumulate(value_weight)
        assert result.size() == torch.Size([3, 2])


class TestSigmoidFuzzyConverter:

    def test_fuzzify_converts_to_binary_set(self):

        converter = conversion.SigmoidFuzzyConverter(2, 3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert isinstance(fuzzy_set, fuzzy.FuzzySet)

    def test_fuzzify_converts_to_fuzzy_set_with_correct_size(self):

        converter = conversion.SigmoidFuzzyConverter(2, 3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert fuzzy_set.data.size() == torch.Size([3, 2, 3])

    def test_fuzzify_converts_to_binary_set_with_correct_size_when_using_same(self):

        converter = conversion.SigmoidFuzzyConverter(1, 3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert fuzzy_set.data.size() == torch.Size([3, 2, 3])

    def test_imply_returns_value_weight_with_correct_size(self):

        converter = conversion.SigmoidFuzzyConverter(1, 3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        value_weight = converter.imply(fuzzy_set)
        assert value_weight.weight.size() == torch.Size([3, 2, 3])
        assert value_weight.value.size() == torch.Size([3, 2, 3])

    def test_accumulate_returns_tensor_of_correct_size(self):

        converter = conversion.SigmoidFuzzyConverter(1, 3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        value_weight = converter.imply(fuzzy_set)
        result = converter.accumulate(value_weight)
        assert result.size() == torch.Size([3, 2])


class TestSigmoidDefuzzifier:

    def test_defuzzifier_defuzzifies(self):

        converter = conversion.SigmoidFuzzyConverter(2, 3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        defuzzifier = conversion.SigmoidDefuzzifier(converter)        
        result = defuzzifier.forward(fuzzy_set)
        assert result.size() == torch.Size([3, 2])


class TestIsoscelesFuzzyConverter:
    

    def test_fuzzify_converts_to_binary_set(self):

        converter = conversion.IsoscelesFuzzyConverter(2, 3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert isinstance(fuzzy_set, fuzzy.FuzzySet)

    def test_fuzzify_converts_to_fuzzy_set_with_correct_size(self):

        converter = conversion.IsoscelesFuzzyConverter(2, 3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert fuzzy_set.data.size() == torch.Size([3, 2, 3])

    def test_fuzzify_converts_to_binary_set_with_correct_size_when_using_same(self):

        converter = conversion.IsoscelesFuzzyConverter(1, 3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        assert fuzzy_set.data.size() == torch.Size([3, 2, 3])

    def test_imply_returns_value_weight_with_correct_size(self):

        converter = conversion.IsoscelesFuzzyConverter(1, 3)
        fuzzy_set = converter.fuzzify(torch.rand(3, 2))
        value_weight = converter.imply(fuzzy_set)
        assert value_weight.weight.size() == torch.Size([3, 2, 3])
        assert value_weight.value.size() == torch.Size([3, 2, 3])

    # def test_accumulate_returns_tensor_of_correct_size(self):

    #     converter = conversion.IsoscelesFuzzyConverter(1, 3)
    #     fuzzy_set = converter.fuzzify(torch.rand(3, 2))
    #     value_weight = converter.imply(fuzzy_set)
    #     result = converter.accumulate(value_weight)
    #     assert result.size() == torch.Size([3, 2])
