import pytest
from mistify.binary import StepCrispConverter
import torch


class TestStepCrispConverter:

    def test_crispify_converts_to_binary_set(self):

        converter = StepCrispConverter(2, 3)
        crisp_set = converter.crispify(torch.rand(3, 2))
        assert isinstance(crisp_set, torch.Tensor)

    def test_crispify_converts_to_binary_set_with_correct_size(self):

        converter = StepCrispConverter(2, 3)
        crisp_set = converter.crispify(torch.rand(3, 2))
        assert crisp_set.data.size() == torch.Size([3, 2, 3])

    def test_crispify_converts_to_binary_set_with_correct_size_when_using_same(self):

        converter = StepCrispConverter(1, 3)
        crisp_set = converter.crispify(torch.rand(3, 2))
        assert crisp_set.data.size() == torch.Size([3, 2, 3])

    def test_imply_returns_value_weight_with_correct_size(self):

        converter = StepCrispConverter(1, 3)
        crisp_set = converter.crispify(torch.rand(3, 2))
        value_weight = converter.imply(crisp_set)
        assert value_weight.weight.size() == torch.Size([3, 2, 3])
        assert value_weight.value.size() == torch.Size([3, 2, 3])

    def test_accumulate_returns_tensor_of_correct_size(self):

        converter = StepCrispConverter(1, 3)
        crisp_set = converter.crispify(torch.rand(3, 2))
        value_weight = converter.imply(crisp_set)
        result = converter.accumulate(value_weight)
        assert result.size() == torch.Size([3, 2])