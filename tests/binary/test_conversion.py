import pytest
from mistify.binary import (
    StepCrispConverter, ConverterCrispifier, ConverterDecrispifier
)
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


class TestConverterCrispifier:

    def test_crispify_converts_to_binary_set(self):

        converter = StepCrispConverter(1, 3)
        crispifier = ConverterCrispifier(
           converter
        )
        x = torch.rand(3, 2)
        t = converter.crispify(x)
        y = crispifier(x)
        assert (y == t).all()


class TestConverterDecrispifier:

    def test_crispify_converts_to_binary_set(self):

        converter = StepCrispConverter(1, 3)
        decrispifier = ConverterDecrispifier(
           converter
        )
        x = torch.rand(3, 3, 1)
        t = converter.decrispify(x)
        y = decrispifier(x)
        assert (y == t).all()

    
