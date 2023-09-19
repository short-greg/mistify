from mistify import fuzzy
import torch


class TestMaxMin(object):
    
    def test_forward_outputs_correct_size_with_no_variables(self):
        composition = fuzzy.MaxMin(2, 4)
        fuzzy_set = fuzzy.rand(4, 2)
        assert composition.forward(fuzzy_set).size() == torch.Size([4, 4])
    
    def test_forward_outputs_correct_size_with_multiple_variablse(self):
        composition = fuzzy.MaxMin(2, 4, in_variables=2)
        fuzzy_set = fuzzy.rand(4, 2, 2)
        assert composition.forward(fuzzy_set).size() == torch.Size([4, 2, 4])

    def test_forward_outputs_all_ones_or_zeros(self):
        composition = fuzzy.MaxMin(2, 4, in_variables=2)
        fuzzy_set = fuzzy.rand(4, 2, 2)
        result = composition.forward(fuzzy_set)
        assert ((result.data <= torch.tensor(1.0)) | (result.data >= torch.tensor(0.0))).all()

    def test_forward_outputs_correct_size(self):
        composition = fuzzy.MaxMin(2, 4, in_variables=2)
        fuzzy_set = fuzzy.rand(4, 2, 2)
        result = composition.forward(fuzzy_set)
        assert result.data.size() == torch.Size([4, 2, 4])

    def test_clamp_clamps_weights_between_zero_and_one(self):
        maxmin = fuzzy.MaxMin(4, 2)
        maxmin.weight.data = torch.randn(maxmin.weight.data.size())
        maxmin.clamp_weights()
        assert ((maxmin.weight.data >= 0.0) & (maxmin.weight.data <= 1.0)).all()


class TestMaxProd(object):
    
    def test_forward_outputs_correct_size_with_no_variables(self):
        composition = fuzzy.MaxProd(2, 4)
        fuzzy_set = fuzzy.rand(4, 2)
        assert composition.forward(fuzzy_set).size() == torch.Size([4, 4])
    
    def test_forward_outputs_correct_size_with_multiple_variablse(self):
        composition = fuzzy.MaxProd(2, 4, in_variables=2)
        fuzzy_set = fuzzy.rand(4, 2, 2)
        assert composition.forward(fuzzy_set).size() == torch.Size([4, 2, 4])

    def test_forward_outputs_all_ones_or_zeros(self):
        composition = fuzzy.MaxProd(2, 4, in_variables=2)
        fuzzy_set = fuzzy.rand(4, 2, 2)
        result = composition.forward(fuzzy_set)
        assert ((result <= torch.tensor(1.0)) | (result >= torch.tensor(0.0))).all()

    def test_forward_outputs_correct_size(self):
        composition = fuzzy.MaxProd(2, 4, in_variables=2)
        fuzzy_set = fuzzy.rand(4, 2, 2)
        result = composition.forward(fuzzy_set)
        assert result.size() == torch.Size([4, 2, 4])


class TestMinMax(object):
    
    def test_forward_outputs_correct_size_with_no_variables(self):
        composition = fuzzy.MinMax(2, 4)
        fuzzy_set = fuzzy.rand(4, 2)
        assert composition.forward(fuzzy_set).size() == torch.Size([4, 4])
    
    def test_forward_outputs_correct_size_with_multiple_variablse(self):
        composition = fuzzy.MinMax(2, 4, in_variables=2)
        fuzzy_set = fuzzy.rand(4, 2, 2)
        assert composition.forward(fuzzy_set).size() == torch.Size([4, 2, 4])

    def test_forward_outputs_all_ones_or_zeros(self):
        composition = fuzzy.MinMax(2, 4, in_variables=2)
        fuzzy_set = fuzzy.rand(4, 2, 2)
        result = composition.forward(fuzzy_set)
        assert ((result.data <= torch.tensor(1.0)) | (result.data >= torch.tensor(0.0))).all()

    def test_forward_outputs_correct_size(self):
        composition = fuzzy.MinMax(2, 4, in_variables=2)
        fuzzy_set = fuzzy.rand(4, 2, 2)
        result = composition.forward(fuzzy_set)
        assert result.data.size() == torch.Size([4, 2, 4])
