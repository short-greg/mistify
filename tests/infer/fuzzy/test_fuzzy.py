from mistify.infer import fuzzy
import torch


class TestFuzzyOr(object):
    
    def test_forward_outputs_correct_size_with_no_variables(self):
        composition = fuzzy.FuzzyOr(2, 4)
        fuzzy_set = fuzzy.rand(4, 2)
        assert composition.forward(fuzzy_set).size() == torch.Size([4, 4])
    
    def test_forward_outputs_correct_size_with_multiple_variablse(self):
        composition = fuzzy.FuzzyOr(2, 4, n_terms=2)
        fuzzy_set = fuzzy.rand(4, 2, 2)
        assert composition.forward(fuzzy_set).size() == torch.Size([4, 2, 4])

    def test_forward_outputs_all_ones_or_zeros(self):
        composition = fuzzy.FuzzyOr(2, 4, n_terms=2)
        fuzzy_set = fuzzy.rand(4, 2, 2)
        result = composition.forward(fuzzy_set)
        assert ((result.data <= torch.tensor(1.0)) | (result.data >= torch.tensor(0.0))).all()

    def test_forward_outputs_correct_size(self):
        composition = fuzzy.FuzzyOr(2, 4, n_terms=2)
        fuzzy_set = fuzzy.rand(4, 2, 2)
        result = composition.forward(fuzzy_set)
        assert result.data.size() == torch.Size([4, 2, 4])
    

class TestMaxProd(object):
    
    def test_forward_outputs_correct_size_with_no_variables(self):
        composition = fuzzy.FuzzyOr(2, 4, f='maxprod')
        fuzzy_set = fuzzy.rand(4, 2)
        assert composition.forward(fuzzy_set).size() == torch.Size([4, 4])
    
    def test_forward_outputs_correct_size_with_multiple_variablse(self):
        composition = fuzzy.FuzzyOr(2, 4, f='maxprod', n_terms=2)
        fuzzy_set = fuzzy.rand(4, 2, 2)
        assert composition.forward(fuzzy_set).size() == torch.Size([4, 2, 4])

    def test_forward_outputs_all_ones_or_zeros(self):
        composition = fuzzy.FuzzyOr(2, 4, f='maxprod', n_terms=2)
        fuzzy_set = fuzzy.rand(4, 2, 2)
        result = composition.forward(fuzzy_set)
        assert ((result <= torch.tensor(1.0)) | (result >= torch.tensor(0.0))).all()

    def test_forward_outputs_correct_size(self):
        composition = fuzzy.FuzzyOr(2, 4, f='maxprod', n_terms=2)
        fuzzy_set = fuzzy.rand(4, 2, 2)
        result = composition.forward(fuzzy_set)
        assert result.size() == torch.Size([4, 2, 4])


class TestMinMax(object):
    
    def test_forward_outputs_correct_size_with_no_variables(self):
        composition = fuzzy.FuzzyAnd(2, 4)
        fuzzy_set = fuzzy.rand(4, 2)
        assert composition.forward(fuzzy_set).size() == torch.Size([4, 4])
    
    def test_forward_outputs_correct_size_with_multiple_variablse(self):
        composition = fuzzy.FuzzyAnd(2, 4, n_terms=2)
        fuzzy_set = fuzzy.rand(4, 2, 2)
        assert composition.forward(fuzzy_set).size() == torch.Size([4, 2, 4])

    def test_forward_outputs_all_ones_or_zeros(self):
        composition = fuzzy.FuzzyAnd(2, 4, n_terms=2)
        fuzzy_set = fuzzy.rand(4, 2, 2)
        result = composition.forward(fuzzy_set)
        assert ((result.data <= torch.tensor(1.0)) | (result.data >= torch.tensor(0.0))).all()

    def test_forward_outputs_correct_size(self):
        composition = fuzzy.FuzzyAnd(2, 4, n_terms=2)
        fuzzy_set = fuzzy.rand(4, 2, 2)
        result = composition.forward(fuzzy_set)
        assert result.data.size() == torch.Size([4, 2, 4])
