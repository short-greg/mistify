from mistify.infer import _neurons as neurons
from mistify.infer import fuzzy, boolean

import torch


class TestFuzzyOr(object):
    
    def test_forward_outputs_correct_size_with_no_variables(self):
        composition = neurons.Or(2, 4)
        fuzzy_set = fuzzy.rand(4, 2)
        assert composition.forward(fuzzy_set).size() == torch.Size([4, 4])
    
    def test_forward_outputs_correct_size_with_multiple_variablse(self):
        composition = neurons.Or(2, 4, n_terms=2)
        fuzzy_set = fuzzy.rand(4, 2, 2)
        assert composition.forward(fuzzy_set).size() == torch.Size([4, 2, 4])

    def test_forward_outputs_all_ones_or_zeros(self):
        composition = neurons.Or(2, 4, n_terms=2)
        fuzzy_set = fuzzy.rand(4, 2, 2)
        result = composition.forward(fuzzy_set)
        assert ((result.data <= torch.tensor(1.0)) | (result.data >= torch.tensor(0.0))).all()

    def test_forward_outputs_correct_size(self):
        composition = neurons.Or(2, 4, n_terms=2)
        fuzzy_set = fuzzy.rand(4, 2, 2)
        result = composition.forward(fuzzy_set)
        assert result.data.size() == torch.Size([4, 2, 4])
    

class TestMaxProd(object):
    
    def test_forward_outputs_correct_size_with_no_variables(self):
        composition = neurons.Or(2, 4, f='max_prod')
        fuzzy_set = fuzzy.rand(4, 2)
        assert composition.forward(fuzzy_set).size() == torch.Size([4, 4])
    
    def test_forward_outputs_correct_size_with_multiple_variablse(self):
        composition = neurons.Or(2, 4, f='max_prod', n_terms=2)
        fuzzy_set = fuzzy.rand(4, 2, 2)
        assert composition.forward(fuzzy_set).size() == torch.Size([4, 2, 4])

    def test_forward_outputs_all_ones_or_zeros(self):
        composition = neurons.Or(2, 4, f='max_prod', n_terms=2)
        fuzzy_set = fuzzy.rand(4, 2, 2)
        result = composition.forward(fuzzy_set)
        assert ((result <= torch.tensor(1.0)) | (result >= torch.tensor(0.0))).all()

    def test_forward_outputs_correct_size(self):
        composition = neurons.Or(2, 4, f='max_prod', n_terms=2)
        fuzzy_set = fuzzy.rand(4, 2, 2)
        result = composition.forward(fuzzy_set)
        assert result.size() == torch.Size([4, 2, 4])


class TestMinMax(object):
    
    def test_forward_outputs_correct_size_with_no_variables(self):
        composition = neurons.And(2, 4)
        fuzzy_set = fuzzy.rand(4, 2)
        assert composition.forward(fuzzy_set).size() == torch.Size([4, 4])
    
    def test_forward_outputs_correct_size_with_multiple_variablse(self):
        composition = neurons.And(2, 4, n_terms=2)
        fuzzy_set = fuzzy.rand(4, 2, 2)
        assert composition.forward(fuzzy_set).size() == torch.Size([4, 2, 4])

    def test_forward_outputs_all_ones_or_zeros(self):
        composition = neurons.And(2, 4, n_terms=2)
        fuzzy_set = fuzzy.rand(4, 2, 2)
        result = composition.forward(fuzzy_set)
        assert ((result.data <= torch.tensor(1.0)) | (result.data >= torch.tensor(0.0))).all()

    def test_forward_outputs_correct_size(self):
        composition = neurons.And(2, 4, n_terms=2)
        fuzzy_set = fuzzy.rand(4, 2, 2)
        result = composition.forward(fuzzy_set)
        assert result.data.size() == torch.Size([4, 2, 4])


class TestBinaryComplement:

    def test_binary_complement_outputs_complement(self):

        complement = neurons.Complement()
        x = torch.rand(2, 3).round()
        assert ((1 - x) == complement(x)).all()


class TestCrispComposition(object):
    
    def test_forward_outputs_correct_size_with_no_variables(self):
        composition = neurons.Or(2, 4)
        crisp_set = boolean.rand(4, 2)
        assert composition.forward(crisp_set).size() == torch.Size([4, 4])

    def test_forward_outputs_correct_size_with_multiple_variablse(self):
        composition = neurons.Or(2, 4, n_terms=2)
        crisp_set = boolean.rand(4, 2, 2)
        assert composition.forward(crisp_set).size() == torch.Size([4, 2, 4])

#     # def test_forward_outputs_all_ones_or_zeros(self):
#     #     composition = crisp.CrispComposition(2, 4, True, in_variables=2)
#     #     crisp_set = crisp.CrispSet.rand(4, 2, 2)
#     #     result = composition.forward(crisp_set)
#     #     assert ((result.data == torch.tensor(1.0)) | (result.data == torch.tensor(0.0))).all()

#     # def test_forward_outputs_correct_size(self):
#     #     composition = crisp.CrispComposition(2, 4, True, in_variables=2)
#     #     crisp_set = crisp.CrispSet.rand(4, 2, 2)
#     #     result = composition.forward(crisp_set)
#     #     assert result.data.size() == torch.Size([4, 2, 4])

