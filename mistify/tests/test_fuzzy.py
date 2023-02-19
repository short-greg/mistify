from .. import fuzzy
import torch


class TestFuzzySet(object):

    def test_zeros_dim_is_1(self):
        
        zeros = fuzzy.FuzzySet.negatives(4)
        assert zeros.data.dim() == 1
    
    def test_zeros_with_batch_dim_is_2(self):
        
        zeros = fuzzy.FuzzySet.negatives(2, 4)
        assert zeros.data.dim() == 2

    def test_zeros_with_batch_and_variables_dim_is_4(self):
        
        zeros = fuzzy.FuzzySet.negatives(2, 3, 2, 4)
        assert zeros.data.dim() == 4

    def test_ones_with_batch_and_variables_dim_is_4(self):
        
        ones = fuzzy.FuzzySet.positives(2, 3, 2, 4)
        assert ones.data.dim() == 4

    def test_ones_with_batch_and_variables_is_1(self):
        
        ones = fuzzy.FuzzySet.positives(2, 3, 2, 4)
        assert (ones.data == torch.tensor(1.0)).all()

    def test_rand_with_batch_and_variables_dim_is_4(self):
        
        ones = fuzzy.FuzzySet.rand(2, 3, 2, 4)
        assert ones.data.dim() == 4

    def test_intersect_results_in_all_values_being_less_or_same(self):
        torch.manual_seed(1)
        c1 = fuzzy.FuzzySet.rand(2, 3, 2, 4)
        c2 = fuzzy.FuzzySet.rand(2, 3, 2, 4)
        c3 = c1 * c2
        assert (c3.data <= c2.data).all()

    def test_intersect_is_included_in_the_tensor(self):
        
        c1 = fuzzy.FuzzySet.rand(2, 3, 2, 4)
        c2 = fuzzy.FuzzySet.rand(2, 3, 2, 4) * c1
        assert (c1.inclusion(c2).data == 1).all()

    def test_union_is_excluded_in_the_tensor(self):
        
        c1 = fuzzy.FuzzySet.rand(2, 3, 2, 4)
        c2 = fuzzy.FuzzySet.rand(2, 3, 2, 4) + c1
        assert (c1.exclusion(c2).data == 1).all()
    
    def test_differ_is_greater_than_zero_for_all(self):
        
        c1 = fuzzy.FuzzySet.rand(2, 3, 2, 4)
        c2 = fuzzy.FuzzySet.rand(2, 3, 2, 4)
        assert ((c1 - c2).data >= 0.0).all()

    def test_differ_is_included_in_tensor(self):
        
        c1 = fuzzy.FuzzySet.rand(2, 3, 2, 4)
        c2 = fuzzy.FuzzySet.rand(2, 3, 2, 4)
        assert (c1.inclusion(c1 - c2).data == 1.0).all()

    def test_transpose_tranposes_dimensions_correctly(self):
        
        c1 = fuzzy.FuzzySet.rand(2, 3, 2, 4)
        assert (c1.transpose(1, 2).data == c1.data.transpose(1, 2)).all()

    def test_union_results_in_all_values_being_greater_or_same(self):
        
        torch.manual_seed(1)
        c1 = fuzzy.FuzzySet.rand(2, 3, 2, 4)
        c2 = fuzzy.FuzzySet.rand(2, 3, 2, 4)
        c3 = c1 + c2
        assert (c3.data >= c2.data).all()
    
    def test_rand_with_batch_and_variables_is_between_one_and_zero(self):
        
        rands = fuzzy.FuzzySet.rand(2, 3, 2, 4)
        assert ((rands.data <= torch.tensor(1.0)) | (rands.data >= torch.tensor(0.0))).all()



class TestMaxMin(object):
    
    def test_forward_outputs_correct_size_with_no_variables(self):
        composition = fuzzy.MaxMin(2, 4)
        fuzzy_set = fuzzy.FuzzySet.rand(4, 2)
        assert composition.forward(fuzzy_set)
    
    def test_forward_outputs_correct_size_with_complement(self):
        composition = fuzzy.MaxMin(2, 4, True)
        fuzzy_set = fuzzy.FuzzySet.rand(4, 2)
        assert composition.forward(fuzzy_set)

    def test_forward_outputs_correct_size_with_multiple_variablse(self):
        composition = fuzzy.MaxMin(2, 4, True, in_variables=2)
        fuzzy_set = fuzzy.FuzzySet.rand(4, 2, 2)
        assert composition.forward(fuzzy_set)

    def test_forward_outputs_all_ones_or_zeros(self):
        composition = fuzzy.MaxMin(2, 4, True, in_variables=2)
        fuzzy_set = fuzzy.FuzzySet.rand(4, 2, 2)
        result = composition.forward(fuzzy_set)
        assert ((result.data <= torch.tensor(1.0)) | (result.data >= torch.tensor(0.0))).all()

    def test_forward_outputs_correct_size(self):
        composition = fuzzy.MaxMin(2, 4, True, in_variables=2)
        fuzzy_set = fuzzy.FuzzySet.rand(4, 2, 2)
        result = composition.forward(fuzzy_set)
        assert result.data.size() == torch.Size([4, 2, 4])

    def test_clamp_clamps_weights_between_zero_and_one(self):
        maxmin = fuzzy.MaxMin(4, 2)
        maxmin.weight.data.data = torch.randn(maxmin.weight.data.size())
        maxmin.clamp()
        assert ((maxmin.weight.data.data >= 0.0) & (maxmin.weight.data.data <= 1.0)).all()

class TestMaxProd(object):
    
    def test_forward_outputs_correct_size_with_no_variables(self):
        composition = fuzzy.MaxProd(2, 4)
        fuzzy_set = fuzzy.FuzzySet.rand(4, 2)
        assert composition.forward(fuzzy_set)
    
    def test_forward_outputs_correct_size_with_complement(self):
        composition = fuzzy.MaxProd(2, 4, True)
        fuzzy_set = fuzzy.FuzzySet.rand(4, 2)
        assert composition.forward(fuzzy_set)

    def test_forward_outputs_correct_size_with_multiple_variablse(self):
        composition = fuzzy.MaxProd(2, 4, True, in_variables=2)
        fuzzy_set = fuzzy.FuzzySet.rand(4, 2, 2)
        assert composition.forward(fuzzy_set)

    def test_forward_outputs_all_ones_or_zeros(self):
        composition = fuzzy.MaxProd(2, 4, True, in_variables=2)
        fuzzy_set = fuzzy.FuzzySet.rand(4, 2, 2)
        result = composition.forward(fuzzy_set)
        assert ((result.data <= torch.tensor(1.0)) | (result.data >= torch.tensor(0.0))).all()

    def test_forward_outputs_correct_size(self):
        composition = fuzzy.MaxProd(2, 4, True, in_variables=2)
        fuzzy_set = fuzzy.FuzzySet.rand(4, 2, 2)
        result = composition.forward(fuzzy_set)
        assert result.data.size() == torch.Size([4, 2, 4])


class TestMinMax(object):
    
    def test_forward_outputs_correct_size_with_no_variables(self):
        composition = fuzzy.MinMax(2, 4)
        fuzzy_set = fuzzy.FuzzySet.rand(4, 2)
        assert composition.forward(fuzzy_set)
    
    def test_forward_outputs_correct_size_with_complement(self):
        composition = fuzzy.MinMax(2, 4, True)
        fuzzy_set = fuzzy.FuzzySet.rand(4, 2)
        assert composition.forward(fuzzy_set)

    def test_forward_outputs_correct_size_with_multiple_variablse(self):
        composition = fuzzy.MinMax(2, 4, True, in_variables=2)
        fuzzy_set = fuzzy.FuzzySet.rand(4, 2, 2)
        assert composition.forward(fuzzy_set)

    def test_forward_outputs_all_ones_or_zeros(self):
        composition = fuzzy.MinMax(2, 4, True, in_variables=2)
        fuzzy_set = fuzzy.FuzzySet.rand(4, 2, 2)
        result = composition.forward(fuzzy_set)
        assert ((result.data <= torch.tensor(1.0)) | (result.data >= torch.tensor(0.0))).all()

    def test_forward_outputs_correct_size(self):
        composition = fuzzy.MinMax(2, 4, True, in_variables=2)
        fuzzy_set = fuzzy.FuzzySet.rand(4, 2, 2)
        result = composition.forward(fuzzy_set)
        assert result.data.size() == torch.Size([4, 2, 4])
