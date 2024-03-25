from mistify.infer import _neurons as neurons
from mistify._functional import fuzzy, boolean, MulG, ClipG

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
    

class TestCrispComposition(object):
    
    def test_forward_outputs_correct_size_with_no_variables(self):
        composition = neurons.Or(2, 4)
        crisp_set = boolean.rand(4, 2)
        assert composition.forward(crisp_set).size() == torch.Size([4, 4])

    def test_forward_outputs_correct_size_with_multiple_variablse(self):
        composition = neurons.Or(2, 4, n_terms=2)
        crisp_set = boolean.rand(4, 2, 2)
        assert composition.forward(crisp_set).size() == torch.Size([4, 2, 4])


class TestAndB(object):

    def test_build_and_builds_and_neuron_with_g_with_correct_shape(self):

        x = torch.Tensor(5, 4)
        neuron = neurons.BuildAnd().boolean_wf(ClipG(0.1)).inter_on(
            MulG(0.1)
        ).union(MulG(0.1)).__call__(4, 8)
        assert neuron(x).shape == torch.Size([5, 8])

    def test_build_and_builds_and_neuron_with_g(self):

        neuron = neurons.BuildAnd().boolean_wf(ClipG(0.1)).inter_on(
            MulG(0.1)
        ).union(MulG(0.1)).__call__(4, 8)
        assert isinstance(neuron, neurons.And)


class TestOrB(object):

    def test_build_or_builds_or_neuron_with_correct_out_shape(self):

        x = torch.Tensor(5, 4)
        neuron = (
            neurons.BuildOr()
                   .clamp_wf()
                   .inter()
                   .union_on()
        ).__call__(4, 8)
        assert neuron(x).shape == torch.Size([5, 8])

    def test_build_or_builds_or_neuron_with_g_with_correct_shape(self):

        x = torch.Tensor(5, 4)
        neuron = neurons.BuildOr().boolean_wf(ClipG(0.1)).inter(
            MulG(0.1)
        ).union_on(MulG(0.1)).__call__(4, 8)
        assert neuron(x).shape == torch.Size([5, 8])

    def test_build_or_builds_or_neuron_with_g(self):

        neuron = neurons.BuildOr().boolean_wf(ClipG(0.1)).inter(
            MulG(0.1)
        ).union_on(MulG(0.1)).__call__(4, 8)
        assert isinstance(neuron, neurons.Or)

    def test_build_or_builds_or_neuron_with_prob(self):

        neuron = neurons.BuildOr().boolean_wf(ClipG(0.1)).prob_inter(
        
        ).prob_union_on().__call__(4, 8)
        assert isinstance(neuron, neurons.Or)

