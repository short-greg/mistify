from mistify.infer.fuzzy import _activations as activations

import torch


class TestDescale:

    def test_descale_produces_an_output_between_zero_and_one(self):

        descale = activations.Descale(0.2)
        m = torch.rand(6, 8)
        descaled = descale(m)
        assert ((0 <= descaled) & (descaled <= 1.0)).all()


class TestSigmoid:

    def test_sigmoid_produces_an_output_between_zero_and_one(self):

        sigmoidal = activations.Sigmoidal(8)
        m = torch.rand(6, 8)
        activated = sigmoidal(m)
        assert ((0 <= activated) & (activated <= 1.0)).all()

    def test_sigmoid_produces_an_output_between_zero_and_one_with_two_vars(self):

        sigmoidal = activations.Sigmoidal(8, n_vars=2)
        m = torch.rand(6, 2, 8)
        activated = sigmoidal(m)
        assert ((0 <= activated) & (activated <= 1.0)).all()
    
    def test_sigmoid_produces_an_output_between_zero_and_one_with_none_vars(self):

        sigmoidal = activations.Sigmoidal(8, n_vars=None)
        m = torch.rand(6, 2, 8)
        activated = sigmoidal(m)
        assert ((0 <= activated) & (activated <= 1.0)).all()

    def test_sigmoid_produces_an_output_between_zero_and_one_with_two_vars_and_no_batch(self):

        sigmoidal = activations.Sigmoidal(8, n_vars=2)
        m = torch.rand(2, 8)
        activated = sigmoidal(m)
        assert ((0 <= activated) & (activated <= 1.0)).all()


class TestTriangular:

    def test_trianglar_produces_an_output_between_zero_and_one(self):

        triangular = activations.Triangular(8)
        m = torch.rand(6, 8)
        activated = triangular(m)
        assert ((0 <= activated) & (activated <= 1.0)).all()

    def test_trianglar_produces_an_output_between_zero_and_one_with_two_vars(self):

        triangular = activations.Triangular(8, n_vars=2)
        m = torch.rand(6, 2, 8)
        activated = triangular(m)
        assert ((0 <= activated) & (activated <= 1.0)).all()


class TestHedge:

    def test_hedge_produces_an_output_between_zero_and_one(self):

        hedge = activations.Hedge(8)
        m = torch.rand(6, 8)
        activated = hedge(m)
        assert ((0 <= activated) & (activated <= 1.0)).all()

    def test_hedge_produces_an_output_between_zero_and_one_with_two_vars(self):

        hedge = activations.Hedge(8, n_vars=2)
        m = torch.rand(6, 2, 8)
        activated = hedge(m)
        assert ((0 <= activated) & (activated <= 1.0)).all()

    def test_hedge_produces_an_output_between_zero_and_one_with_none_vars(self):

        hedge = activations.Hedge(8, n_vars=None)
        m = torch.rand(6, 2, 8)
        activated = hedge(m)
        assert ((0 <= activated) & (activated <= 1.0)).all()

    def test_hedge_produces_an_output_between_zero_and_one_with_lower_bound(self):

        hedge = activations.Hedge(8, n_vars=None, lower_bound=0.8, upper_bound=1.2)
        m = torch.rand(6, 2, 8)
        activated = hedge(m)
        assert ((0 <= activated) & (activated <= 1.0)).all()

