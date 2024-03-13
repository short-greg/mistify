from mistify import fuzzify
import torch


class TestGaussianConverter:
    
    def test_fuzzify_converts_to_fuzzy_set(self):

        converter = fuzzify.GaussianFuzzifier(
            4
        )
        fuzzy_set = converter(torch.rand(3, 2))
        assert fuzzy_set.shape == torch.Size((3, 2, 4))
    
    def test_fuzzify_converts_to_fuzzy_set_with_multiple(self):

        converter = fuzzify.GaussianFuzzifier(
            4, 2
        )
        fuzzy_set = converter(torch.rand(3, 2))
        assert fuzzy_set.shape == torch.Size((3, 2, 4))
