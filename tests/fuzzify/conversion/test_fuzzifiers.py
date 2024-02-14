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
    # def test_fuzzify_defuzzifies_fuzzy_set(self):

    #     torch.manual_seed(1)
    #     converter = fuzzify.GaussianFuzzifier(
    #         4
    #     )
    #     x = torch.rand(3, 2)
    #     fuzzy_set = converter(x)
    #     defuzzified = converter.defuzzify(fuzzy_set)
    #     assert defuzzified.shape == torch.Size((3, 2))

    #     print(defuzzified, x)
    #     assert False


# import torch

# from mistify import membership

# class TestSigmoidDefuzzifier:

#     def test_defuzzifier_defuzzifies(self):

#         converter = membership.SigmoidFuzzyConverter(2, 3)
#         fuzzy_set = converter.fuzzify(torch.rand(3, 2))
#         defuzzifier = membership.SigmoidDefuzzifier(converter)        
#         result = defuzzifier.forward(fuzzy_set)
#         assert result.size() == torch.Size([3, 2])


# class TestConverterDecrispifier:

#     def test_crispify_converts_to_binary_set(self):

#         converter = membership.StepFuzzyConverter(1, 3)
#         decrispifier = membership.ConverterDefuzzifier(
#            converter
#         )
#         x = torch.rand(3, 3, 1)
#         t = converter.defuzzify(x)
#         y = decrispifier(x)
#         assert (y == t).all()


# class TestConverterCrispifier:

#     def test_crispify_converts_to_binary_set(self):

#         converter = membership.StepFuzzyConverter(1, 3)
#         crispifier = membership.ConverterFuzzifier(
#            converter
#         )
#         x = torch.rand(3, 2)
#         t = converter.fuzzify(x)
#         y = crispifier(x)
#         assert (y == t).all()


