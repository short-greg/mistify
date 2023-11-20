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
#         print(y, t)
#         assert (y == t).all()
