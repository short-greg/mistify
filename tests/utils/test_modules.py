# import torch

# from mistify._functional import _modules as modules


# class TestBoolean:

#     def test_boolean_outputs_either_zero_or_one(self):
        
#         x = torch.rand(4, 2)
#         boolean = modules.Boolean()
#         y = boolean(x)
#         assert ((y == 0.0) | (y == 1.0)).all()

#     def test_boolean_sets_the_grad_on_backward(self):
        
#         x = torch.rand(4, 2, requires_grad=True)

#         boolean = modules.Boolean()
#         boolean(x).sum().backward()
#         assert (x.grad.abs() != 0.0).any()


# class TestSign:

#     def test_boolean_outputs_either_zero_or_one(self):
        
#         x = torch.randn(4, 2)
#         sign = modules.Sign()
#         y = sign(x)
#         assert ((y == -1.0) | (y == 1.0)).all()

#     def test_sign_sets_the_grad_on_backward(self):
        
#         x = torch.rand(4, 2, requires_grad=True)

#         sign = modules.Sign()
#         sign(x).sum().backward()
#         assert (x.grad.abs() != 0.0).any()


# class TestClamp:

#     def test_clamp_outputs_value_between_zero_and_one(self):
        
#         x = torch.randn(4, 2)
#         clamp = modules.Clamp()
#         y = clamp(x)
#         assert ((y >= -1.0) & (y <= 1.0)).all()

#     def test_clamp_sets_the_grad_on_backward(self):
        
#         x = torch.rand(4, 2, requires_grad=True) + 1.0
#         x.retain_grad()
#         clamp = modules.Clamp()
#         clamp(x).sum().backward()
#         assert (x.grad.abs() != 0.0).any()
