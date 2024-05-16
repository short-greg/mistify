# from mistify.learn import _neurons as wrap
# import torch
# import mistify


# class TestRelOut:

#     def test_min_sum_rel_out_outputs_correct_shape(self):

#         x = torch.rand(32, 16)
#         w = torch.rand(16, 4)
#         t = torch.rand(32, 4)
#         rel = wrap.MinSumRelOut()
#         chosen_x, chosen_w = rel(x, w, t)
#         assert chosen_x.shape == t.shape
#         assert chosen_w.shape == t.shape

#     def test_max_prod_rel_out_outputs_correct_shape(self):

#         x = torch.rand(32, 16)
#         w = torch.rand(16, 4)
#         t = torch.rand(32, 4)
#         rel = wrap.MaxProdRelOut()
#         chosen_x, chosen_w = rel(x, w, t)
#         assert chosen_x.shape == t.shape
#         assert chosen_w.shape == t.shape

#     def test_max_min_rel_out_outputs_correct_shape(self):

#         x = torch.rand(32, 16)
#         w = torch.rand(16, 4)
#         t = torch.rand(32, 4)
#         rel = wrap.MaxMinRelOut()
#         chosen_x, chosen_w = rel(x, w, t)
#         assert chosen_x.shape == t.shape
#         assert chosen_w.shape == t.shape

#     def test_min_max_rel_out_outputs_correct_shape(self):

#         x = torch.rand(32, 16)
#         w = torch.rand(16, 4)
#         t = torch.rand(32, 4)
#         rel = wrap.MinMaxRelOut()
#         chosen_x, chosen_w = rel(x, w, t)
#         assert chosen_x.shape == t.shape
#         assert chosen_w.shape == t.shape


# class TestWrap:

#     def test_wrap_outputs_y_with_correct_shape(self):

#         orf = mistify.OrF(
#             mistify.infer.Inter(),
#             mistify.infer.UnionOn(-2)
#         )
#         wrapper = wrap.WrapNeuron(orf, wrap.MaxMinRelOut())

#         x = torch.randn(2, 4)
#         w = torch.randn(4, 8)
#         y = wrapper(x, w)
#         assert y.shape == torch.Size([2, 8])

#     def test_wrap_computes_grad_for_weight(self):

#         orf = mistify.OrF(
#             mistify.infer.Inter(),
#             mistify.infer.UnionOn(-2)
#         )
#         wrapper = wrap.WrapNeuron(orf, wrap.MaxMinRelOut())

#         x = torch.randn(2, 4)
#         w = torch.randn(4, 8, requires_grad=True)
#         y = wrapper(x, w)
#         y.sum().backward()
#         assert w.grad.shape == torch.Size([4, 8])

#     def test_wrap_computes_grad_for_x_and_weight(self):

#         orf = mistify.OrF(
#             mistify.infer.Inter(),
#             mistify.infer.UnionOn(-2)
#         )
#         wrapper = wrap.WrapNeuron(orf, wrap.MaxMinRelOut())

#         x = torch.randn(2, 4, requires_grad=True)
#         w = torch.randn(4, 8, requires_grad=True)
#         y = wrapper(x, w)
#         y.sum().backward()
#         assert w.grad.shape == torch.Size([4, 8])
#         assert x.grad.shape == torch.Size([2, 4])
