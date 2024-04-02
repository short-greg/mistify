from mistify.learn import _wrap as wrap
import torch


class TestRelOut:

    def test_min_sum_rel_out_outputs_correct_shape(self):

        x = torch.rand(32, 16)
        w = torch.rand(16, 4)
        t = torch.rand(32, 4)
        rel = wrap.MinSumRelOut()
        chosen_x, chosen_w = rel(x, w, t)
        assert chosen_x.shape == t.shape
        assert chosen_w.shape == t.shape

    def test_max_prod_rel_out_outputs_correct_shape(self):

        x = torch.rand(32, 16)
        w = torch.rand(16, 4)
        t = torch.rand(32, 4)
        rel = wrap.MaxProdRelOut()
        chosen_x, chosen_w = rel(x, w, t)
        assert chosen_x.shape == t.shape
        assert chosen_w.shape == t.shape

    def test_max_min_rel_out_outputs_correct_shape(self):

        x = torch.rand(32, 16)
        w = torch.rand(16, 4)
        t = torch.rand(32, 4)
        rel = wrap.MaxMinRelOut()
        chosen_x, chosen_w = rel(x, w, t)
        assert chosen_x.shape == t.shape
        assert chosen_w.shape == t.shape

    def test_min_max_rel_out_outputs_correct_shape(self):

        x = torch.rand(32, 16)
        w = torch.rand(16, 4)
        t = torch.rand(32, 4)
        rel = wrap.MinMaxRelOut()
        chosen_x, chosen_w = rel(x, w, t)
        assert chosen_x.shape == t.shape
        assert chosen_w.shape == t.shape
