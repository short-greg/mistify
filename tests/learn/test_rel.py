import torch
import torch.nn as nn
from mistify.learn import _rel
from mistify.infer import MaxMin, MaxProd


class TestMaxMinRel:

    def test_max_min_rel_outputs_correct_size_with_w(self):

        w = torch.rand(4, 5)
        t = torch.rand(6, 5)

        rel = _rel.MaxMinRel()
        x_rel = rel(w[None], t[:,None], -1).squeeze(-1)
        assert x_rel.shape == torch.Size([6, 4])

    def test_max_min_rel_outputs_correct_size_with_x(self):

        x = torch.rand(6, 4)
        t = torch.rand(6, 5)

        rel = _rel.MaxMinRel()
        w_rel = rel(x[...,None], t[:,None], 0).squeeze(0)
        assert w_rel.shape == torch.Size([4, 5])

    def test_max_min_rel_outputs_correct_size_with_x_with_terms(self):

        x = torch.rand(6, 3, 4)
        t = torch.rand(6, 3, 5)

        rel = _rel.MaxMinRel()
        w_rel = rel(x[...,None], t[:,:,None], 0).squeeze(0)
        assert w_rel.shape == torch.Size([3, 4, 5])


class TestXRel:

    def test_x_rel_outputs_correct_size_with_w(self):

        w = torch.rand(4, 5)
        t = torch.rand(6, 5)

        rel = _rel.XRel(_rel.MaxMinRel())
        x_rel = rel(w, t)
        assert x_rel.shape == torch.Size([6, 4])

    def test_x_rel_outputs_correct_size_with_terms(self):

        w = torch.rand(3, 4, 5)
        t = torch.rand(6, 3, 5)

        rel = _rel.XRel(_rel.MaxMinRel())
        x_rel = rel(w, t)
        assert x_rel.shape == torch.Size([6, 3, 4])


class TestWRel:

    def test_x_rel_outputs_correct_size_with_w(self):

        x = torch.rand(6, 4)
        t = torch.rand(6, 5)

        rel = _rel.WRel(_rel.MaxMinRel())
        w_rel = rel(x, t)
        assert w_rel.shape == torch.Size([4, 5])

    def test_x_rel_outputs_correct_size_with_terms(self):

        x = torch.rand(6, 3, 4)
        t = torch.rand(6, 3, 5)

        rel = _rel.WRel(_rel.MaxMinRel())
        w_rel = rel(x, t)
        assert w_rel.shape == torch.Size([3, 4, 5])


class TestMaxProdRel:

    def test_max_prod_rel_outputs_correct_size_with_w(self):

        w = torch.rand(4, 5)
        t = torch.rand(6, 5)

        rel = _rel.MaxProdRel()
        x_rel = rel(w[None], t[:,None], -1).squeeze(-1)
        assert x_rel.shape == torch.Size([6, 4])

    def test_max_prod_rel_outputs_correct_size_with_x(self):

        x = torch.rand(6, 4)
        t = torch.rand(6, 5)

        rel = _rel.MaxProdRel()
        w_rel = rel(x[...,None], t[:,None], 0).squeeze(0)
        assert w_rel.shape == torch.Size([4, 5])

    def test_max_prod_rel_outputs_correct_size_with_x_with_terms(self):

        x = torch.rand(6, 3, 4)
        t = torch.rand(6, 3, 5)

        rel = _rel.MaxProdRel()
        w_rel = rel(x[...,None], t[:,:,None], 0).squeeze(0)
        assert w_rel.shape == torch.Size([3, 4, 5])


class TestMinSumRel:

    def test_min_sum_rel_outputs_correct_size_with_w(self):

        w = torch.rand(4, 5)
        t = torch.rand(6, 5)

        rel = _rel.MinSumRel()
        x_rel = rel(w[None], t[:,None], -1).squeeze(-1)
        assert x_rel.shape == torch.Size([6, 4])

    def test_min_sum_rel_outputs_correct_size_with_x(self):

        x = torch.rand(6, 4)
        t = torch.rand(6, 5)

        rel = _rel.MinSumRel()
        w_rel = rel(x[...,None], t[:,None], 0).squeeze(0)
        assert w_rel.shape == torch.Size([4, 5])

    def test_min_sum_rel_outputs_correct_size_with_x_with_terms(self):

        x = torch.rand(6, 3, 4)
        t = torch.rand(6, 3, 5)

        rel = _rel.MinSumRel()
        w_rel = rel(x[...,None], t[:,:,None], 0).squeeze(0)
        assert w_rel.shape == torch.Size([3, 4, 5])


class TestMinMaxRel:

    def test_min_max_rel_outputs_correct_size_with_w(self):

        w = torch.rand(4, 5)
        t = torch.rand(6, 5)

        rel = _rel.MinMaxRel()
        x_rel = rel(w[None], t[:,None], -1).squeeze(-1)
        assert x_rel.shape == torch.Size([6, 4])

    def test_min_max_rel_outputs_correct_size_with_x(self):

        x = torch.rand(6, 4)
        t = torch.rand(6, 5)

        rel = _rel.MinMaxRel()
        w_rel = rel(x[...,None], t[:,None], 0).squeeze(0)
        assert w_rel.shape == torch.Size([4, 5])

    def test_min_max_rel_outputs_correct_size_with_x_with_terms(self):

        x = torch.rand(6, 3, 4)
        t = torch.rand(6, 3, 5)

        rel = _rel.MinMaxRel()
        w_rel = rel(x[...,None], t[:,:,None], 0).squeeze(0)
        assert w_rel.shape == torch.Size([3, 4, 5])


class TestRelLoss:

    def test_rel_loss_computes_loss(self):

        max_min = MaxMin(
            4, 5
        )
        loss = _rel.RelLoss(
            nn.MSELoss(), max_min, _rel.MaxMinRel(),
            _rel.MaxMinRel()
        )

        x = torch.rand(6, 4)
        y = max_min(x)
        t = torch.rand(6, 5)

        cost = loss(x, y, t)
        assert cost.dim() == 0

    def test_rel_loss_computes_loss_without_w_rel(self):

        max_min = MaxMin(
            4, 5
        )
        loss = _rel.RelLoss(
            nn.MSELoss(), max_min, _rel.MaxMinRel()
        )

        x = torch.rand(6, 4)
        y = max_min(x)
        t = torch.rand(6, 5)

        cost = loss(x, y, t)
        assert cost.dim() == 0

