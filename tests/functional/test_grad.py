import torch
from mistify._functional import _grad as F, ClipG, ZeroG


class TestG:

    def test_zerog_zeros_the_value(self):
        g = F.ZeroG()
        x = torch.rand(2)
        grad = torch.rand(2)
        oob = torch.tensor([True, False], dtype=torch.bool)
        zerod = g(x, grad, oob)
        assert zerod[0] == 0.0
        assert zerod[1] == grad[1]

    def test_zerog_doesnt_zero_the_value_if_false(self):
        g = F.ZeroG()
        x = torch.rand(2)
        grad = torch.rand(2)
        oob = False
        zerod = g(x, grad, oob)
        assert zerod[0] == grad[0]
        assert zerod[1] == grad[1]

    def test_mulg_multiplies_the_value(self):
        g = F.MulG(0.1)
        x = torch.rand(2)
        grad = torch.rand(2)
        oob = torch.tensor([True, False], dtype=torch.bool)
        muld = g(x, grad, oob)
        assert muld[0] == grad[0] * 0.1
        assert muld[1] == grad[1]

    def test_mulg_multiplies_all_values_if_true(self):
        g = F.MulG(0.1)
        x = torch.rand(2)
        grad = torch.rand(2)
        oob = True
        muld = g(x, grad, oob)
        assert muld[0] == grad[0] * 0.1
        assert muld[1] == grad[1] * 0.1

    def test_mulg_doesnt_multiply_if_none(self):
        g = F.MulG(0.1)
        x = torch.rand(2)
        grad = torch.rand(2)
        oob = None
        muld = g(x, grad, oob)
        assert muld[0] == grad[0] 
        assert muld[1] == grad[1]

    def test_all_passes_all_values(self):
        g = F.AllG()
        x = torch.rand(2)
        grad = torch.rand(2)
        oob = torch.tensor([True, False], dtype=torch.bool)
        muld = g(x, grad, oob)
        assert muld[0] == grad[0]
        assert muld[1] == grad[1]



class TestSignG:

    def test_sign_g(self):

        x = torch.randn(3, 3, requires_grad=True)
        x.retain_grad()
        y = F.SignG.apply(x)
        y.sum().backward()
        assert (x.grad != 0.0).all()

    def test_sign_g_with_clipped_value_to_zero(self):

        x = torch.randn(3, 3, requires_grad=True)
        x.retain_grad()
        y = F.SignG.apply(x, ZeroG())
        y.sum().backward()
        assert (x.grad == 0.0).all()

    def test_sign_g_with_no_clip(self):

        x = torch.randn(3, 3, requires_grad=True)
        x.retain_grad()
        y = F.SignG.apply(x, ClipG(0.1))
        y.sum().backward()
        assert (x.grad != 0.0).all()


class TestBinaryG:

    def test_sign_g(self):

        x = torch.randn(3, 3, requires_grad=True)
        x.retain_grad()
        y = F.BinaryG.apply(x)
        y.sum().backward()
        assert (x.grad != 0.0).all()

    def test_sign_g_with_clipped_value_to_zero(self):

        x = torch.randn(3, 3, requires_grad=True)
        x.retain_grad()
        y = F.BinaryG.apply(x, ZeroG())
        y.sum().backward()
        assert (x.grad == 0.0).all()

    def test_sign_g_with_no_clip(self):

        x = torch.randn(3, 3, requires_grad=True)
        x.retain_grad()
        y = F.BinaryG.apply(x, ClipG(0.1))
        y.sum().backward()
        assert (x.grad != 0.0).all()


class TestClampG:

    def test_clamp_g(self):

        torch.manual_seed(1)
        x = torch.randn(3, 3, requires_grad=True)
        x.retain_grad()
        y = F.ClampG.apply(x)
        y.sum().backward()
        assert (x.grad != 0.0).all()

    def test_clamp_g_with_min(self):

        torch.manual_seed(1)
        x = torch.randn(3, 3, requires_grad=True)
        x.retain_grad()
        y = F.ClampG.apply(x, 0.0, None, None)
        assert (y >= 0.0).all()

    def test_clamp_g_with_max(self):

        torch.manual_seed(1)
        x = torch.randn(3, 3, requires_grad=True)
        x.retain_grad()
        y = F.ClampG.apply(x, None, 0.0, None)
        assert (y <= 0.0).all()

    def test_clamp_g_with_clip(self):

        torch.manual_seed(1)
        x = torch.randn(3, 3, requires_grad=True)
        x.retain_grad()
        y = F.ClampG.apply(x, 0.0, 1.0, ClipG(0.01))
        y.sum().backward()
        oob = (x < 0.0) | (x > 1.0)
        assert ((x.grad[oob] <= 0.01) & (x.grad[oob] >= -0.01)).all()

    def test_clamp_g_with_clip2(self):

        torch.manual_seed(1)
        x = torch.randn(3, 3, requires_grad=True)
        x.retain_grad()
        y = F.ClampG.apply(x, 0.0, None, ClipG(0.01))
        y.sum().backward()
        oob = (x < 0.0)
        assert ((x.grad[oob] <= 0.01) & (x.grad[oob] >= -0.01)).all()


class TestMaxG:

    def test_max_g(self):

        x = torch.randn(3, 3, requires_grad=True)
        x2 = torch.randn(3, 3, requires_grad=True)
        x.retain_grad()
        x2.retain_grad()
        y = F.MaxG.apply(x, x2)
        y.sum().backward()
        assert (x.grad != 0.0).any()


class TestMinG:

    def test_min_g(self):

        x = torch.randn(3, 3, requires_grad=True)
        x2 = torch.randn(3, 3, requires_grad=True)
        x.retain_grad()
        x2.retain_grad()
        y = F.MinG.apply(x, x2)
        y.sum().backward()
        assert (x.grad != 0.0).any()


class TestMaxOnG:

    def test_max_on_g(self):

        x = torch.randn(3, 3, requires_grad=True)
        x.retain_grad()
        y = F.MaxOnG.apply(x, -1, False, ClipG(0.1))
        y[0].sum().backward()
        assert (x.grad != 0.0).any()

    def test_max_on_g_without_keepdim(self):

        x = torch.randn(3, 3, requires_grad=True)
        x.retain_grad()
        y = F.MaxOnG.apply(x, -1, False, ClipG(0.1))
        y[0].sum().backward()
        assert (x.grad != 0.0).any()

    def test_max_on_g_with_dim_0(self):

        x = torch.randn(3, 3, requires_grad=True)
        x.retain_grad()
        y = F.MaxOnG.apply(x, -1, False, ClipG(0.1))
        y[0].sum().backward()
        assert (x.grad != 0.0).any()


class TestMinOnG:

    def test_min_on_g(self):

        x = torch.randn(3, 3, requires_grad=True)
        x.retain_grad()
        y = F.MinOnG.apply(x, -1, False, ClipG(0.1))
        y[0].sum().backward()
        assert (x.grad != 0.0).any()

    def test_min_on_g_with_dim_0(self):

        x = torch.randn(3, 3, requires_grad=True)
        x.retain_grad()
        y = F.MinOnG.apply(x, 0)
        y[0].sum().backward()
        assert (x.grad != 0.0).any()
