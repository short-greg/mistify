import torch
from mistify.functional import _logic as F


class TestOr:

    def test_or_outputs_correct_size(self):

        x = torch.rand(6, 4)
        w = torch.rand(4, 8)
        assert F.or_(
            x, w
        ).shape == torch.Size([6, 8])

    def test_or_results_in_gradient(self):

        x = torch.rand(6, 4, requires_grad=True)
        x.retain_grad()
        w = torch.rand(4, 8)
        y = F.or_(x, w)
        y.sum().backward()
        assert (x.grad != 0).any()

    def test_or_results_in_gradient_with_g(self):

        x = torch.rand(6, 4, requires_grad=True)
        x.retain_grad()
        w = torch.rand(4, 8)
        y = F.or_(x, w, g=True)
        y.sum().backward()
        assert (x.grad != 0).any()


class TestAnd:

    def test_and_outputs_correct_size(self):

        x = torch.rand(6, 4)
        w = torch.rand(4, 8)
        assert F.and_(
            x, w
        ).shape == torch.Size([6, 8])

    def test_and_results_in_gradient(self):

        x = torch.rand(6, 4, requires_grad=True)
        x.retain_grad()
        w = torch.rand(4, 8)
        y = F.and_(x, w)
        y.sum().backward()
        assert (x.grad != 0).any()

    def test_and_results_in_gradient_with_g(self):

        x = torch.rand(6, 4, requires_grad=True)
        x.retain_grad()
        w = torch.rand(4, 8)
        y = F.and_(x, w, g=True)
        y.sum().backward()
        assert (x.grad != 0).any()


class TestAdaOr:

    def test_or_outputs_correct_size(self):

        x = torch.rand(6, 4)
        w = torch.rand(4, 8)
        assert F.ada_or(
            x, w
        ).shape == torch.Size([6, 8])

    def test_or_results_in_gradient(self):

        x = torch.rand(6, 4, requires_grad=True)
        x.retain_grad()
        w = torch.rand(4, 8)
        y = F.ada_or(x, w)
        y.sum().backward()
        assert (x.grad != 0).any()

    def test_or_results_in_gradient_with_g(self):

        x = torch.rand(6, 4, requires_grad=True)
        x.retain_grad()
        w = torch.rand(4, 8)
        y = F.ada_or(x, w)
        y.sum().backward()
        assert (x.grad != 0).any()


class TestAdaAnd:

    def test_or_outputs_correct_size(self):

        x = torch.rand(6, 4)
        w = torch.rand(4, 8)
        assert F.ada_and(
            x, w
        ).shape == torch.Size([6, 8])

    def test_or_results_in_gradient(self):

        x = torch.rand(6, 4, requires_grad=True)
        x.retain_grad()
        w = torch.rand(4, 8)
        y = F.ada_and(x, w)
        y.sum().backward()
        assert (x.grad != 0).any()

    def test_or_results_in_gradient_with_g(self):

        x = torch.rand(6, 4, requires_grad=True)
        x.retain_grad()
        w = torch.rand(4, 8)
        y = F.ada_and(x, w)
        y.sum().backward()
        assert (x.grad != 0).any()


class TestOrProd:

    def test_or_outputs_correct_size(self):

        x = torch.rand(6, 4)
        w = torch.rand(4, 8)
        assert F.or_prod(
            x, w
        ).shape == torch.Size([6, 8])

    def test_or_results_in_gradient(self):

        x = torch.rand(6, 4, requires_grad=True)
        x.retain_grad()
        w = torch.rand(4, 8)
        y = F.or_prod(x, w)
        y.sum().backward()
        assert (x.grad != 0).any()

    def test_or_results_in_gradient_with_g(self):

        x = torch.rand(6, 4, requires_grad=True)
        x.retain_grad()
        w = torch.rand(4, 8)
        y = F.or_prod(x, w, g=True)
        y.sum().backward()
        assert (x.grad != 0).any()
