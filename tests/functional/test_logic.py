import torch
from mistify._functional import _logic as F, ClipG


class TestOr:

    def test_or_outputs_correct_size(self):

        x = torch.rand(6, 4)
        w = torch.rand(4, 8)
        assert F.max_min(
            x, w
        ).shape == torch.Size([6, 8])

    def test_or_results_in_gradient(self):

        x = torch.rand(6, 4, requires_grad=True)
        x.retain_grad()
        w = torch.rand(4, 8)
        y = F.max_min(x, w)
        y.sum().backward()
        assert (x.grad != 0).any()

    def test_or_results_in_gradient_with_g(self):

        x = torch.rand(6, 4, requires_grad=True)
        x.retain_grad()
        w = torch.rand(4, 8)
        y = F.max_min(x, w, g=ClipG(0.1))
        y.sum().backward()
        assert (x.grad != 0).any()


class TestAnd:

    def test_and_outputs_correct_size(self):

        x = torch.rand(6, 4)
        w = torch.rand(4, 8)
        assert F.min_max(
            x, w
        ).shape == torch.Size([6, 8])

    def test_and_results_in_gradient(self):

        x = torch.rand(6, 4, requires_grad=True)
        x.retain_grad()
        w = torch.rand(4, 8)
        y = F.min_max(x, w)
        y.sum().backward()
        assert (x.grad != 0).any()

    def test_and_results_in_gradient_with_g(self):

        x = torch.rand(6, 4, requires_grad=True)
        x.retain_grad()
        w = torch.rand(4, 8)
        y = F.min_max(x, w, g=ClipG(0.1))
        y.sum().backward()
        assert (x.grad != 0).any()


class TestAdaOr:

    def test_or_outputs_correct_size(self):

        x = torch.rand(6, 4)
        w = torch.rand(4, 8)
        assert F.ada_max_min(
            x, w
        ).shape == torch.Size([6, 8])

    def test_or_results_in_gradient(self):

        x = torch.rand(6, 4, requires_grad=True)
        x.retain_grad()
        w = torch.rand(4, 8)
        y = F.ada_max_min(x, w)
        y.sum().backward()
        assert (x.grad != 0).any()

    def test_or_results_in_gradient_with_g(self):

        x = torch.rand(6, 4, requires_grad=True)
        x.retain_grad()
        w = torch.rand(4, 8)
        y = F.ada_max_min(x, w)
        y.sum().backward()
        assert (x.grad != 0).any()


class TestAdaAnd:

    def test_or_outputs_correct_size(self):

        x = torch.rand(6, 4)
        w = torch.rand(4, 8)
        assert F.ada_min_max(
            x, w
        ).shape == torch.Size([6, 8])

    def test_or_results_in_gradient(self):

        x = torch.rand(6, 4, requires_grad=True)
        x.retain_grad()
        w = torch.rand(4, 8)
        y = F.ada_min_max(x, w)
        y.sum().backward()
        assert (x.grad != 0).any()

    def test_or_results_in_gradient_with_g(self):

        x = torch.rand(6, 4, requires_grad=True)
        x.retain_grad()
        w = torch.rand(4, 8)
        y = F.ada_min_max(x, w)
        y.sum().backward()
        assert (x.grad != 0).any()


class TestOrProd:

    def test_or_outputs_correct_size(self):

        x = torch.rand(6, 4)
        w = torch.rand(4, 8)
        assert F.max_prod(
            x, w
        ).shape == torch.Size([6, 8])

    def test_or_results_in_gradient(self):

        x = torch.rand(6, 4, requires_grad=True)
        x.retain_grad()
        w = torch.rand(4, 8)
        y = F.max_prod(x, w)
        y.sum().backward()
        assert (x.grad != 0).any()

    def test_or_results_in_gradient_with_g(self):

        x = torch.rand(6, 4, requires_grad=True)
        x.retain_grad()
        w = torch.rand(4, 8)
        y = F.max_prod(x, w, g=ClipG(0.1))
        y.sum().backward()
        assert (x.grad != 0).any()
