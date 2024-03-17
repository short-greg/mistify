from mistify.functional import _shape as F
import torch

T = torch.tensor

class TestTriangle:
    
    def test_triangle_outputs_correct_value(self):

        x = T([0.5, 1.5])
        y = F.triangle(x, T([0.0]), T([1.0]), T([2.0]))
        assert (y == torch.tensor([0.5, 0.5])).all()

    def test_triangle_outputs_correct_shape(self):

        x = torch.rand(4, 2)
        left = torch.rand(1, 2)
        mid = left + torch.rand(1, 2)
        right = mid + torch.rand(1, 2)
        y = F.triangle(x, left, mid, right)
        assert y.shape == torch.Size([4, 2])

    def test_triangle_outputs_gradients_if_oob(self):

        torch.manual_seed(2)
        x = torch.rand(4, 2, requires_grad=True)
        x.retain_grad()
        left = torch.rand(1, 2)
        mid = left + torch.rand(1, 2)
        right = mid + torch.rand(1, 2)
        y = F.triangle(x, left, mid, right, g=True, clip=0.1)
        y.sum().backward()
        assert (x.grad != 0.0).any()


class TestIsosceles:

    def test_isosceles_outputs_correct_value(self):

        x = T([0.5, 1.5])
        y = F.isosceles(x, T([0.0]), T([1.0]))
        assert (y == torch.tensor([0.5, 0.5])).all()

    def test_isosceles_outputs_correct_shape(self):

        x = torch.rand(4, 2)
        left = torch.rand(1, 2)
        mid = left + torch.rand(1, 2)
        y = F.isosceles(x, left, mid)
        assert y.shape == torch.Size([4, 2])

    def test_isosceles_outputs_gradients_if_oob(self):

        torch.manual_seed(2)
        x = torch.rand(4, 2, requires_grad=True)
        x.retain_grad()
        left = torch.rand(1, 2)
        mid = left + torch.rand(1, 2)
        y = F.isosceles(x, left, mid, g=True, clip=0.1)
        y.sum().backward()
        assert (x.grad != 0.0).any()



class TestTrapezoid:
    
    def test_trapezoid_outputs_correct_value(self):

        x = T([0.5, 1.5, 2.5])
        y = F.trapezoid(x, T([0.0]), T([1.0]), T([2.0]), T([3.0]))
        assert (y == torch.tensor([0.5, 1.0, 0.5])).all()

    def test_trapezoid_outputs_correct_shape(self):

        x = torch.rand(4, 2)
        left = torch.rand(1, 2)
        mid = left + torch.rand(1, 2)
        mid2 = mid + torch.rand(1, 2)
        right = mid2 + torch.rand(1, 2)
        y = F.trapezoid(x, left, mid, mid2, right)
        assert y.shape == torch.Size([4, 2])

    def test_trapezoid_outputs_gradients_if_oob(self):

        torch.manual_seed(2)
        x = torch.rand(4, 2, requires_grad=True)
        x.retain_grad()
        left = torch.rand(1, 2)
        mid = left + torch.rand(1, 2)
        mid2 = mid + torch.rand(1, 2)
        right = mid2 + torch.rand(1, 2)
        y = F.trapezoid(x, left, mid, mid2, right, g=True, clip=0.1)
        y.sum().backward()
        assert (x.grad != 0.0).any()


class TestIsoscelesTrapezoid:

    def test_isosceles_trapezoid_outputs_correct_value(self):

        x = T([0.5, 1.5, 2.5])
        y = F.isosceles_trapezoid(x, T([0.0]), T([1.0]), T([2.0]))
        assert (y == torch.tensor([0.5, 1.0, 0.5])).all()

    def test_isoscele_trapezoid_outputs_correct_shape(self):

        x = torch.rand(4, 2)
        left = torch.rand(1, 2)
        mid = left + torch.rand(1, 2)
        mid2 = mid + torch.rand(1, 2)
        y = F.isosceles_trapezoid(x, left, mid, mid2)
        assert y.shape == torch.Size([4, 2])

    def test_isosceles_trapezoid_outputs_gradients_if_oob(self):

        torch.manual_seed(2)
        x = torch.rand(4, 2, requires_grad=True)
        x.retain_grad()
        left = torch.rand(1, 2)
        mid = left + torch.rand(1, 2)
        mid2 = mid + torch.rand(1, 2)
        y = F.isosceles_trapezoid(x, left, mid, mid2, g=True, clip=0.1)
        y.sum().backward()
        assert (x.grad != 0.0).any()
