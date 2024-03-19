from mistify._functional import _shape as F, ClipG
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
        y = F.triangle(x, left, mid, right, g=ClipG(0.1))
        y.sum().backward()
        assert (x.grad != 0.0).any()


class TestRightTriangle:
    
    def test_right_triangle_outputs_correct_value(self):

        x = T([0.25, 1.5])
        y = F.right_triangle(x, T([0.0]), T([1.0]))
        assert (y == torch.tensor([0.75, 0.0])).all()

    def test_right_triangle_outputs_correct_value_with_increasing(self):

        x = T([0.25, 1.5])
        y = F.right_triangle(x, T([0.0]), T([1.0]), True)
        assert (y == torch.tensor([0.25, 0.0])).all()

    def test_right_triangle_outputs_correct_value_with_increasing_on_edge(self):

        x = T([0.25, 1.5])
        y = F.right_triangle(x, T([0.25]), T([1.0]), True)
        assert (y == torch.tensor([0.0, 0.0])).all()

    def test_right_triangle_outputs_correct_shape(self):

        x = torch.rand(4, 2)
        left = torch.rand(1, 2)
        mid = left + torch.rand(1, 2)
        y = F.right_triangle(x, left, mid)
        assert y.shape == torch.Size([4, 2])

    def test_right_triangle_outputs_gradients_if_oob(self):

        torch.manual_seed(2)
        x = torch.rand(4, 2, requires_grad=True)
        x.retain_grad()
        left = torch.rand(1, 2)
        mid = left + torch.rand(1, 2)
        y = F.right_triangle(x, left, mid, g=ClipG(0.1))
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
        y = F.isosceles(x, left, mid, g=ClipG(0.1))
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
        y = F.trapezoid(x, left, mid, mid2, right, g=ClipG(0.1))
        y.sum().backward()
        assert (x.grad != 0.0).any()


class TestRightTrapezoid:
    
    def test_right_trapezoid_outputs_correct_value(self):

        x = T([0.5, 1.5, 2.5])
        y = F.right_trapezoid(x, T([0.0]), T([1.0]), T([2.0]))
        assert (y == torch.tensor([1.0, 0.5, 0.0])).all()

    def test_right_trapezoid_outputs_correct_value(self):

        x = T([0.5, 1.5, 2.5])
        y = F.right_trapezoid(x, T([0.0]), T([1.0]), T([2.0]))
        assert (y == torch.tensor([1.0, 0.5, 0.0])).all()

    def test_trapezoid_outputs_correct_shape(self):

        x = T([0.5, 1.5, 2.5])
        y = F.right_trapezoid(x, T([0.0]), T([1.0]), T([2.0]), True)
        assert (y == torch.tensor([0.5, 1.0, 0.0])).all()
    
    def test_trapezoid_outputs_gradients_if_oob(self):

        torch.manual_seed(2)
        x = torch.rand(4, 2, requires_grad=True)
        x.retain_grad()
        left = torch.rand(1, 2)
        mid = left + torch.rand(1, 2)
        mid2 = mid + torch.rand(1, 2)
        y = F.right_trapezoid(x, left, mid, mid2, g=ClipG(0.1))
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
        y = F.isosceles_trapezoid(x, left, mid, mid2, g=ClipG(0.1))
        y.sum().backward()
        assert (x.grad != 0.0).any()


class TestTrapezoidArea:
    
    def test_trapezoid_area_outputs_correct_value(self):

        area = F.trapezoid_area(T([0.0]), T([1.0]), T([2.0]), T([3.0]))
        assert (area == torch.tensor([2.0])).all()

    def test_isosceles_trapezoid_area_outputs_correct_value(self):

        area = F.isosceles_trapezoid_area(T([0.0]), T([1.0]), T([2.0]))
        assert (area == torch.tensor([2.0])).all()


class TestTriangleArea:
    
    def test_triangle_area_outputs_correct_value(self):

        area = F.triangle_area(T([0.0]), T([1.0]), T([2.0]))
        assert (area == torch.tensor([1.0])).all()

    def test_triangle_area_outputs_correct_value(self):

        area = F.isosceles_area(T([0.0]), T([1.0]))
        assert (area == torch.tensor([1.0])).all()
