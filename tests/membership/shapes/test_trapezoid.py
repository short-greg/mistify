import torch
from mistify.membership._shapes import _trapezoid, ShapeParams


class TestIncreasingRightTrapezoid(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        x = torch.rand(2, 3)
        right_trapezoid = _trapezoid.IncreasingRightTrapezoid(
            ShapeParams(p)
        )
        m = right_trapezoid.join(x)
        assert m.size() == torch.Size([2, 3, 4])

    def test_scale_returns_shape_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_trapezoid = _trapezoid.IncreasingRightTrapezoid(
            ShapeParams(p)
        )
        shape = right_trapezoid.scale(m)
        assert isinstance(shape, _trapezoid.IncreasingRightTrapezoid)

    def test_mean_core_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_trapezoid = _trapezoid.IncreasingRightTrapezoid(
            ShapeParams(p)
        )
        shape = right_trapezoid.scale(m)
        assert shape.mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_trapezoid = _trapezoid.IncreasingRightTrapezoid(
            ShapeParams(p)
        )
        shape = right_trapezoid.scale(m)
        assert shape.centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_trapezoid = _trapezoid.IncreasingRightTrapezoid(
            ShapeParams(p)
        )
        shape = right_trapezoid.scale(m)
        assert shape.areas.shape == torch.Size([2, 3, 4])

    def test_truncate_returns_right_trapezoid(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = torch.rand(2, 3, 4)
        right_trapezoid = _trapezoid.IncreasingRightTrapezoid(
            ShapeParams(p)
        )
        shape = right_trapezoid.truncate(m)
        assert isinstance(shape, _trapezoid.IncreasingRightTrapezoid)



class TestTrapezoid(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        p = torch.rand(3, 4, 4).cumsum(2)
        x = torch.rand(2, 3)
        trapezoid = _trapezoid.Trapezoid(
            ShapeParams(p)
        )
        m = trapezoid.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])

    def test_scale_returns_shape_with_correct_size(self):

        p = torch.rand(3, 4, 4).cumsum(2)
        m = torch.rand(2, 3, 4)
        trapezoid = _trapezoid.Trapezoid(
            ShapeParams(p)
        )
        shape = trapezoid.scale(m)
        assert isinstance(shape, _trapezoid.Trapezoid)

    def test_mean_core_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 4).cumsum(2)
        m = torch.rand(2, 3, 4)
        trapezoid = _trapezoid.Trapezoid(
            ShapeParams(p)
        )
        shape = trapezoid.scale(m)
        assert shape.mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 4).cumsum(2)
        m = torch.rand(2, 3, 4)
        trapezoid = _trapezoid.Trapezoid(
            ShapeParams(p)
        )
        shape = trapezoid.scale(m)
        assert shape.centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 4).cumsum(2)
        m = torch.rand(2, 3, 4)
        trapezoid = _trapezoid.Trapezoid(
            ShapeParams(p)
        )
        shape = trapezoid.scale(m)
        assert shape.areas.shape == torch.Size([2, 3, 4])

    def test_truncate_returns_trapezoid(self):

        p = torch.rand(3, 4, 4).cumsum(2)
        m = torch.rand(2, 3, 4)
        trapezoid = _trapezoid.Trapezoid(
            ShapeParams(p)
        )
        shape = trapezoid.truncate(m)
        assert isinstance(shape, _trapezoid.Trapezoid)


class TestIsoscelesTrapezoid(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        x = torch.rand(2, 3)
        trapezoid = _trapezoid.IsoscelesTrapezoid(
            ShapeParams(p)
        )
        m = trapezoid.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])

    def test_scale_returns_shape_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = torch.rand(2, 3, 4)
        trapezoid = _trapezoid.IsoscelesTrapezoid(
            ShapeParams(p)
        )
        shape = trapezoid.scale(m)
        assert isinstance(shape, _trapezoid.IsoscelesTrapezoid)

    def test_mean_core_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = torch.rand(2, 3, 4)
        trapezoid = _trapezoid.IsoscelesTrapezoid(
            ShapeParams(p)
        )
        shape = trapezoid.scale(m)
        assert shape.mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = torch.rand(2, 3, 4)
        trapezoid = _trapezoid.IsoscelesTrapezoid(
            ShapeParams(p)
        )
        shape = trapezoid.scale(m)
        assert shape.centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = torch.rand(2, 3, 4)
        trapezoid = _trapezoid.IsoscelesTrapezoid(
            ShapeParams(p)
        )
        shape = trapezoid.scale(m)
        assert shape.areas.shape == torch.Size([2, 3, 4])

    def test_truncate_returns_trapezoid(self):

        p = torch.rand(3, 4, 3).cumsum(2)
        m = torch.rand(2, 3, 4)
        trapezoid = _trapezoid.IsoscelesTrapezoid(
            ShapeParams(p)
        )
        shape = trapezoid.truncate(m)
        assert isinstance(shape, _trapezoid.IsoscelesTrapezoid)