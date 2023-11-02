import pytest
from mistify._base import _conversion
import torch
from mistify import boolean
from mistify import ShapeParams
from mistify.boolean import Square


class TestSquare(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        x = torch.rand(2, 3)
        right_trapezoid = boolean.Square(
            ShapeParams(p)
        )
        m = right_trapezoid.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])

    def test_scale_returns_shape_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        square = Square(
            ShapeParams(p)
        )
        square = square.scale(m)
        assert isinstance(square, Square)

    def test_mean_core_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        square = Square(
            ShapeParams(p)
        )
        shape = square.scale(m)
        assert shape.mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        square = Square(
            ShapeParams(p)
        )
        shape = square.scale(m)
        assert shape.centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        square = Square(
            ShapeParams(p)
        )
        shape = square.scale(m)
        assert shape.areas.shape == torch.Size([2, 3, 4])

    def test_truncate_returns_right_trapezoid(self):

        p = torch.rand(3, 4, 2).cumsum(2)
        m = torch.rand(2, 3, 4)
        square = Square(
            ShapeParams(p)
        )
        shape = square.truncate(m)
        assert isinstance(shape, Square)
