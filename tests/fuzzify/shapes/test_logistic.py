import torch

from mistify.fuzzify import ShapeParams
from mistify.fuzzify._shapes import _logistic
from pprint import pprint

class TestLogistic(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        x = torch.rand(2, 3)
        logistic = _logistic.LogisticBell(
            b, s
        )
        m = logistic.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])

    def test_scale_returns_shape_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        logistic = _logistic.LogisticBell(
            b, s
        )
        shape = logistic.scale(m)
        assert isinstance(shape, _logistic.LogisticBell)

    def test_mean_core_returns_tensor_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.ones(2, 3, 4)
        logistic = _logistic.LogisticBell(
            b, s
        )
        shape = logistic.scale(m)
        assert shape.mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        logistic = _logistic.LogisticBell(
            b, s
        )
        shape = logistic.scale(m)
        assert shape.centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        logistic = _logistic.LogisticBell(
            b, s
        )
        shape = logistic.scale(m)
        assert shape.areas.shape == torch.Size([2, 3, 4])

    def test_truncate_returns_trapezoid(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        logistic = _logistic.LogisticBell(
            b, s
        )
        shape = logistic.truncate(m)
        assert isinstance(shape, _logistic.LogisticTrapezoid)


class TestLogisticTrapezoid(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        truncated_m = torch.rand(2, 3, 4)
        x = torch.rand(2, 3)
        logistic = _logistic.LogisticTrapezoid(
            b, s, truncated_m
        )
        m = logistic.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])

    def test_scale_returns_shape_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        truncated_m = torch.rand(2, 3, 4)
        m = torch.rand(2, 3, 4)
        logistic = _logistic.LogisticTrapezoid(
            b, s, truncated_m
        )
        shape = logistic.scale(m)
        assert isinstance(shape, _logistic.LogisticTrapezoid)

    def test_mean_core_returns_tensor_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        truncated_m = torch.rand(2, 3, 4)

        logistic = _logistic.LogisticTrapezoid(
            b, s, truncated_m
        )
        shape = logistic.scale(m)
        assert shape.mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        truncated_m = torch.rand(2, 3, 4)
        m = torch.rand(2, 3, 4)
        logistic = _logistic.LogisticTrapezoid(
            b, s, truncated_m
        )
        shape = logistic.scale(m)
        assert shape.centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        truncated_m = torch.rand(2, 3, 4)
        m = torch.rand(2, 3, 4)
        logistic = _logistic.LogisticTrapezoid(
            b, s, truncated_m
        )
        shape = logistic.scale(m)
        assert shape.areas.shape == torch.Size([2, 3, 4])

    def test_truncate_returns_trapezoid(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        truncated_m = torch.rand(2, 3, 4)

        logistic = _logistic.LogisticTrapezoid(
            b, s, truncated_m
        )
        shape = logistic.truncate(m)
        assert isinstance(shape, _logistic.LogisticTrapezoid)


class TestRightLogistic(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        truncated_m = torch.rand(2, 3, 4)
        x = torch.rand(2, 3)
        logistic = _logistic.RightLogistic(
            b, s, True, truncated_m
        )
        m = logistic.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])

    def test_scale_returns_shape_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        truncated_m = torch.rand(2, 3, 4)
        m = torch.rand(2, 3, 4)
        logistic = _logistic.RightLogistic(
            b, s, True, truncated_m
        )
        shape = logistic.scale(m)
        assert isinstance(shape, _logistic.RightLogistic)

    def test_mean_core_returns_tensor_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        truncated_m = torch.rand(2, 3, 4)

        logistic = _logistic.RightLogistic(
            b, s, True, truncated_m
        )
        shape = logistic.scale(m)
        assert shape.mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        truncated_m = torch.rand(2, 3, 4)
        m = torch.rand(2, 3, 4)
        logistic = _logistic.RightLogistic(
            b, s, True, truncated_m
        )
        shape = logistic.scale(m)
        assert shape.centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        truncated_m = torch.rand(2, 3, 4)
        m = torch.rand(2, 3, 4)
        logistic = _logistic.RightLogistic(
            b, s, True, truncated_m
        )
        shape = logistic.scale(m)
        assert shape.areas.shape == torch.Size([2, 3, 4])

#     def test_truncate_returns_trapezoid(self):
#         b = ShapeParams(torch.rand(3, 4, 1))
#         s = ShapeParams(torch.rand(3, 4, 1))
#         m = torch.rand(2, 3, 4)
#         truncated_m = torch.rand(2, 3, 4)

#         logistic = _logistic.RightLogistic(
#             b, s, True, truncated_m
#         )
#         shape = logistic.truncate(m)
#         assert isinstance(shape, _logistic.RightLogisticTrapezoid)


class TestRightLogisticTrapezoid(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        truncated_m = torch.rand(2, 3, 4)
        x = torch.rand(2, 3)
        logistic = _logistic.RightLogisticTrapezoid(
            b, s, True, truncated_m
        )
        m = logistic.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])

    def test_scale_returns_shape_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        truncated_m = torch.rand(2, 3, 4)
        m = torch.rand(2, 3, 4)
        logistic = _logistic.RightLogisticTrapezoid(
            b, s, True, truncated_m
        )
        shape = logistic.scale(m)
        assert isinstance(shape, _logistic.RightLogisticTrapezoid)

    def test_mean_core_returns_tensor_with_correct_size(self):

        # b = ShapeParams(torch.rand(3, 4, 1))
        # s = ShapeParams(torch.rand(3, 4, 1))
        # m = torch.rand(2, 3, 4)
        # truncated_m = torch.rand(2, 3, 4)

        # logistic = _logistic.RightLogisticTrapezoid(
        #     b, s, True, truncated_m
        # )
        # pprint(type(logistic))
        # shape = logistic.scale(m)
        # assert shape.mean_cores.shape == torch.Size([2, 3, 4])

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        truncated_m = torch.rand(2, 3, 4)
        m = torch.rand(2, 3, 4)
        logistic = _logistic.RightLogisticTrapezoid(
            b, s, True, truncated_m
        )
        shape = logistic.scale(m)
        assert shape.mean_cores.shape == torch.Size([2, 3, 4])


    def test_centroids_returns_tensor_with_correct_size(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        truncated_m = torch.rand(2, 3, 4)
        m = torch.rand(2, 3, 4)
        logistic = _logistic.RightLogisticTrapezoid(
            b, s, True, truncated_m
        )
        shape = logistic.scale(m)
        assert shape.centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        truncated_m = torch.rand(2, 3, 4)
        m = torch.rand(2, 3, 4)
        logistic = _logistic.RightLogisticTrapezoid(
            b, s, True, truncated_m
        )
        shape = logistic.scale(m)
        assert shape.areas.shape == torch.Size([2, 3, 4])

    def test_truncate_returns_trapezoid(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        truncated_m = torch.rand(2, 3, 4)

        logistic = _logistic.RightLogisticTrapezoid(
            b, s, True, truncated_m
        )
        shape = logistic.truncate(m)
        assert isinstance(shape, _logistic.RightLogisticTrapezoid)
