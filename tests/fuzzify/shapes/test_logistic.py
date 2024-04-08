import torch

from mistify.fuzzify import ShapeParams
from mistify.fuzzify._shapes import _logistic


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

    def test_mean_core_returns_tensor_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.ones(2, 3, 4)
        logistic = _logistic.LogisticBell(
            b, s
        )
        mean_cores = logistic.mean_cores(m, False)
        assert mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        logistic = _logistic.LogisticBell(
            b, s
        )
        centroids = logistic.centroids(m, True)
        assert centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        logistic = _logistic.LogisticBell(
            b, s
        )
        areas = logistic.areas(m, False)
        assert areas.shape == torch.Size([2, 3, 4])

    def test_truncate_returns_trapezoid(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        logistic = _logistic.LogisticBell(
            b, s
        )
        areas = logistic.areas(m, True)
        assert areas.shape == torch.Size([2, 3, 4])


class TestHalfLogistic(object):

    def test_join_returns_fuzzy_set_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        x = torch.rand(2, 3)
        logistic = _logistic.HalfLogisticBell(
            b, s, True
        )
        m = logistic.join(x)
        assert m.data.size() == torch.Size([2, 3, 4])

    def test_mean_core_returns_tensor_with_correct_size(self):

        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)

        logistic = _logistic.HalfLogisticBell(
            b, s, True
        )

        mean_cores = logistic.mean_cores(m, False)
        assert mean_cores.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size_scaled(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        truncated_m = torch.rand(2, 3, 4)
        m = torch.rand(2, 3, 4)
        logistic = _logistic.HalfLogisticBell(
            b, s, True
        )
        centroids = logistic.centroids(m, False)
        assert centroids.shape == torch.Size([2, 3, 4])

    def test_centroids_returns_tensor_with_correct_size_truncated(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        logistic = _logistic.HalfLogisticBell(
            b, s, True
        )
        centroids = logistic.centroids(m, True)
        assert centroids.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size_truncated(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        logistic = _logistic.HalfLogisticBell(
            b, s, True
        )
        areas = logistic.areas(m, True)
        assert areas.shape == torch.Size([2, 3, 4])

    def test_areas_returns_tensor_with_correct_size_scaled(self):
        b = ShapeParams(torch.rand(3, 4, 1))
        s = ShapeParams(torch.rand(3, 4, 1))
        m = torch.rand(2, 3, 4)
        logistic = _logistic.HalfLogisticBell(
            b, s, True
        )
        areas = logistic.areas(m, False)
        assert areas.shape == torch.Size([2, 3, 4])
