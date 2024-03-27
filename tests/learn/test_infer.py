from mistify.learn import _infer as infer_learn
from mistify import infer
import torch
import zenkai
from zenkai import IO


class TestMaxMinLoss(object):

    def test_maxmin_loss(self):
        or_ = infer.Or(4, 8)
        criterion = infer_learn.MaxMinLoss(or_)
        x = torch.randn(8, 4)
        t = torch.randn(8, 8)
        y = or_(x)
        loss = criterion(x, y, t)
        assert (loss.item() != 0)

    def test_maxmin_loss_with_sum_reduction(self):
        or_ = infer.Or(4, 8)
        criterion = infer_learn.MaxMinLoss(or_, 'sum')
        x = torch.randn(8, 4)
        t = torch.randn(8, 8)
        y = or_(x)
        loss = criterion(x, y, t)
        assert (loss.item() != 0)

class TestMinMaxLoss(object):

    def test_minmax_loss(self):
        and_ = infer.And(4, 8)
        criterion = infer_learn.MinMaxLoss(and_)
        x = torch.randn(8, 4)
        t = torch.randn(8, 8)
        y = and_(x)
        loss = criterion(x, y, t)
        assert (loss.item() != 0)

    def test_minmax_loss_with_sum_reduction(self):
        and_ = infer.And(4, 8)
        criterion = infer_learn.MinMaxLoss(and_, 'sum')
        x = torch.randn(8, 4)
        t = torch.randn(8, 8)
        y = and_(x)
        loss = criterion(x, y, t)
        assert (loss.item() != 0)


class TestMaxMinPredictorLoss(object):

    def test_maxmin_loss(self):
        or_ = infer.Or(4, 8)
        criterion = infer_learn.MaxMinPredictorLoss(or_)
        x = torch.randn(8, 4)
        t = torch.randn(8, 8)
        y = or_(x)
        loss = criterion(x, y, t)
        assert (loss.item() != 0)

    def test_maxmin_loss_with_sum_reduction(self):
        or_ = infer.Or(4, 8)
        criterion = infer_learn.MaxMinPredictorLoss(or_, 'sum')
        x = torch.randn(8, 4)
        t = torch.randn(8, 8)
        y = or_(x)
        loss = criterion(x, y, t)
        assert (loss.item() != 0)


class TestMaxMinSortedPredictorLoss(object):

    def test_maxmin_loss(self):
        or_ = infer.Or(4, 8)
        criterion = infer_learn.MaxMinSortedPredictorLoss(or_)
        x = torch.randn(8, 4)
        t = torch.randn(8, 8)
        y = or_(x)
        loss = criterion(x, y, t)
        assert (loss.item() != 0)

    def test_maxmin_loss_with_sum_reduction(self):
        or_ = infer.Or(4, 8)
        criterion = infer_learn.MaxMinSortedPredictorLoss(or_, 'sum')
        x = torch.randn(8, 4)
        t = torch.randn(8, 8)
        y = or_(x)
        loss = criterion(x, y, t)
        assert (loss.item() != 0)


class TestMinMaxPredictorLoss(object):

    def test_minmax_loss(self):
        and_ = infer.And(4, 8)
        criterion = infer_learn.MinMaxPredictorLoss(and_)
        x = torch.randn(8, 4)
        t = torch.randn(8, 8)
        y = and_(x)
        loss = criterion(x, y, t)
        assert (loss.item() != 0)

    def test_minmax_loss_with_sum_reduction(self):
        and_ = infer.And(4, 8)
        criterion = infer_learn.MinMaxPredictorLoss(and_, 'sum')
        x = torch.randn(8, 4)
        t = torch.randn(8, 8)
        y = and_(x)
        loss = criterion(x, y, t)
        assert (loss.item() != 0)


class TestMinMaxSortedPredictorLoss(object):

    def test_minmax_loss(self):
        and_ = infer.And(4, 8)
        criterion = infer_learn.MinMaxSortedPredictorLoss(and_)
        x = torch.randn(8, 4)
        t = torch.randn(8, 8)
        y = and_(x)
        loss = criterion(x, y, t)
        assert (loss.item() != 0)

    def test_minmax_loss_with_sum_reduction(self):
        and_ = infer.And(4, 8)
        criterion = infer_learn.MinMaxSortedPredictorLoss(and_, 'sum')
        x = torch.randn(8, 4)
        t = torch.randn(8, 8)
        y = and_(x)
        loss = criterion(x, y, t)
        assert (loss.item() != 0)


class TestUnionOnLoss(object):

    def test_union_on_loss(self):
        or_ = infer.UnionOnBase()
        criterion = infer_learn.UnionOnLoss(or_)
        x = torch.randn(8, 4)
        t = torch.randn(8, 8)
        y = or_(x)
        loss = criterion(x, y, t)
        assert (loss.item() != 0)

    def test_union_on_loss_with_sum(self):
        or_ = infer.UnionOnBase()
        criterion = infer_learn.UnionOnLoss(or_, reduction='sum')
        x = torch.randn(8, 4)
        t = torch.randn(8, 8)
        y = or_(x)
        loss = criterion(x, y, t)
        assert (loss.item() != 0)


class TestIntersectionOnLoss(object):

    def test_intersection_on_loss(self):
        and_ = infer.InterOnBase()
        criterion = infer_learn.IntersectionOnLoss(and_)
        x = torch.randn(8, 4)
        t = torch.randn(8, 8)
        y = and_(x)
        loss = criterion(x, y, t)
        assert (loss.item() != 0)

    def test_intersection_on_loss_with_sum(self):
        and_ = infer.InterOnBase()
        criterion = infer_learn.IntersectionOnLoss(and_, reduction='sum')
        x = torch.randn(8, 4)
        t = torch.randn(8, 8)
        y = and_(x)
        loss = criterion(x, y, t)
        assert (loss.item() != 0)
