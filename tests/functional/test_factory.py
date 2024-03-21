from mistify._functional._factory import (
    InterOn, UnionOn, 
    Inter, Union,
    # LogicalF, AndF,
    # OrF
)
import torch


class TestInter:

    def test_inter_on_executes(self):
        x = torch.rand(2, 2, 1)
        x2 = torch.rand(1, 2, 2)
        assert Inter.inter(x, x2).shape == torch.Size([2, 2, 2])

    def test_inter_on_executes_with_factory(self):
        x = torch.rand(2, 2, 1)
        x2 = torch.rand(1, 2, 2)
        assert Inter.bounded_inter.f()(x, x2).shape == torch.Size([2, 2, 2])

class TestUnion:

    def test_union_executes(self):
        x = torch.rand(2, 2, 1)
        x2 = torch.rand(1, 2, 2)
        assert Union.union(x, x2).shape == torch.Size([2, 2, 2])

    def test_union_executes_with_factory(self):
        x = torch.rand(2, 2, 1)
        x2 = torch.rand(1, 2, 2)
        print(Inter.inter)
        assert Union.bounded_union.f()(x, x2).shape == torch.Size([2, 2, 2])

class TestInterOn:

    def test_inter_on_executes(self):
        x = torch.rand(2, 2, 1)
        assert InterOn.inter_on(x, dim=-2, keepdim=False).shape == torch.Size([2, 1])

    def test_inter_on_executes_with_factory(self):
        x = torch.rand(2, 2, 1)
        assert InterOn.bounded_inter_on.f()(x, dim=-2, keepdim=True).shape == torch.Size([2, 1, 1])


class TestUnionOn:

    def test_union_on_executes(self):
        x = torch.rand(2, 2, 1)
        assert UnionOn.union_on(x, dim=-2, keepdim=True).shape == torch.Size([2, 1, 1])

    def test_union_on_executes_with_factory(self):
        x = torch.rand(2, 2, 1)
        print(Inter.inter)
        assert UnionOn.bounded_union_on.f()(x, dim=-2, keepdim=False).shape == torch.Size([2, 1])


# class TestAndF(object):

#     def test_andf_works_with_strings(self):

#         x = torch.rand(4, 3)
#         x2 = torch.rand(3, 4)
#         and_ = AndF('bounded_union', 'inter_on')
#         y = and_(x, x2)
#         assert y.shape == torch.Size([4, 4])

#     def test_andf_works_with_enum(self):

#         x = torch.rand(4, 3)
#         x2 = torch.rand(3, 4)
#         and_ = AndF(Union.prob_union, InterOn.smooth_inter_on)
#         y = and_(x, x2)
#         assert y.shape == torch.Size([4, 4])


# class TestOrF(object):

#     def test_orf_works_with_strings(self):

#         x = torch.rand(4, 3)
#         x2 = torch.rand(3, 4)
#         or_ = OrF('bounded_inter', 'union_on')
#         y = or_(x, x2)
#         assert y.shape == torch.Size([4, 4])

#     def test_orf_works_with_enum(self):

#         x = torch.rand(4, 3)
#         x2 = torch.rand(3, 4)
#         or_ = OrF(Inter.prob_inter, UnionOn.ada_union_on)
#         y = or_(x, x2)
#         assert y.shape == torch.Size([4, 4])

