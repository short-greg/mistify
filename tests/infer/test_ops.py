from mistify.infer import _ops as ops
import torch


class TestIntersectionOn:

    def test_intersection_on_returns_min_value(self):

        intersect = ops.IntersectionOn()
        m = torch.rand(3, 3, 2)
        m_out = intersect(m)
        t = m.min(dim=2)[0]
        assert (m_out == t).all()

    def test_intersection_on_returns_min_value_with_keepdim(self):

        intersect = ops.IntersectionOn(keepdim=True)
        m = torch.rand(3, 3, 2)
        m_out = intersect(m)
        t = m.min(dim=2, keepdim=True)[0]
        assert (m_out == t).all()

    def test_intersection_on_returns_min_value_on_dim_1(self):

        intersect = ops.IntersectionOn(dim=1)
        m = torch.rand(3, 3, 2)
        m_out = intersect(m)
        t = m.min(dim=1)[0]
        assert (m_out == t).all()

    def test_intersection_on_returns_prod_value_on_dim_1(self):

        intersect = ops.IntersectionOn(f='prod', dim=1)
        m = torch.rand(3, 3, 2)
        m_out = intersect(m)
        t = m.prod(dim=1)
        assert (m_out == t).all()


class TestUnionOn:

    def test_intersection_on_returns_max_value(self):

        intersect = ops.UnionOn()
        m = torch.rand(3, 3, 2)
        m_out = intersect(m)
        t = m.max(dim=2)[0]
        assert (m_out == t).all()

    def test_intersection_on_returns_max_value_with_keepdim(self):

        intersect = ops.UnionOn(keepdim=True)
        m = torch.rand(3, 3, 2)
        m_out = intersect(m)
        t = m.max(dim=2, keepdim=True)[0]
        assert (m_out == t).all()

    def test_intersection_on_returns_max_value_on_dim_1(self):

        intersect = ops.UnionOn(dim=1)
        m = torch.rand(3, 3, 2)
        m_out = intersect(m)
        t = m.max(dim=1)[0]
        assert (m_out == t).all()


class TestElse:

    def test_else_returns_zero_if_all_one(self):

        else_ = ops.Else(keepdim=True)
        x = torch.ones(2, 2)
        y = else_(x)
        t = torch.zeros(2, 1)
        assert (y == t).all()

    def test_else_returns_one_if_all_zero(self):

        else_ = ops.Else(keepdim=True)
        x = torch.zeros(2, 2)
        y = else_(x)
        t = torch.ones(2, 1)
        assert (y == t).all()

    def test_else_returns_one_if_all_zero(self):

        else_ = ops.Else(keepdim=True)
        x = torch.zeros(2, 2)
        y = else_(x)
        t = torch.ones(2, 1)
        assert (y == t).all()

    def test_else_returns_point4_if_all_sum_is_point6(self):

        else_ = ops.Else(keepdim=True)
        x = torch.full((2, 2), 0.3)
        y = else_(x)
        t = torch.full((2, 1), 0.7)
        assert torch.isclose(y, t).all()

    def test_else_returns_point7_if_all_max_is_point3(self):

        else_ = ops.Else('boolean', keepdim=True)
        x = torch.full((2, 2), 0.3)
        y = else_(x)
        t = torch.full((2, 1), 0.7)
        assert torch.isclose(y, t).all()

    def test_else_returns_neg1_if_all_positive(self):

        else_ = ops.Else('signed', keepdim=True)
        x = torch.full((2, 2), 1)
        y = else_(x)
        t = torch.full((2, 1), -1)
        assert torch.isclose(y, t).all()

    def test_else_returns_pos1_if_all_negative(self):

        else_ = ops.Else('signed', keepdim=True)
        x = torch.full((2, 2), -1)
        y = else_(x)
        t = torch.full((2, 1), 1)
        assert torch.isclose(y, t).all()


class TestComplement:

    def test_complement_returns_zero_if_all_one(self):

        complement = ops.Complement()
        x = torch.ones(2, 2)
        y = complement(x)
        t = torch.zeros(2, 2)
        assert (y == t).all()

    def test_complement_returns_one_if_all_zero(self):

        complement = ops.Complement()
        x = torch.zeros(2, 2)
        y = complement(x)
        t = torch.ones(2, 2)
        assert (y == t).all()

    def test_complement_returns_neg_one_if_all_positive(self):

        complement = ops.Complement('signed')
        x = torch.ones(2, 2)
        y = complement(x)
        t = -torch.ones(2, 2)
        assert (y == t).all()

    def test_complement_returns_one_if_all_negative(self):

        complement = ops.Complement('signed')
        x = -torch.ones(2, 2)
        y = complement(x)
        t = torch.ones(2, 2)
        assert (y == t).all()

    def test_boolean_complement_outputs_complement(self):

        complement = ops.Complement()
        x = torch.rand(2, 3).round()
        assert ((1 - x) == complement(x)).all()
