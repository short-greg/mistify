from mistify.process import _processors as processors
import torch


class TestStdDev:

    def test_fit_fits_the_standard_deviation(self):

        torch.manual_seed(1)
        X = torch.rand(8, 4)
        stddev = processors.StdDev.fit(X)
        assert (stddev.mean == X.mean(dim=0, keepdim=True)).all()
        assert (stddev.std == X.std(dim=0, keepdim=True)).all()

    def test_forward_will_scale_the_results(self):

        torch.manual_seed(1)
        X = torch.rand(8, 4)
        y = torch.rand(8, 4)
        stddev = processors.StdDev.fit(X)
        y_out = stddev(y)
        
        assert y_out.shape == torch.Size([8, 4])

    def test_forward_reduces_results_by_3_if_divisor_set(self):

        torch.manual_seed(1)
        X = torch.rand(8, 4)
        y = torch.rand(8, 4)
        stddev = processors.StdDev.fit(X, divisor=3)
        stddev_target = processors.StdDev.fit(X)
        y_out = stddev(y)
        y_target = stddev_target(y)
        
        assert torch.isclose(y_out, y_target / 3).all()

    def test_reverse_undoes_the_transformation(self):

        torch.manual_seed(1)
        X = torch.rand(8, 4)
        y = torch.rand(8, 4)
        stddev = processors.StdDev.fit(X, divisor=3)
        y_out = stddev(y)
        y_target = stddev.reverse(y_out)
        
        assert torch.isclose(y, y_target).all()


class TestCumGaussian:

    def test_fit_fits_the_standard_deviation(self):

        X = torch.rand(8, 4)
        stddev = processors.CumGaussian.fit(X)
        assert (stddev.mean == X.mean(dim=0, keepdim=True)).all()
        assert (stddev.std == X.std(dim=0, keepdim=True)).all()

    def test_forward_will_scale_the_results(self):

        X = torch.rand(8, 4)
        y = torch.rand(8, 4)
        cum_gaussian = processors.CumGaussian.fit(X)
        y_out = cum_gaussian(y)
        
        assert y_out.shape == torch.Size([8, 4])
        assert ((0 <= y_out) & (y_out <= 1)).all()

    def test_reverse_undoes_the_transformation(self):

        X = torch.rand(8, 4)
        y = torch.rand(8, 4)
        cum_gaussian = processors.CumGaussian.fit(X)
        y_out = cum_gaussian(y)
        y_target = cum_gaussian.reverse(y_out)
        
        assert torch.isclose(y, y_target).all()


class TestCumLogistic:

    def test_fit_fits_the_logistic(self):

        X = torch.rand(8, 4)
        logistic = processors.CumLogistic.fit(X)
        assert (logistic.scale.shape == torch.Size([1, 4]))
        assert (logistic.loc.shape == torch.Size([1, 4]))

    def test_forward_will_scale_the_results(self):

        torch.manual_seed(1)
        X = torch.rand(8, 4)
        y = torch.rand(8, 4)
        logistic = processors.CumLogistic.fit(X, iterations=10)
        y_out = logistic(y)
        
        assert y_out.shape == torch.Size([8, 4])
        assert ((0 <= y_out) & (y_out <= 1)).all()

    def test_reverse_undoes_the_transformation(self):

        torch.manual_seed(1)
        X = torch.rand(8, 4)
        y = torch.rand(8, 4)
        logistic = processors.CumLogistic.fit(X, iterations=10)
        y_out = logistic(y)
        y_target = logistic.reverse(y_out)
        
        assert torch.isclose(y, y_target, 1e-4).all()


class TestSigmoidParam:

    def test_creates_parameters_of_the_correct_size(self):

        logistic = processors.SigmoidParam(4)
        assert (logistic.scale.shape == torch.Size([1, 4]))
        assert (logistic.loc.shape == torch.Size([1, 4]))

    def test_forward_will_scale_the_results(self):

        torch.manual_seed(1)
        y = torch.rand(8, 4)
        logistic = processors.SigmoidParam(4)
        y_out = logistic(y)
        
        assert y_out.shape == torch.Size([8, 4])
        assert ((0 <= y_out) & (y_out <= 1)).all()

    def test_reverse_undoes_the_transformation(self):

        torch.manual_seed(1)
        y = torch.rand(8, 4)
        logistic = processors.SigmoidParam(4)
        y_out = logistic(y)
        y_target = logistic.reverse(y_out)
        
        assert torch.isclose(y, y_target, 1e-4).all()


class TestMinMaxScaler:

    def test_creates_parameters_of_the_correct_size(self):

        X = torch.rand(8, 4)
        min_max = processors.MinMaxScaler.fit(X)
        assert (min_max.lower.shape == torch.Size([1, 4]))
        assert (min_max.upper.shape == torch.Size([1, 4]))

    def test_forward_will_scale_the_results(self):

        torch.manual_seed(1)
        X = torch.cat([torch.rand(8, 4), torch.ones(1, 4), torch.zeros(1, 4)], dim=0)
        y = torch.rand(8, 4)
        min_max = processors.MinMaxScaler.fit(X)
        y_out = min_max(y)
        
        assert y_out.shape == torch.Size([8, 4])
        assert ((0 <= y_out) & (y_out <= 1)).all()

    def test_reverse_undoes_the_transformation(self):

        torch.manual_seed(1)
        y = torch.rand(8, 4)
        X = torch.cat([torch.rand(8, 4), torch.ones(1, 4), torch.zeros(1, 4)], dim=0)
        min_max = processors.MinMaxScaler.fit(X)
        y_out = min_max(y)
        y_target = min_max.reverse(y_out)
        
        assert torch.isclose(y, y_target, 1e-4).all()
