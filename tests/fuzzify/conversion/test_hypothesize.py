# Test all conversion modules
import torch
from mistify.fuzzify import _conversion, _shapes

import pandas as pd
import numpy as np
import torch.nn as nn


class TestHypothesis(object):

    def test_mean_core_hypothesis(self):

        x = torch.rand(2, 3, 4, 2)
        m = torch.rand(2, 3, 4)
        
        logistic = _shapes.LogisticBell.from_combined(_shapes.Coords(x))
        hypothesis = _conversion.MeanCoreHypothesis()
        assert (hypothesis([logistic], m).hypo == x[:,:,:,0]).all()

    def test_centroid_hypothesis(self):

        x = torch.rand(2, 3, 4, 2)
        m = torch.rand(2, 3, 4)
        
        logistic = _shapes.LogisticBell.from_combined(_shapes.Coords(x))
        hypothesis = _conversion.CentroidHypothesis()
        assert (hypothesis([logistic], m).hypo == x[:,:,:,0]).all()

    def test_area_hypothesis(self):

        x = torch.rand(2, 3, 4, 2)
        m = torch.rand(2, 3, 4)
        
        logistic = _shapes.LogisticBell.from_combined(_shapes.Coords(x))
        hypothesis = _conversion.AreaHypothesis()
        assert (hypothesis([logistic], m).hypo.shape == torch.Size([2, 3, 4]))
    