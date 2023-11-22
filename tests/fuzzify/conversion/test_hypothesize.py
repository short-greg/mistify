# Test all conversion modules
import torch
from mistify.fuzzify import _conversion, _shapes

import pandas as pd
import numpy as np
import torch.nn as nn


class TestHypothesis(object):

    def test_mean_core_hypothesis(self):

        x = torch.rand(2, 3, 4, 2)
        
        logistic = _shapes.LogisticBell.from_combined(_shapes.ShapeParams(x))
        hypothesis = _conversion.MeanCoreHypothesis()
        assert (hypothesis(logistic) == x[:,:,:,0]).all()

    def test_centroid_hypothesis(self):

        x = torch.rand(2, 3, 4, 2)
        
        logistic = _shapes.LogisticBell.from_combined(_shapes.ShapeParams(x))
        hypothesis = _conversion.CentroidHypothesis()
        assert (hypothesis(logistic) == x[:,:,:,0]).all()

    def test_area_hypothesis(self):

        x = torch.rand(2, 3, 4, 2)
        
        logistic = _shapes.LogisticBell.from_combined(_shapes.ShapeParams(x))
        hypothesis = _conversion.AreaHypothesis()
        assert (hypothesis(logistic).shape == torch.Size([2, 3, 4]))
    