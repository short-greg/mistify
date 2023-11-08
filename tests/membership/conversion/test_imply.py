# Test all conversion modules
import torch
from mistify.membership import _conversion, _shapes

import pandas as pd
import numpy as np
import torch.nn as nn


class TestImplication(object):

    def test_mean_core_implication(self):

        x = torch.rand(2, 3, 4, 2)
        
        logistic = _shapes.LogisticBell.from_combined(_shapes.ShapeParams(x))
        implication = _conversion.MeanCoreImplication()
        assert (implication(logistic) == x[:,:,:,0]).all()

    def test_centroid_implication(self):

        x = torch.rand(2, 3, 4, 2)
        
        logistic = _shapes.LogisticBell.from_combined(_shapes.ShapeParams(x))
        implication = _conversion.CentroidImplication()
        assert (implication(logistic) == x[:,:,:,0]).all()

    def test_area_implication(self):

        x = torch.rand(2, 3, 4, 2)
        
        logistic = _shapes.LogisticBell.from_combined(_shapes.ShapeParams(x))
        implication = _conversion.AreaImplication()
        assert (implication(logistic).shape == torch.Size([2, 3, 4]))