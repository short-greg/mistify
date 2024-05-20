import torch
import logging


class PostFit(object):

    def fit_postprocessor(self, X: torch.Tensor, t: torch.Tensor=None):
        if not hasattr(self, 'postprocessor'):
            logging.warn('Trying to fit the postprocessor but no postprocessor is defined')
            return
        self.postprocessor.fit(X, t)


class PreFit(object):
    
    def fit_preprocessor(self, X: torch.Tensor, t: torch.Tensor=None):
        if not hasattr(self, 'preprocessor'):
            logging.warn('Trying to fit the preprocessor but no postprocessor is defined')
            return
        self.preprocessor.fit(X, t)
