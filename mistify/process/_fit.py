# 1st party
import logging

# 3rd party
import torch


class PostFit(object):
    """Mixin for fitting a PostProcessor
    """

    def fit_postprocessor(self, X: torch.Tensor, t: torch.Tensor=None):
        """Fit the postprocessor

        Args:
            X (torch.Tensor): The input to fit with
            t (torch.Tensor, optional): The target to fit. Defaults to None.
        """
        if not hasattr(self, 'postprocessor'):
            logging.warn('Trying to fit the postprocessor but no postprocessor is defined')
            return
        self.postprocessor.fit(X, t)


class PreFit(object):
    """Mixin for fitting a PreProcessor
    """

    def fit_preprocessor(self, X: torch.Tensor, t: torch.Tensor=None):
        """Fit the preprocessor

        Args:
            X (torch.Tensor): The input to fit with
            t (torch.Tensor, optional): The target to fit. Defaults to None.
        """
        if not hasattr(self, 'preprocessor'):
            logging.warn('Trying to fit the preprocessor but no preprocessor is defined')
            return
        self.preprocessor.fit(X, t)
