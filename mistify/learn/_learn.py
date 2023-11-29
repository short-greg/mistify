from abc import abstractmethod


class PostFit(object):

    @abstractmethod
    def fit_postprocessor(self):
        pass


class PreFit(object):
    
    @abstractmethod
    def fit_preprocessor(self):
        pass
