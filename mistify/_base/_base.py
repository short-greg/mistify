import typing
import torch.nn as nn
from abc import abstractmethod, ABC
from torch.optim import Optimizer


class Constrained(ABC):

    @abstractmethod
    def constrain(self):
        pass


class ConstrainedOptim(object):

    def __init__(self, optim: Optimizer, nets: typing.List[Constrained]):

        self.optim = optim
        self.nets = nets

    def step(self):

        self.optim.step()
        constrain(self.nets)

    def zero_grad(self):

        self.optim.zero_grad()


def constrain(nets: typing.List[Constrained]):

    for net in nets:
        net.constrain()
