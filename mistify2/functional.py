from .fuzzy import FuzzySet
import torch
from functools import singledispatch


def smooth_max(m1: FuzzySet, m2: FuzzySet, a: float) -> FuzzySet:
    z1 = ((m1.data + 1) ** a).detach()
    z2 = ((m2.data + 1) ** a).detach()
    return FuzzySet((m1.data * z1 + m2.data * z2) / (z1 + z2))


def smooth_max_on(m: FuzzySet, dim: int, a: float) -> FuzzySet:
    z = ((m.data + 1) ** a).detach()
    return FuzzySet((m.data * z).sum(dim=dim) / z.sum(dim=dim))


def smooth_min(m1: FuzzySet, m2: FuzzySet, a: float) -> FuzzySet:
    
    return smooth_max(m1, m2, -a)


def smooth_min_on(m: FuzzySet, dim: int, a: float) -> FuzzySet:
    return smooth_max_on(m, dim, -a)


def adamax(m1: FuzzySet, m2: FuzzySet):
    
    q = torch.clamp(-690 / torch.log(torch.min(m1.data, m2.data)), min=-1000).detach()
    
    return FuzzySet((m1.data ** q + m2.data ** q) ** (1 / q) / 2)


def adamin(m1: FuzzySet, m2: FuzzySet):

    q = torch.clamp(690 / torch.log(torch.min(m1.data, m2.data)).detach(), max=1000)
    
    return FuzzySet((m1.data ** q + m2.data ** q) ** (1 / q) / 2)


def adamax_on(m: torch.Tensor, dim: int):

    q = torch.clamp(-690 / torch.log(torch.min(m.data, dim=dim)[0]).detach(), min=-1000)
    return FuzzySet((torch.sum(m ** q.unsqueeze(dim), dim=dim) / m.size(dim)) ** (1 / q))

def adamin_on(m: FuzzySet, dim: int):

    q = torch.clamp(690 / torch.log(torch.min(m.data, dim=dim)[0]).detach(), max=1000)
    return FuzzySet((torch.sum(m.data ** q.unsqueeze(dim), dim=dim) / m.size(dim)) ** (1 / q))

