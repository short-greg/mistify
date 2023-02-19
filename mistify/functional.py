import torch
from functools import singledispatch


def smooth_max(x: torch.Tensor, x2: torch.Tensor, a: float) -> torch.Tensor:
    z1 = ((x + 1) ** a).detach()
    z2 = ((x2 + 1) ** a).detach()
    return (x * z1 + x2 * z2) / (z1 + z2)

def smooth_max_on(x: torch.Tensor, dim: int, a: float) -> torch.Tensor:
    z = ((x + 1) ** a).detach()
    return (x * z).sum(dim=dim) / z.sum(dim=dim)


def smooth_min(x: torch.Tensor, x2: torch.Tensor, a: float) -> torch.Tensor:
    return smooth_max(x, x2, -a)


def smooth_min_on(x: torch.Tensor, dim: int, a: float) -> torch.Tensor:
    return smooth_max_on(x, dim, -a)


def adamax(x: torch.Tensor, x2: torch.Tensor):
    q = torch.clamp(-690 / torch.log(torch.min(x, x2)), min=-1000).detach()    
    return ((x ** q + x2 ** q) ** (1 / q) / 2)


def adamin(x: torch.Tensor, x2: torch.Tensor):
    q = torch.clamp(690 / torch.log(torch.min(x, x2)).detach(), max=1000)
    
    return (x ** q + x2 ** q) ** (1 / q) / 2


def adamax_on(x: torch.Tensor, dim: int):

    q = torch.clamp(-690 / torch.log(torch.min(x, dim=dim)[0]).detach(), min=-1000)
    return (torch.sum(x ** q.unsqueeze(dim), dim=dim) / x.size(dim)) ** (1 / q)


def adamin_on(x: torch.Tensor, dim: int):

    q = torch.clamp(690 / torch.log(torch.min(x, dim=dim)[0]).detach(), max=1000)
    return (torch.sum(x ** q.unsqueeze(dim), dim=dim) / x.size(dim)) ** (1 / q)
