import typing

import torch
from functools import partial


def weight_func(wf: typing.Union[str, typing.Callable]) -> typing.Callable:
    if isinstance(wf, str):
        if wf == 'sigmoid':
            return torch.sigmoid
        if wf == 'clamp':
            return partial(torch.clamp, min=0, max=1)
        if wf == 'sign':
            return torch.sign
        if wf == 'binary':
            return lambda x: torch.clamp(torch.round(x), 0, 1)

        raise ValueError(f'Invalid weight function {wf}')


    return wf


def unsqueeze(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(dim=x.dim())
