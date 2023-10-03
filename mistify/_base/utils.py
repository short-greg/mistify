import typing

import torch

def weight_func(wf: typing.Union[str, typing.Callable]) -> typing.Callable:
    if isinstance(wf, str):
        return torch.sigmoid if wf == "sigmoid" else torch.clamp
    return wf

