import torch.nn as nn
import torch
import torch.nn.functional as nn_func


class Reversible(object):
        
    def reverse(self, *y: torch.Tensor) -> torch.Tensor:
        pass


class Softplus(nn.Module, Reversible):

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return nn_func.softplus(x)

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        
        return torch.log(
            torch.exp(y - 1)
        )


class Exp(nn.Module, Reversible):

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return torch.exp(x)

    def reverse(self, y: torch.Tensor) -> torch.Tensor:
        
        return torch.log(y)
