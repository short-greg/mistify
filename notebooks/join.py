# %%
import torch.nn as nn
import torch
import zenkai


class IntersectionOn(nn.Module):

    def __init__(self, dim: int):
        
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x.min(dim=self.dim)


class UnionOn(nn.Module):

    def __init__(self, dim: int):
        
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x.max(dim=self.dim)
    

class IntersectionOnLoss(nn.Module):

    def __init__(self, intersection_on: IntersectionOn):

        super().__init__()
        self.intersection_on = intersection_on
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        y = y.unsqueeze(self.intersection_on.dim)
        t = t.unsqueeze(self.intersection_on.dim)
        # x_max > t => only use x_max
        # x_max < t => use all xs that are less than t
        greater_than = torch.relu(y - t).pow(2).mean()
        less_than = torch.relu(t - x).pow(2).mean()
        return greater_than + less_than


class UnionOnLoss(nn.Module):

    def __init__(self, union_on: UnionOn):

        super().__init__()
        self.union_on = union_on
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        # x_max > t => only use x_max
        # x_max < t => use all xs that are less than t
        less_than = torch.relu(t - y).pow(2).mean()
        greater_than = torch.relu(x - t.unsqueeze(self.union_on.dim)).pow(2).mean()
        return greater_than + less_than

