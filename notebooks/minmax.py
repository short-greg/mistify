# %%
import torch.nn as nn
import torch
import zenkai


# %%

class MinMax(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.w = nn.parameter.Parameter(
            torch.rand(in_features, out_features)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return torch.min(torch.max(x.unsqueeze(-1), self.w[None]), dim=-2)[0]

# %%


class MinMaxLoss(nn.Module):

    def __init__(self, minmax: MinMax):

        super().__init__()
        self.minmax = minmax
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        x = x.unsqueeze(-1)
        y = y.unsqueeze(-2)
        t = t.unsqueeze(-2)
        w = self.minmax.w.unsqueeze(0)

        shape = list(w.shape)
        shape[0] = x.shape[0]
        inner = torch.min(x, w)
        
        greater_than_x = ((t > x.detach()) & (x.detach() > w)).type_as(w)
        greater_than_x_loss = (greater_than_x * (w - t)).pow(2).mean()

        greater_than_theta = ((t > w.detach()) & (w.detach() > x)).type_as(w)
        greater_than_theta_loss = (greater_than_theta * (x - t)).pow(2).mean()

        # Is this okay?
        less_than = (torch.relu(t - inner)).pow(2).mean()
        greater_than = torch.relu(y - t).pow(2).mean()
        loss = less_than + greater_than + greater_than_x_loss + greater_than_theta_loss
        return loss


class MinMaxPredictorLoss(nn.Module):

    def __init__(self, minmax: MinMax, base_minmax: MinMax):

        super().__init__()
        self.minmax = minmax
        self.w_local = minmax.w.clone().detach()
        self.base_minmax = base_minmax
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        x = x.unsqueeze(-1)
        y = y.unsqueeze(-2)
        t = t.unsqueeze(-2)
        w = self.minmax.w.unsqueeze(0)
        shape = list(w.shape)
        shape[0] = x.shape[0]
        x_comp = 1 - x
        inner = torch.min(x_comp, 1 - w)
        negatives = torch.relu(x - t)
        positives = torch.min(x, t)

        # find out which of the xs
        # correspond to 
        score = (1 - positives.sum(
            dim=0, keepdim=True
        ) / x_comp.sum(dim=0, keepdim=True))

        score[score.isnan()] = 0.0

        inner2 = torch.max(x, score)
        chosen_val = torch.min(inner2, dim=-2, keepdim=True)[0]

        minimum = (inner2 == chosen_val).type_as(positives) * inner2
        cur_w = minimum.sum(dim=0, keepdim=True) / (
            minimum.sum(dim=0, keepdim=True) + positives.sum(dim=0, keepdim=True)
        )

        # # Looks like this
        # # 
        # output = 0.0... Need to find out the cases
        # x = 0.0, t=0.0
        # x = 1.0, 
        #  # 
        # # # check which ones are in bounds and out of bounds
        # # # average based on the the targets that are "out of bounds"
        # # # continue this
        # (1 - x) and (1 - t) / sum(1 - x)
        #
        # sum(1 - t) / sum((1 - x) or (1 - t) ) 
        # t is 0 / (t is 0 or x is 0)


        # # Negative prediction
        #  sum(t) / union(x, t) <- don't do this
        # x = 1.0 t = 0.0
        # x = 1.0 t = 1.0

        # Best to compare by val.. This will be faster though
        inner_validx = torch.min(torch.max(x, cur_w), dim=-2, keepdim=True)

        local_y = inner.gather(-2, inner_validx[1])
        return (local_y - t).pow(2).mean()


class MinMaxSortedPredictorLoss(nn.Module):

    def __init__(self, minmax: MinMax, base_minmax: MinMax):

        super().__init__()
        self.minmax = minmax
        self.w_local = minmax.w.clone().detach()
        self.base_minmax = base_minmax
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        x = x.unsqueeze(-1)
        y = y.unsqueeze(-2)
        t = t.unsqueeze(-2)
        w = self.minmax.w.unsqueeze(0)
        shape = list(w.shape)
        shape[0] = x.shape[0]
        negatives = torch.relu(x - t)
        score = negatives.sum(dim=0, keepdim=True) / x.sum(dim=0, keepdim=True)

        score[score.isnan()] = 0.0

        _, sorted_score_indices = score.sort(-2, True)

        base_w2 = self.base_minmax.w[None].gather(-2, sorted_score_indices)
        y_base = torch.min(torch.max(base_w2, x), dim=-2, keepdim=True)[0]
        print((y_base - t).pow(2).mean())

        sorted_w_vals, _ = w.sort(-2, True)
        target_w_vals = w.gather(-2, sorted_score_indices).detach()

        return (sorted_w_vals - target_w_vals).pow(2).mean()



## %%


# %%

X = torch.rand(2000, 10)

minmax_t = MinMax(10, 20)
T = minmax_t(X).detach()

maxmin = MinMax(10, 20)

maxmin_loss = MinMaxLoss(maxmin)
predictor_loss = MinMaxSortedPredictorLoss(maxmin, minmax_t)

# %%

from torch.utils.data import DataLoader, TensorDataset

optim = torch.optim.Adam(maxmin.parameters(), lr=1e-3)
data = TensorDataset(X, T)


for i in range(1):
    for x_i, t_i in DataLoader(data, batch_size=2000, shuffle=True):

        optim.zero_grad()
        y_i = maxmin(x_i)
        loss = maxmin_loss(x_i, y_i, t_i) + 0.01 * predictor_loss(x_i, y_i, t_i)
        loss.backward()
        optim.step()
        mse = (y_i - t_i).pow(2).mean()
        if (i + 1) % 50 == 0:
            print(f'{i}: {mse.item()}')
        zenkai.utils.apply_to_parameters(
            maxmin.parameters(), lambda x: x.clamp(0, 1)
        )