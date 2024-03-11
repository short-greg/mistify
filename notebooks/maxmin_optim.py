# %%
import torch.nn as nn
import torch
import zenkai


# %%

class MaxMin(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.w = nn.parameter.Parameter(
            torch.rand(in_features, out_features)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return torch.max(torch.min(x.unsqueeze(-1), self.w[None]), dim=-2)[0]

# %%

def converge_loop(w: torch.Tensor, t: torch.Tensor, lower_bound: torch.Tensor, upper_bound: torch.Tensor, chosen: torch.BoolTensor=None):

    i = 0
    converged = False
    w = ((upper_bound < 1) * t).sum(dim=0, keepdim=True) / ((upper_bound < 1).sum(dim=0, keepdim=True))
    while not converged:

        oob = ((w >= upper_bound) | (w <= lower_bound))
        if chosen is not None:
            oob = oob & chosen
        new_w = (oob * t).sum(dim=0, keepdim=True) / oob.sum(dim=0, keepdim=True)
        new_w[new_w.isnan()] = 1.0
        if (new_w == w).all():
            print('CONVERGED')
            converged = True
        print(w[new_w < w], new_w[new_w < w])
        w = new_w
        i += 1
        if i > 10000:
            print('Could not converge')
            break
    return w


def converge_loop_chosen(x: torch.Tensor, w: torch.Tensor, t: torch.Tensor, lower_bound: torch.Tensor, upper_bound: torch.Tensor, chosen: torch.BoolTensor=None):

    i = 0
    converged = False
    # w = ((upper_bound < 1) * t).sum(dim=0, keepdim=True) / ((upper_bound < 1).sum(dim=0, keepdim=True))
    while not converged:

        oob = ((w >= upper_bound) | (w <= lower_bound))
        if chosen is not None:
            oob = oob & chosen
        new_w = (oob * t).sum(dim=0, keepdim=True) / oob.sum(dim=0, keepdim=True)
        new_w[new_w.isnan()] = 1.0
        if (new_w == w).all():
            print('CONVERGED')
            converged = True
        w = new_w

        inner = torch.min(x, w)
        chosen_val = torch.max(inner, dim=-2, keepdim=True)[0]
        chosen = (inner == chosen_val)
        # chosen.scatter_(-2, chosen_idx, 1)
        i += 1
        if i > 10000:
            print('Could not converge')
            break
    return w


def optimize(maxmin: MaxMin, x: torch.Tensor, t: torch.Tensor):

    w = maxmin.w[None]
    x = x.unsqueeze(-1)
    t = t.unsqueeze(-2)
    less_than = (t <= x)
    greater_than = (t >= x)
    upper_bound = torch.max(greater_than * 1, less_than * t)
    lower_bound = torch.max(greater_than * x, less_than * t)

    # print(lower_bound, upper_bound)

    base_w = converge_loop(w, t, lower_bound, upper_bound)
    inner = torch.min(x, base_w)
    chosen_idx = torch.max(inner, dim=-2, keepdim=True)[1]
    chosen = torch.zeros(inner.shape, dtype=bool, device=inner.device)
    chosen.scatter_(-2, chosen_idx, 1)
    # print(chosen_idx)

    return converge_loop(w, t, lower_bound, upper_bound, chosen)


def optimize_loop(maxmin: MaxMin, x: torch.Tensor, t: torch.Tensor):

    w = maxmin.w[None]
    x = x.unsqueeze(-1)
    t = t.unsqueeze(-2)
    less_than = (t <= x)
    greater_than = (t >= x)
    upper_bound = torch.max(greater_than * 1, less_than * t)
    lower_bound = torch.max(greater_than * x, less_than * t)

    base_w = converge_loop(w, t, lower_bound, upper_bound)
    return converge_loop_chosen(x, base_w, t, lower_bound, upper_bound)

# %%



X = torch.rand(128, 8)

base_maxmin = MaxMin(8, 1)
T = base_maxmin(X).detach()

maxmin = MaxMin(2, 1)

new_w = optimize(maxmin, X, T)

print(new_w, base_maxmin.w)

maxmin.w.data = new_w.squeeze(0)

y = maxmin(X)

# print((y == T).sum())

# print('X: ', X)
# print('Y: ', y)
# print('W: ', maxmin.w)
# print('T: ', T)
# print('W2: ', base_maxmin.w)


chosen_idx1 = torch.max(torch.min(X.unsqueeze(-2), maxmin.w[None]), dim=-2, keepdim=True)[1]
chosen_idx2 = torch.max(torch.min(X.unsqueeze(-2), base_maxmin.w[None]), dim=-2, keepdim=True)[1]

print('Same: ', (chosen_idx1 == chosen_idx2).float().mean())
# print('Same: ', (chosen_idx1 == chosen_idx2).all())

print(torch.isclose(y, T).all())

# %%
