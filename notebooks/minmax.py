# %%
import torch.nn as nn
import torch
import zenkai
from mistify.learn._infer import MinMaxLoss, MinMax, MinMaxSortedPredictorLoss

## %%


# %%

X = torch.rand(2000, 10)

minmax_t = MinMax(10, 20)
T = minmax_t(X).detach()

minmax = MinMax(10, 20)

maxmin_loss = MinMaxLoss(minmax)
predictor_loss = MinMaxSortedPredictorLoss(minmax, minmax_t)

# %%

from torch.utils.data import DataLoader, TensorDataset

optim = torch.optim.Adam(minmax.parameters(), lr=1e-3)
data = TensorDataset(X, T)


for i in range(1):
    for x_i, t_i in DataLoader(data, batch_size=2000, shuffle=True):

        optim.zero_grad()
        y_i = minmax(x_i)
        loss = maxmin_loss(x_i, y_i, t_i) + 0.01 * predictor_loss(x_i, y_i, t_i)
        loss.backward()
        optim.step()
        mse = (y_i - t_i).pow(2).mean()
        if (i + 1) % 50 == 0:
            print(f'{i}: {mse.item()}')
        zenkai.utils.apply_to_parameters(
            minmax.parameters(), lambda x: x.clamp(0, 1)
        )
