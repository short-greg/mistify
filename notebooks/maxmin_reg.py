# %%

import initialize
import torch
import torch.nn as nn
from torch.utils import data as torch_data
from itertools import chain

import mistify
import zenkai

# %%

X = torch.rand(5000, 16)


layer1_t = mistify.infer.Or(16, 8, f='smooth_max_min', wf=None)

layer2_t = mistify.infer.And(8, 16, f='smooth_min_max' wf=None)
T = layer2_t(layer1_t(X)).detach()

# T = minmax_t(X).detach()

layer = mistify.infer.Or(16, 64, wf=None)
layer2 = mistify.infer.And(64, 16, wf=None)

dataset = torch_data.TensorDataset(X, T)
optim = torch.optim.Adam(chain(layer.parameters(), layer2.parameters()), lr=1e-3)



# %%

regularize = True

for epoch in range(1000):

    if epoch > 500:
        regularize = True

    for x, t in torch_data.DataLoader(dataset, batch_size=128, shuffle=True):

        optim.zero_grad()
        y = layer(x)
        y = layer2(y)
        loss = (y - t).pow(2).mean()

        reg = None
        for p in layer.parameters():
            cur_reg = (1 - p).pow(2).sum()
            reg = cur_reg if reg is None else reg + cur_reg
            
        for p in layer2.parameters():
            cur_reg = p.pow(2).sum()
            reg = cur_reg if reg is None else reg + cur_reg

        if regularize:
            (loss + 1e-7 * reg).backward()
        else: 
            loss.backward()
        print(loss.item())

        # if epoch % 2 == 0:
        optim.step()
        # elif (epoch + 1) % 2 == 0:
        zenkai.utils.apply_to_parameters(layer.parameters(), lambda x: x.clamp(0, 1))
        zenkai.utils.apply_to_parameters(layer2.parameters(), lambda x: x.clamp(0, 1))



# %%
