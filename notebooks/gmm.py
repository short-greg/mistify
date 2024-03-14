# %%

import initialize
import torch
import torch.utils.data as torch_data
from mistify import fuzzify

# %%

def gen_data(n: int, features: int, dist: int):

    n_per_dist = torch.softmax(
        torch.randn(dist), dim=0
    ) * n
    data = []

    means = []
    stds = []

    for i in range(dist):
        print(i)
        mean = torch.randn(1, features) * 2
        std = torch.rand(1, features) 
        data.append(
            torch.randn(
                int(n_per_dist[i].item()), features
            ) 
            * std
            + mean
        )
        means.append(mean)
        stds.append(std)
    return torch.cat(data), torch.cat(means), torch.cat(stds)



X, means, stds = gen_data(2000, 4, 4)

gaussian = fuzzify.GaussianFuzzifier(4, 4, tunable=True)


dataset = torch_data.TensorDataset(X)

optim = torch.optim.Adam(gaussian.parameters(), lr=1e-3)

for x_i, in torch_data.DataLoader(dataset, batch_size=128, shuffle=True):

    optim.zero_grad()
    gaussian.resp_loss(x_i)
    optim.step()

# %%
    

print(gaussian._loc)
print(gaussian._scale)

# %%
