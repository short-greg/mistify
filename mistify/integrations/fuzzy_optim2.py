import torch
from .. import fuzzy
from torch.utils import data as data_utils
import itertools

from .._core import ToOptim
from ..fuzzy import MaxMinLoss, MaxProdLoss, MinMaxLoss
from torch.nn.utils import parameters_to_vector


def check_if_maxmin_optimizes_x():
    torch.manual_seed(1)
    maxmin = fuzzy.MaxMin(8, 4)
    maxmin.weight.data = fuzzy.rand(*maxmin.weight.data.size())
    x_in = fuzzy.rand(512, 8)
    t = maxmin.forward(x_in)
    x_update = fuzzy.rand(512, 8)

    x_update = x_update.requires_grad_(True)
    dataset = data_utils.TensorDataset(x_update, t)
    optimizer = torch.optim.SGD([x_update], lr=1e0, weight_decay=0.0)
    loss = MaxMinLoss(maxmin, reduction='none', default_optim=ToOptim.X)
    mse = torch.nn.MSELoss()

    for i in range(100):
        data_loader = data_utils.DataLoader(dataset, batch_size=128, shuffle=True)

        for x_i, t_i in data_loader:
            optimizer.zero_grad()
            y = maxmin.forward(x_i)
            result = loss.forward(x_i, y, t_i).sum() / len(x_i)
            result.backward()
            optimizer.step()
            print(i, mse.forward(maxmin(x_update), t).item())

# Removed because not using LossGrad atm until i figure out how to simplify it

# def check_if_maxmin_optimizes_x_with_two_layers():
#     torch.manual_seed(1)
#     maxmin = fuzzy.MaxMin(8, 4)
#     maxmin2 = fuzzy.MaxMin(4, 6)
#     maxmin.weight.data = fuzzy.rand(*maxmin.weight.data.size())
#     maxmin2.weight.data = fuzzy.rand(*maxmin2.weight.data.size())
#     x_in = fuzzy.rand(512, 8)
#     t = maxmin2.forward(maxmin.forward(x_in))
#     x_update = fuzzy.rand(512, 8)

#     x_update = x_update.requires_grad_(True)
#     dataset = data_utils.TensorDataset(x_update, t)
#     optimizer = torch.optim.SGD([x_update], lr=1e0, weight_decay=0.0)

#     loss1 = MaxMinLoss(maxmin, reduction='sum', default_optim=ToOptim.X)
#     loss2 = MaxMinLoss(maxmin2, reduction='none', default_optim=ToOptim.X)
#     mse = torch.nn.MSELoss()

#     for i in range(1000):
#         data_loader = data_utils.DataLoader(dataset, batch_size=128, shuffle=True)

#         for x_i, t_i in data_loader:
#             optimizer.zero_grad()
#             x_i = LossGrad.apply(x_i, loss1)
#             y = maxmin2.forward(x_i)
#             result = loss2.forward(x_i, y, t_i).sum() / len(x_i)
#             result.backward()
#             optimizer.step()
#             print(i, mse.forward(maxmin2(maxmin(x_update)), t).item())

# Removed because not using LossGrad atm until i figure out how to simplify it

# def check_if_maxmin_optimizes_theta_with_two_layers():
#     torch.manual_seed(1)
#     maxmin = fuzzy.MaxMin(8, 4)
#     maxmin2 = fuzzy.MaxMin(4, 6)
#     maxmin.weight.data = fuzzy.rand(*maxmin.weight.data.size())
#     maxmin2.weight.data = fuzzy.rand(*maxmin2.weight.data.size())
#     x = fuzzy.rand(512, 8)
#     t = maxmin2(maxmin.forward(x))
#     maxmin.weight.data = fuzzy.rand(*maxmin.weight.data.size())
#     maxmin2.weight.data = fuzzy.rand(*maxmin2.weight.data.size())

#     # x_update = fuzzy.rand(512, 8)

#     dataset = data_utils.TensorDataset(x, t)
#     optimizer = torch.optim.SGD(itertools.chain(maxmin.parameters()), lr=1e-1, weight_decay=0.0)
#     optimizer2 = torch.optim.SGD(itertools.chain(maxmin2.parameters()), lr=1e-1, weight_decay=0.0)

#     loss1 = MaxMinLoss(maxmin, reduction='batchmean', default_optim=ToOptim.BOTH, not_chosen_theta_weight=1)
#     loss2 = MaxMinLoss(maxmin2, reduction='batchmean', default_optim=ToOptim.BOTH, not_chosen_theta_weight=1, not_chosen_x_weight=1)
#     mse = torch.nn.MSELoss()

#     count = 0
#     for i in range(1000):
#         data_loader = data_utils.DataLoader(dataset, batch_size=128, shuffle=True)

#         optim_first = True
#         for j, (x_i, t_i) in enumerate(data_loader):
#             optimizer.zero_grad()
#             optimizer2.zero_grad()
#             # x_i.requires_grad_()
#             x_i = losses.LossGrad.apply(x_i, loss1)
#             y = maxmin2(x_i)
#             result = loss2(x_i, y, t_i) # .sum() / len(x_i)
#             result.backward()
#             if optim_first:
#                 before = parameters_to_vector(maxmin.parameters())
#                 optimizer.step()
#                 after = parameters_to_vector(maxmin.parameters())
#                 print((after - before).abs().sum())
#             #     optimizer.step()
#             #     # 
#             #     # print((after - before).abs().sum())

#             else:
#                 optimizer2.step()
#             #     # pass
#             maxmin2.clamp()
#             maxmin.clamp()
#             count += 1
#             if count == 10:
#                 optim_first = not optim_first
#                 count = 0
#             # print(y[0:5], t[0:5])
#             print(i, mse.forward(maxmin2(maxmin(x)), t).item())
