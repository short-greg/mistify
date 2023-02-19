from .. import fuzzy
from .. import optim
import torch


def check_if_output_is_valid():
    maxmin = fuzzy.MaxMin(2, 4)
    loss = optim.MaxMinLoss(maxmin)
    x = fuzzy.FuzzySet.rand(4, 2)
    t = fuzzy.FuzzySet.rand(4, 4)
    y = maxmin.forward(x)
    result = loss.forward(x, y, t)
    print(result)

from torch.nn.utils import parameters_to_vector
from torch.utils import data as data_utils

def check_if_optimizes_theta():
    torch.manual_seed(1)
    maxmin = fuzzy.MaxMin(8, 4)
    maxmin.weight = fuzzy.FuzzySetParam(
        fuzzy.FuzzySet.rand(*maxmin.weight.data.size(), is_batch=False)
    )
    maxprod = fuzzy.MaxProd(8, 4)
    maxprod.weight = fuzzy.FuzzySetParam(
        fuzzy.FuzzySet.rand(*maxprod.weight.data.size(), is_batch=False)
    )
    maxmin_train = fuzzy.MaxMin(8, 4)
    maxmin_train.weight = fuzzy.FuzzySetParam(
        fuzzy.FuzzySet.rand(*maxmin.weight.data.size(), is_batch=False)
    )


    maxmin_train2 = fuzzy.MaxMin(8, 4)
    maxprod_train = fuzzy.MaxProd(8, 4)
    maxprod_train.weight = fuzzy.FuzzySetParam(
        fuzzy.FuzzySet.rand(*maxprod.weight.data.size(), is_batch=False)
    )
    # maxmin_train2.weight = fuzzy.FuzzySetParam(
    #     fuzzy.FuzzySet.rand(*maxmin.weight.data.size(), is_batch=False)
    # )
    loss = optim.MaxMinLoss2(maxmin_train, default_optim=optim.ToOptim.THETA)
    loss2 = optim.MaxProdLoss2(maxprod_train, default_optim=optim.ToOptim.THETA)
    x = fuzzy.FuzzySet.rand(128, 8)
    
    t = maxprod.forward(x)

    dataset = data_utils.TensorDataset(x.data, t.data)
    optimizer = torch.optim.SGD(maxmin_train.parameters(), lr=1e0, weight_decay=0.0)
    optimizer2 = torch.optim.SGD(maxmin_train2.parameters(), lr=1e0, weight_decay=0.0)
    optimizer3 = torch.optim.SGD(maxprod_train.parameters(), lr=1e0,  weight_decay=0.0)
    for i in range(10000):
        data_loader = data_utils.DataLoader(dataset, batch_size=128, shuffle=True)

        for x_i, t_i in data_loader:
            x_i = fuzzy.FuzzySet(x_i, True)
            t_i = fuzzy.FuzzySet(t_i, True)
            optimizer.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            y = maxmin_train.forward(x_i)
            y2 = maxmin_train2.forward(x_i)
            y3 = maxprod_train.forward(x_i)

            print('?', (x.data < 0).any())
            # t = fuzzy.FuzzySet.rand(4, 4)
            result = loss.forward(x_i, y, t_i)
            result2 = ((y2.data - t_i.data.detach()) ** 2).mean()
            result3 = loss2.forward(x_i, y3, t_i)

            # before = parameters_to_vector(maxmin_train.parameters())
            result.backward()
            result2.backward()
            result3.backward()
            error = ((y.data - t_i.data).abs()).mean()
            error2 = ((y2.data - t_i.data).abs()).mean()
            error3 = ((y3.data - t_i.data).abs()).mean()
            # print('---', x.data, y.data, t.data)
            print('Output Error: ', error.item())
            print('Output Error2: ', error2.item())
            print('Output Error3: ', error3.item())
            # print('Error: ', result.item())
            
            optimizer.step()
            optimizer2.step()
            optimizer3.step()

            # loss2.clamp()
            if (i + 1) % 50 == 0:
                loss.not_chosen_weight *= 0.5
                loss2.not_chosen_weight *= 0.5
            # after = parameters_to_vector(maxmin_train.parameters())
            # print('Change: ',(before - after).abs().mean().item())
        # print('X: ', x.data) 
        #print('Y: ', y.data)
        # print('T: ', t.data) 
        # print('Error: ', (y.data - t.data).abs()) 
        # print('W: ', maxmin_train.weight.data[:,2]) 
        # print('WT: ', maxmin.weight.data[:,2])
        # for p in maxmin_train.parameters():
        #     print(p)
