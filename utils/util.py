from torch import optim
import torch.nn as nn
import os
import pickle
import json
import torch
import numpy as np
import copy

def get_optimizer(opt_name,par, lr):
    if opt_name == 'Adam':
        return optim.Adam(par, lr)
    else:
        print("no optimizer")

def loss_calculate(y, y_pred, running_loss, phase,node_num, loss_func):
    bike_node,taxi_node = node_num[0],node_num[1]
    running_loss[phase][0] += loss_func(y[:,:,:bike_node],y_pred[:,:,:bike_node]) * y.size(0)
    running_loss[phase][1] += loss_func(y[:,:, bike_node:], y_pred[:,:, bike_node:]) * y.size(0)
    running_loss[phase][2] += loss_func(y, y_pred) * y.size(0)
    # print(running_loss[phase][0],running_loss[phase][1],running_loss[phase][2])
    return running_loss




def get_loss(name):
    if name == 'rmse':
        return RMSELoss()



class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, truth, predict, bias=None):
        return self.mse_loss(truth, predict) ** 0.5

class RMSE_Lasso_loss(nn.Module):
    def __init__(self, alfa):
        super(RMSE_Lasso_loss, self).__init__()
        self.mse_loss = nn.MSELoss(reduce='mean')
        self.alfa = alfa

    def forward(self, truth, predict, bias):
        out = 0
        for b in bias:
            out +=torch.sum(torch.abs(b)) * self.alfa
        return self.mse_loss(truth, predict) **0.5 + out

def save_model(path: str, **save_dict):

    os.makedirs(os.path.split(path)[0], exist_ok=True)
    torch.save(save_dict, path)

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data
def get_number_of_parameters(model: nn.Module):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def get_scheduler(name, optimizer, **kwargs):
    return getattr(optim.lr_scheduler, name)(optimizer, **kwargs)

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)