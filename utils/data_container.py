import pandas as pd
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import utils
from utils import normalization


class traffic_demand_prediction_dataset(Dataset):
    def __init__(self, x, y, key, val_len, test_len):
        self.x = x
        self.y = y
        self.key = key
        self._len = {"train_len": x.shape[0] - val_len - test_len,
                     "validate_len": val_len, "test_len": test_len}

    def __getitem__(self, item):
        if self.key == 'train':
            return self.x[item], self.y[item]
        elif self.key == 'validate':
            return self.x[self._len["train_len"] + item], self.y[self._len["train_len"] + item]
        elif self.key == 'test':
            return self.x[-self._len["test_len"] + item], self.y[-self._len["test_len"] + item]
        else:
            raise NotImplementedError()

    def __len__(self):
        return self._len[f"{self.key}_len"]


def get_data_loader(data_category: list,
                    batch_size: int,

                    X_list: list,
                    Y_list: list,
                    _len: list,
                    dataset,
                    device,
                    model_name: str,
                    Normal_Method: str):
    val_len, test_len = _len[0], _len[1]

    data, normal = list(), list()
    # if(Normal_Method == 'Nonormal'):
    #     for category in data_category:
    #         with h5py.File(f"data/{dataset}/{category}_data.h5", 'r') as hf:
    #             data_pick = hf[f'{category}_pick'][:]
    #         with h5py.File(f"data/{dataset}/{category}_data.h5", 'r') as hf:
    #             data_drop = hf[f'{category}_drop'][:]
    #         data.append(np.stack([data_pick, data_drop], axis=2))
    #     data = np.concatenate(data, axis=1)
    # else:
    normal_method = getattr(normalization, Normal_Method)

    for i,category in enumerate(data_category):
        normal.append(normal_method())
        print()
        with h5py.File(f"data/{dataset}/{category}_data.h5", 'r') as hf:
            data_pick = hf[f'{category}_pick'][:]
        with h5py.File(f"data/{dataset}/{category}_data.h5", 'r') as hf:
            data_drop = hf[f'{category}_drop'][:]
        data.append(normal[i].fit_transform(np.stack([data_pick, data_drop], axis=2)))
        # data.append(np.stack([data_pick, data_drop], axis=2))
    data = np.concatenate(data, axis=1)



    X_, Y_ = list(), list()
    for i in range(max(X_list), data.shape[0] - max(Y_list)):
        X_.append([data[i - j] for j in X_list])
        Y_.append([data[i + j] for j in Y_list])
    X_ = torch.from_numpy(np.asarray(X_)).float()
    Y_ = torch.from_numpy(np.asarray(Y_)).float()
    dls = dict()


    for key in ['train', 'validate', 'test']:
        dataset = traffic_demand_prediction_dataset(X_, Y_, key, val_len, test_len)
        dls[key] = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size, num_workers=16)
    return dls, normal


def get_data_loader_base(base_model_name,
                         data_category: list,
                         batch_size: int,
                         _len: list,
                         dataset,
                         device,
                         Normal_Method):
    val_len, test_len = _len[0], _len[1]



    if base_model_name == 'LinearDecompose':
        data = []
        normal_method = getattr(normalization,Normal_Method)
        for category in data_category:
            normal = normal_method()
            with h5py.File(f"data/{dataset}/{category}_data.h5", 'r') as hf:
                data_pick = hf[f'{category}_pick'][:]
            with h5py.File(f"data/{dataset}/{category}_data.h5", 'r') as hf:
                data_drop = hf[f'{category}_drop'][:]
            data.append(normal.fit_transform(np.stack([data_pick, data_drop], axis=2)))
            # data.append(np.stack([data_pick, data_drop], axis=2))
        data = np.concatenate(data, axis=1)
        print(data.shape)


        data = torch.from_numpy(np.asarray(data)).float()
        dls = dict()

        for key in ['train', 'validate', 'test']:
            dataset = traffic_demand_prediction_dataset(data, data, key, val_len, test_len)
            dls[key] = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size, num_workers=16)
        return dls
    elif base_model_name == 'SvdDecompose':
        data = []
        normal_method = getattr(normalization, Normal_Method)
        for category in data_category:
            normal = normal_method()
            with h5py.File(f"data/{dataset}/{category}_data.h5", 'r') as hf:
                data_pick = hf[f'{category}_pick'][:]
            with h5py.File(f"data/{dataset}/{category}_data.h5", 'r') as hf:
                data_drop = hf[f'{category}_drop'][:]
            data.append(normal.fit_transform(np.stack([data_pick, data_drop], axis=2)))

        data = np.concatenate(data, axis=1)
        return data
    else:
        raise ValueError("No such base_model_name")