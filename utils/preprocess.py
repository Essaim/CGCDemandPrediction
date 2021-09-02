import os
import torch
import shutil
import h5py
import numpy as np
from scipy.spatial.distance import cdist
from utils import normalization


def preprocessing_for_metric(data_category: list,
                             dataset:str,
                             hidden_size:int,
                             Normal_Method: str,
                             _len: list):
    data = []
    normal_method = getattr(normalization, Normal_Method)
    for category in data_category:
        normal = normal_method()
        with h5py.File(f"data/{dataset}/{category}_data.h5", 'r') as hf:
            data_pick = hf[f'{category}_pick'][:]
        with h5py.File(f"data/{dataset}/{category}_data.h5", 'r') as hf:
            data_drop = hf[f'{category}_drop'][:]
        data.append(normal.fit_transform(np.stack([data_pick, data_drop], axis=2)))


    data = np.concatenate(data, axis=1).transpose((0,2,1))
    data = data[:-(_len[0]+_len[1])]
    T, input_dim, N = data.shape
    inputs = data.reshape(-1, N)
    u, s, v = np.linalg.svd(inputs)
    w = np.diag(s[:hidden_size]).dot(v[:hidden_size,:]).T



    graph = cdist(w, w, metric='euclidean')
    support = graph * -1 / np.std(graph) ** 2
    support = np.exp(support)

    return support
