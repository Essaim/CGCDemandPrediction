import os
import torch
import shutil
import h5py
import numpy as np
from scipy.spatial.distance import cdist
from utils.data_container import get_data_loader_base
from models.model import create_model
from utils.util import get_optimizer, get_scheduler
from utils.train import train_decompose
from utils import normalization


def preprocessing(base_model_name,
                  conf,
                  loss,
                  graph,
                  data_category,
                  device,
                  data_set,
                  optimizer_name,
                  scheduler_name):
    if base_model_name == 'LinearDecompose':
        data_loader = get_data_loader_base(base_model_name=base_model_name, dataset=conf['data']['dataset'],
                                           batch_size=conf['batch_size_base'],
                                           _len=conf['data']['_len'], data_category=data_category, device=device,
                                           Normal_Method=conf['data']['Normal_Method'])
        model, trainer = create_model(base_model_name, loss, conf['Base'][base_model_name], data_category, device,
                                      graph)
        save_folder = os.path.join('saves', f"{conf['name']}_{base_model_name}", f'{data_set}_{"".join(data_category)}')
        run_folder = os.path.join('run', f"{conf['name']}_{base_model_name}", f'{data_set}_{"".join(data_category)}')
        optimizer = get_optimizer(optimizer_name, model.parameters(), conf['optimizerbase'][optimizer_name]['lr'])
        scheduler = get_scheduler(scheduler_name, optimizer, **conf['scheduler'][scheduler_name])
        shutil.rmtree(save_folder, ignore_errors=True)
        os.makedirs(save_folder)
        shutil.rmtree(run_folder, ignore_errors=True)
        os.makedirs(run_folder)
        model = train_decompose(model=model,
                                dataloaders=data_loader,
                                trainer=trainer,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                folder=save_folder,
                                tensorboard_floder=run_folder,
                                device=device,
                                **conf['train'])
        model.load_state_dict(torch.load(f"{os.path.join(save_folder, 'best_model.pkl')}")['model_state_dict'])
        return model.encoder, model.decoder
    if base_model_name == 'SvdDecompose':
        data = get_data_loader_base(base_model_name=base_model_name, dataset=conf['data']['dataset'],
                                    batch_size=conf['batch_size_base'],
                                    _len=conf['data']['_len'], data_category=data_category, device=device,
                                    Normal_Method=conf['data']['Normal_Method'])
        data = torch.from_numpy(data).float().to(device)
        save_folder = os.path.join('saves', f"{conf['name']}_{base_model_name}", f'{data_set}_{"".join(data_category)}')
        run_folder = os.path.join('run', f"{conf['name']}_{base_model_name}", f'{data_set}_{"".join(data_category)}')
        model, trainer = create_model(base_model_name, loss, conf['Base'][base_model_name], data_category, device,
                                      graph)
        shutil.rmtree(save_folder, ignore_errors=True)
        os.makedirs(save_folder)
        shutil.rmtree(run_folder, ignore_errors=True)
        os.makedirs(run_folder)
        model.decompose(data)
        return model.encoder, model.decoder


def preprocessing_for_metric(data_category: list,
                             dataset:str,
                             method:str,
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

    support = None
    if method == 'big':
        graph = cdist(w, w, metric='euclidean')
        # support = cdist(w, w, metric='correlation')
        # support[support<=0.75] = 0
        # support[support>0.75] = 1
        # s,v,d = np.linalg.svd(graph)
        # print(v)
        support = graph * -1 / np.std(graph) ** 2
        support = np.exp(support)
        # s,v,d = np.linalg.svd(support)
        # print(support)
        # print(v)
    elif method == 'small':
        support = w
        print(w.shape)
    return support
