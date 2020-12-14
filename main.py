import argparse
import json
import os
import shutil

import h5py
import torch
import yaml
import torch.nn as nn
import torch

from models import create_model
from utils.train import train_model, train_baseline, test_model, test_baseline
from utils.util import get_optimizer, get_loss, get_scheduler
from utils.data_container import get_data_loader, get_data_loader_base
from utils.preprocess import preprocessing, preprocessing_for_metric


# 250， 266
def train(conf, data_category):
    print(json.dumps(conf, indent=4))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(conf['device'])
    device = torch.device(0)

    model_name = conf['model']['name']
    optimizer_name = conf['optimizer']['name']
    data_set = conf['data']['dataset']
    graph = h5py.File(os.path.join('data', data_set, 'all_graph.h5'), 'r')
    scheduler_name = conf['scheduler']['name']
    loss = get_loss(**conf['loss'])
    # data_category = conf['data']['data_category']

    loss.to(device)
    encoder, decoder, support = None, None, None
    if model_name == 'Costnet':
        base_model_name = conf['Base']['name']
        encoder, decoder = preprocessing(base_model_name, conf, loss, graph, data_category, device, data_set,
                                         optimizer_name, scheduler_name)
    if model_name == 'Metricnet' or model_name == 'GWNET' or model_name == 'Evonet' or model_name == 'STGCN' or model_name == 'DCRNN' or model_name == 'STG2Seq' or model_name == 'Evonet2':
        support = preprocessing_for_metric(data_category=data_category, dataset=conf['data']['dataset'],
                                           Normal_Method=conf['data']['Normal_Method'], _len=conf['data']['_len'], **conf['preprocess'])
    model, trainer = create_model(model_name,
                                  loss,
                                  conf['model'][model_name],
                                  data_category,
                                  device,
                                  graph,
                                  encoder,
                                  decoder,
                                  support)

    optimizer = get_optimizer(optimizer_name, model.parameters(), conf['optimizer'][optimizer_name]['lr'])
    scheduler = get_scheduler(scheduler_name, optimizer, **conf['scheduler'][scheduler_name])
    if torch.cuda.device_count() > 1:
        print("use ", torch.cuda.device_count(), "GPUS")
        model = nn.DataParallel(model)
    else:
        model.to(device)

    save_folder = os.path.join('save', conf['name'], f'{data_set}_{"".join(data_category)}', conf['tag'])
    run_folder = os.path.join('run', conf['name'], f'{data_set}_{"".join(data_category)}', conf['tag'])

    shutil.rmtree(save_folder, ignore_errors=True)
    os.makedirs(save_folder)
    shutil.rmtree(run_folder, ignore_errors=True)
    os.makedirs(run_folder)

    with open(os.path.join(save_folder, 'config.yaml'), 'w+') as _f:
        yaml.safe_dump(conf, _f)

    data_loader, normal = get_data_loader(**conf['data'], data_category=data_category, device=device,
                                          model_name=model_name)

    if len(data_category) == 2:
        train_model(model=model,
                    dataloaders=data_loader,
                    trainer=trainer,
                    node_num=conf['node_num'],
                    loss_func=loss,
                    optimizer=optimizer,
                    normal=normal,
                    scheduler=scheduler,
                    folder=save_folder,
                    tensorboard_folder=run_folder,
                    device=device,
                    **conf['train'])
        # test_model(folder = save_folder)
    else:
        train_baseline(model=model,
                       dataloaders=data_loader,
                       trainer=trainer,
                       optimizer=optimizer,
                       normal=normal,
                       scheduler=scheduler,
                       folder=save_folder,
                       tensorboard_folder=run_folder,
                       device=device,
                       **conf['train'])
        test_baseline(folder=save_folder,
                      trainer=trainer,
                      model=model,
                      normal=normal,
                      dataloaders=data_loader,
                      device=device)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config', required=True, type=str,
    #                     help='Configuration filename for restoring the model.')
    # parser.add_argument('--resume', required=False, type=bool, default=False,
    #                     help='Resume.')
    # parser.add_argument('--test', required=False, type=bool, default=False,
    #                     help='Test.')
    #
    # args = parser.parse_args()
    con = 'evoconv2-config'
    data = ['taxi']
    with open(os.path.join('config', f'{con}.yaml')) as f:
        conf = yaml.safe_load(f)
    train(conf, data)
