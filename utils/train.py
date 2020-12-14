import os
import numpy as np
import math
import json
import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from tqdm import tqdm
import copy, time
from utils.util import save_model, get_number_of_parameters
from collections import defaultdict
from utils.util import loss_calculate
from utils.evaluate import evaluate
from utils.util import MyEncoder


def train_model(model: nn.Module,
                dataloaders,
                optimizer,
                normal,
                scheduler,
                folder: str,
                trainer,
                node_num: list,
                loss_func,
                tensorboard_folder,
                epochs: int,
                device,
                max_grad_norm: float = None,
                early_stop_steps: float = None):
    normal_bike, normal_taxi = normal[0], normal[1]

    save_path = os.path.join(folder, 'best_model.pkl')

    if os.path.exists(save_path):
        save_dict = torch.load(save_path)

        model.load_state_dict(save_dict['model_state_dict'])
        optimizer.load_state_dict(save_dict['optimizer_state_dict'])

        best_val_loss = save_dict['best_val_loss']
        begin_epoch = save_dict['epoch'] + 1
    else:
        save_dict = dict()
        best_val_loss = float('inf')
        begin_epoch = 0

    phases = ['train', 'validate', 'test']

    writer = SummaryWriter(tensorboard_folder)

    since = time.perf_counter()

    model = model.to(device)
    print(model)
    print(f'Trainable parameters: {get_number_of_parameters(model)}.')

    try:
        for epoch in range(begin_epoch, begin_epoch + epochs):

            running_loss, running_metrics = {phase: [0.0 for i in range(3)] for phase in phases}, {
                phase: [0.0 for i in range(2)] for phase in phases}
            for phase in phases:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                steps, predictions, running_targets = 0, list(), list()
                tqdm_loader = tqdm(enumerate(dataloaders[phase]))
                for step, (inputs, targets) in tqdm_loader:
                    running_targets.append(targets.numpy())

                    with torch.no_grad():
                        # inputs[..., 0] = scaler.transform(inputs[..., 0])
                        inputs = inputs.to(device)
                        # targets[..., 0] = scaler.transform(targets[..., 0])
                        targets = targets.to(device)

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs, loss = trainer.train(inputs, targets, phase)

                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            if max_grad_norm is not None:
                                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                            optimizer.step()

                    with torch.no_grad():
                        # predictions.append(scaler.inverse_transform(outputs).cpu().numpy())
                        predictions.append(outputs.cpu().numpy())

                    running_loss  = loss_calculate(outputs, targets, running_loss, phase, node_num, loss_func)
                    steps += len(targets)

                    tqdm_loader.set_description(
                        f'{phase:5} epoch: {epoch:3}, {phase:5} loss: {normal_bike.rmse_transform(running_loss[phase][2] / steps):3.6}  '
                        f'{normal_taxi.rmse_transform(running_loss[phase][1] / steps):3.6}  '
                        f'{normal_bike.rmse_transform(running_loss[phase][0] / steps):3.6}')

                    # For the issue that the CPU memory increases while training. DO NOT know why, but it works.
                    torch.cuda.empty_cache()
                # 性能
                # running_metrics[phase] = nomask_evaluate(np.concatenate(predictions), np.concatenate(running_targets))

                if phase == 'validate':
                    if running_loss['validate'][2] <= best_val_loss:
                        best_val_loss = running_loss['validate'][2]
                        save_dict.update(model_state_dict=copy.deepcopy(model.state_dict()),
                                         epoch=epoch,
                                         best_val_loss=best_val_loss,
                                         optimizer_state_dict=copy.deepcopy(optimizer.state_dict()))
                        save_model(save_path, **save_dict)
                        print(f'Better model at epoch {epoch} recorded.')
                    elif epoch - save_dict['epoch'] > early_stop_steps:
                        raise ValueError('Early stopped.')

            scheduler.step(running_loss['train'][2])

            # for metric in running_metrics['train'].keys():
            #     for phase in phases:
            #         for key, val in running_metrics[phase][metric].items():
            #             writer.add_scalars(f'{metric}/{key}', {f'{phase}': val}, global_step=epoch)
            # writer.add_scalars('Loss', {
            #     f'{phase} loss': running_loss[phase] / len(dataloaders[phase].dataset) for phase in phases},
            #                    global_step=epoch)
    except (ValueError, KeyboardInterrupt):
        time_elapsed = time.perf_counter() - since
        print(f"cost {time_elapsed} seconds")
        print(f'model of epoch {save_dict["epoch"]} successfully saved at `{save_path}`')


def train_baseline(model: nn.Module,
                   dataloaders,
                   optimizer,
                   normal,
                   scheduler,
                   folder: str,
                   trainer,
                   tensorboard_folder,
                   epochs: int,
                   device,
                   max_grad_norm: float = None,
                   early_stop_steps: float = None):

    # normal = normal[0]
    save_path = os.path.join(folder, 'best_model.pkl')

    if os.path.exists(save_path):
        print("path exist")
        save_dict = torch.load(save_path)

        model.load_state_dict(save_dict['model_state_dict'])
        optimizer.load_state_dict(save_dict['optimizer_state_dict'])

        best_val_loss = save_dict['best_val_loss']
        begin_epoch = save_dict['epoch'] + 1
    else:
        print("path does not exist")
        save_dict = dict()
        best_val_loss = float('inf')
        begin_epoch = 0

    phases = ['train', 'validate', 'test']

    writer = SummaryWriter(tensorboard_folder)

    since = time.perf_counter()

    model = model.to(device)
    print(model)
    print(f'Trainable parameters: {get_number_of_parameters(model)}.')

    try:
        for epoch in range(begin_epoch, begin_epoch + epochs):

            running_loss, running_metrics = defaultdict(float), dict()
            for phase in phases:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                steps, predictions, running_targets = 0, list(), list()
                tqdm_loader = tqdm(enumerate(dataloaders[phase]))
                for step, (inputs, targets) in tqdm_loader:
                    running_targets.append(targets.numpy())

                    with torch.no_grad():
                        inputs = inputs.to(device)
                        targets = targets.to(device)

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs, loss = trainer.train(inputs, targets, phase)

                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            if max_grad_norm is not None:
                                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                            optimizer.step()

                    with torch.no_grad():
                        predictions.append(outputs.cpu().numpy())

                    running_loss[phase] += loss * len(targets)
                    steps += len(targets)

                    tqdm_loader.set_description(
                        f'{phase:5} epoch: {epoch:3}, {phase:5} loss: {normal[0].rmse_transform(running_loss[phase] / steps):3.6}')
                    # tqdm_loader.set_description(
                    #     f'{phase:5} epoch: {epoch:3}, {phase:5} loss: {running_loss[phase] / steps:3.6}')

                    # For the issue that the CPU memory increases while training. DO NOT know why, but it works.
                    torch.cuda.empty_cache()
                # 性能
                running_metrics[phase] = evaluate(np.concatenate(predictions), np.concatenate(running_targets), normal)
                # print(running_metrics[phase]['loss'])
                running_metrics[phase].pop('loss')
                running_metrics[phase].pop('pcc')
                running_metrics[phase].pop('mae')


                if phase == 'validate':
                    if running_loss['validate'] <= best_val_loss or math.isnan(running_loss['validate']):
                        best_val_loss = running_loss['validate']
                        save_dict.update(model_state_dict=copy.deepcopy(model.state_dict()),
                                         epoch=epoch,
                                         best_val_loss=best_val_loss,
                                         optimizer_state_dict=copy.deepcopy(optimizer.state_dict()))
                        save_model(save_path, **save_dict)
                        print(f'Better model at epoch {epoch} recorded.')
                    elif epoch - save_dict['epoch'] > early_stop_steps:
                        raise ValueError('Early stopped.')

            scheduler.step(running_loss['train'])

            for metric in running_metrics['train'].keys():
                for phase in phases:
                    for key, val in running_metrics[phase][metric].items():
                        writer.add_scalars(f'{metric}/{key}', {f'{phase}': val}, global_step=epoch)
            writer.add_scalars('Loss', {
                f'{phase} loss': running_loss[phase] / len(dataloaders[phase].dataset) for phase in phases},
                               global_step=epoch)
    except (ValueError, KeyboardInterrupt):
        writer.close()
        time_elapsed = time.perf_counter() - since
        print(f"cost {time_elapsed} seconds")
        print(f'model of epoch {save_dict["epoch"]} successfully saved at `{save_path}`')



def train_decompose(model: nn.Module,
                    dataloaders,
                    optimizer,
                    scheduler,
                    folder: str,
                    trainer,
                    tensorboard_floder,
                    epochs: int,
                    device,
                    max_grad_norm: float = None,
                    early_stop_steps: float = None):


    save_path = os.path.join(folder, 'best_model.pkl')

    if os.path.exists(save_path):
        save_dict = torch.load(save_path)

        model.load_state_dict(save_dict['model_state_dict'])
        optimizer.load_state_dict(save_dict['optimizer_state_dict'])

        best_val_loss = save_dict['best_val_loss']
        begin_epoch = save_dict['epoch'] + 1
    else:
        save_dict = dict()
        best_val_loss = float('inf')
        begin_epoch = 0

    phases = ['train', 'validate', 'test']

    writer = SummaryWriter(tensorboard_floder)

    since = time.perf_counter()

    model = model.to(device)

    print(model)
    print(f'Trainable parameters: {get_number_of_parameters(model)}.')

    try:
        for epoch in range(begin_epoch, begin_epoch + epochs):

            running_loss, running_metrics = defaultdict(float), dict()
            for phase in phases:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                steps, predictions, running_targets = 0, list(), list()
                tqdm_loader = tqdm(enumerate(dataloaders[phase]))
                for step, (inputs, targets) in tqdm_loader:
                    running_targets.append(targets.numpy())

                    with torch.no_grad():
                        # inputs[..., 0] = scaler.transform(inputs[..., 0])
                        inputs = inputs.to(device)
                        # targets[..., 0] = scaler.transform(targets[..., 0])
                        targets = targets.to(device)

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs, loss = trainer.train(inputs, targets, phase)

                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            if max_grad_norm is not None:
                                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                            optimizer.step()

                    with torch.no_grad():
                        predictions.append(outputs.cpu().numpy())
                    running_loss[phase] += loss * len(targets)
                    steps += len(targets)

                    tqdm_loader.set_description(
                        f'{phase:5} epoch: {epoch:3}, {phase:5} loss: {running_loss[phase] / steps:3.6}')

                    # For the issue that the CPU memory increases while training. DO NOT know why, but it works.
                    torch.cuda.empty_cache()
                # 性能
                # running_metrics[phase] = trainer.loss(torch.cat(predictions), torch.cat(running_targets)).cpu().numpy()

                if phase == 'validate':
                    if running_loss['validate'] < best_val_loss:
                        best_val_loss = running_loss['validate']
                        save_dict.update(model_state_dict=copy.deepcopy(model.state_dict()),
                                         epoch=epoch,
                                         best_val_loss=best_val_loss,
                                         optimizer_state_dict=copy.deepcopy(optimizer.state_dict()))
                        save_model(save_path, **save_dict)
                        print(f'Better model at epoch {epoch} recorded.')
                    elif epoch - save_dict['epoch'] > early_stop_steps:
                        raise ValueError('Early stopped.')

            scheduler.step(running_loss['train'])

            # for metric in running_metrics['train'].keys():
            #     for phase in phases:
            #         for key, val in running_metrics[phase][metric].items():
            #             writer.add_scalars(f'{metric}/{key}', {f'{phase}': val}, global_step=epoch)
            # writer.add_scalars('Loss', {
            #     f'{phase} loss': running_loss[phase] / len(dataloaders[phase].dataset) for phase in phases},
            #                    global_step=epoch)
    except (ValueError, KeyboardInterrupt):
        time_elapsed = time.perf_counter() - since
        print(f"cost {time_elapsed} seconds")
        model.load_state_dict(save_dict['model_state_dict'])
        print(f'model of epoch {save_dict["epoch"]} successfully saved at `{save_path}`')
    return model


def test_model():
    pass



def test_baseline(folder: str,
                  trainer,
                  model,
                  normal,
                  dataloaders,
                  device):

    save_path = os.path.join(folder, 'best_model.pkl')
    save_dict = torch.load(save_path)
    # model.load_state_dict(save_dict['model_state_dict'])

    # model.eval()
    steps, predictions, running_targets = 0, list(), list()
    tqdm_loader = tqdm(enumerate(dataloaders['test']))
    for step, (inputs, targets) in tqdm_loader:
        running_targets.append(targets.numpy())

        with torch.no_grad():
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs, loss = trainer.train(inputs, targets, 'test')
            predictions.append(outputs.cpu().numpy())

    running_targets, predictions = np.concatenate(running_targets, axis=0), np.concatenate(predictions, axis=0)

    scores = evaluate(running_targets, predictions,normal)

    print('test results:')
    print(json.dumps(scores,cls=MyEncoder, indent=4))
    with open(os.path.join(folder, 'test-scores.json'), 'w+') as f:
        json.dump(scores, f,cls=MyEncoder, indent=4)
    if trainer.model.graph0 is not None:
        np.save(os.path.join(folder, 'graph0'),trainer.model.graph0.detach().cpu().numpy())
        np.save(os.path.join(folder, 'graph1'),trainer.model.graph1.detach().cpu().numpy())
        np.save(os.path.join(folder, 'graph2'),trainer.model.graph2.detach().cpu().numpy())

    np.savez(os.path.join(folder, 'test-results.npz'), predictions=predictions, targets=running_targets)

