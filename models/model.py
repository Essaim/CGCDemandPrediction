from typing import Tuple, List

import numpy as np
from .evonn2 import EvoNN2
import torch
from torch import nn, Tensor
from utils.graph import sparse_scipy2torch, load_graph_data, cheb_poly_approx, normalized_laplacian, random_walk_matrix


def graph_preprocess(model_name, graph_h5, support, data_category=None, graph_category=None, normalized_category=None,
                     device=None):
    if model_name == 'H_GAT':
        pass
    else:
        if data_category[0] == 'taxi':
            if graph_category == 'pcc':
                matrix = graph_h5['pcc_tt'][:]
                matrix[matrix < 0.75] = 0
                matrix[matrix >= 0.75] = 1
            elif graph_category == 'dis':
                matrix = graph_h5['dis_tt'][:]
            elif graph_category == 'gau':
                matrix = support
            else:
                raise KeyError()
        elif data_category[0] == 'bike':
            if graph_category == 'pcc':
                matrix = graph_h5['pcc_bb'][:]
                matrix[matrix < 0.75] = 0
                matrix[matrix >= 0.75] = 1
            elif graph_category == 'dis':
                matrix = graph_h5['dis_bb'][:]
            elif graph_category == 'gau':
                matrix = support
            else:
                raise KeyError()
        matrix = matrix - np.identity(matrix.shape[0])
        if normalized_category == 'randomwalk':
            matrix = random_walk_matrix(matrix)
        elif normalized_category == 'laplacian':
            matrix = normalized_laplacian(matrix)
        else:
            raise KeyError()
        return matrix



def create_model(model_name, loss, conf, data_category, device, graph_h5, encoder=None, decoder=None, support=None):
    normalized_category, graph_category = conf.pop('normalized_category'), conf.pop('graph_category')
    graph = graph_preprocess(model_name, graph_h5, support, data_category, graph_category, normalized_category, device)
    graph_h5.close()

    if model_name == 'Evonet2':
        model = EvoNN2(**conf, support=torch.from_numpy(graph).float(), device=device)
        for name, parameters in model.named_parameters():
            print(name, ':', parameters.size())
        return model, MetricNNTrainer(model, loss)




class Trainer:
    def __init__(self, model: nn.Module, loss):
        self.model = model
        self.loss = loss

    def train(self, inputs: Tensor, targets: Tensor, phase: str) -> Tuple[Tensor, Tensor]:
        raise ValueError('Not implemented.')


class MetricNNTrainer(Trainer):
    def __init__(self, model, loss):
        super(MetricNNTrainer, self).__init__(model, loss)
        self.train_batch_seen: int = 0

    def train(self, inputs: Tensor, targets: Tensor, phase: str):
        if phase == 'train':
            self.train_batch_seen += 1
        i_targets = targets if phase == 'train' else None
        outputs, graph = self.model(inputs, i_targets, self.train_batch_seen)
        loss = self.loss(outputs, targets, graph)
        # loss = self.loss(outputs, targets)
        return outputs, loss

