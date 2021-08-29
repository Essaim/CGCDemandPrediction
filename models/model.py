from typing import Tuple
from .evonn2 import EvoNN2
import torch
from torch import nn, Tensor



def create_model(model_name, loss, conf, data_category, device,  support=None):
    if model_name == 'Evonet2':
        model = EvoNN2(**conf, support=torch.from_numpy(support).float(), device=device)
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

