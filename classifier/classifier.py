import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class GenericModel:
    def __init__(self):
        self.is_train = True

    def train(self, is_train: bool = True):
        self.is_train = is_train


class ReferenceModel(GenericModel):
    def __init__(
        self, net_factory, optimizer_factory, lr_scheduler_factory, loss_functor
    ):
        super().__init__()

        self.net_factory = net_factory
        self.optimizer_factory = optimizer_factory
        self.lr_scheduler_factory = lr_scheduler_factory
        self.loss_functor = loss_functor

        self.model: nn.Module = net_factory()
        self.optimizer = optimizer_factory(self.model.parameters())
        self.scheduler = self.lr_scheduler_factory(self.optimizer)

    def step(self, data, target, no_grad=False, dry_run=False):
        if not no_grad:
            self.optimizer.zero_grad()
        with torch.no_grad() if no_grad else contextlib.nullcontext():
            output = self.model(data)
            loss = self.loss_functor(output, target)
        if not no_grad:
            loss.backward()
        if not dry_run and not no_grad:
            self.optimizer.step()
        return dict(loss=loss.item(), pred=output.detach())

    def get_model(self):
        return self.model

    def named_parameters(self):
        return self.model.named_parameters()

    def train(self, is_train: bool = True):
        super().train(is_train)
        self.model.train(is_train)

    def lr_scheduler_step(self):
        self.scheduler.step()
