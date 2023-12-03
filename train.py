import logging
import os
from functools import partial

import hydra
import mlflow
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from classifier.classifier import Net, ReferenceModel

logger = logging.getLogger(__name__)
mlflow.set_tracking_uri("http://localhost:13416/")


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

        use_cuda = not cfg.no_cuda and torch.cuda.is_available()

        torch.manual_seed(cfg.seed)

        self.device = torch.device("cuda" if use_cuda else "cpu")

        train_kwargs = {"batch_size": cfg.batch_size}
        test_kwargs = {"batch_size": cfg.test_batch_size}
        if use_cuda:
            cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
            train_kwargs.update(cuda_kwargs)
            test_kwargs.update(cuda_kwargs)

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.dataset_train = datasets.MNIST(
            "../data", train=False, download=True, transform=transform
        )
        self.dataset_val = datasets.MNIST("../data", train=False, transform=transform)

        shrink_dataset = True
        if shrink_dataset:
            train_data_size = 2000
            val_data_size = 1000

            def _shrink_dataset(dataset, size):
                dataset.data = dataset.data[:size]
                dataset.targets = dataset.targets[:size]

            _shrink_dataset(self.dataset_train, train_data_size)
            _shrink_dataset(self.dataset_val, val_data_size)

        self.train_loader = torch.utils.data.DataLoader(
            self.dataset_train, **train_kwargs
        )
        self.test_loader = torch.utils.data.DataLoader(self.dataset_val, **test_kwargs)

        def net_factory():
            return Net()

        def optimizer_factory(params):
            return optim.Adadelta(params, lr=cfg.lr)

        self.loss_func = partial(F.nll_loss, reduction="mean")

        def lr_scheduler_factory(optimizer):
            return StepLR(optimizer, step_size=1, gamma=cfg.gamma)

        self.reference_model = ReferenceModel(
            net_factory, optimizer_factory, lr_scheduler_factory, self.loss_func
        )
        self.reference_model.get_model().to(self.device)

    def train(self):
        for epoch in range(1, self.cfg.epochs + 1):
            self.train_epoch(epoch)
            self.test()
            self.reference_model.lr_scheduler_step()

        if self.cfg.save_model:
            if not os.path.exists(self.cfg.path):
                os.makedirs(self.cfg.path)
            torch.save(
                self.reference_model.get_model().state_dict(),
                "checkpoints/mnist_cnn_ref.pt",
            )

    def train_epoch(self, epoch):
        self.reference_model.train(True)

        with mlflow.start_run():
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                step_info_ref = self.reference_model.step(data, target)
                ref_loss = step_info_ref["loss"]

                if batch_idx % self.cfg.log_interval == 0:
                    logger.info(
                        "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t".format(
                            epoch,
                            batch_idx * len(data),
                            len(self.train_loader.dataset),
                            100.0 * batch_idx / len(self.train_loader),
                            ref_loss,
                        )
                    )
                    mlflow.log_metric(key="loss", value=ref_loss, step=epoch)

    def test(self):
        self.reference_model.train(False)

        test_loss = 0
        correct = 0
        for data, target in self.test_loader:
            data, target = data.to(self.device), target.to(self.device)
            step_info_dp = self.reference_model.step(data, target, no_grad=True)
            test_loss += step_info_dp["loss"]  # sum up batch loss
            pred = step_info_dp["pred"].argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)

        logger.info(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(self.test_loader.dataset),
                100.0 * correct / len(self.test_loader.dataset),
            )
        )


@hydra.main(config_path="conf", config_name="config", version_base="1.3.2")
def main(cfg):
    trainer = Trainer(cfg.train)
    trainer.train()


if __name__ == "__main__":
    main()
