import argparse
import os
from functools import partial

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from classifier.classifier import Net, ReferenceModel


class Trainer:
    def __init__(self, args):
        self.args = args

        use_cuda = not args.no_cuda and torch.cuda.is_available()

        torch.manual_seed(args.seed)

        self.device = torch.device("cuda" if use_cuda else "cpu")

        train_kwargs = {"batch_size": args.batch_size}
        test_kwargs = {"batch_size": args.test_batch_size}
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
            return optim.Adadelta(params, lr=args.lr)

        self.loss_func = partial(F.nll_loss, reduction="mean")

        def lr_scheduler_factory(optimizer):
            return StepLR(optimizer, step_size=1, gamma=args.gamma)

        self.reference_model = ReferenceModel(
            net_factory, optimizer_factory, lr_scheduler_factory, self.loss_func
        )
        self.reference_model.get_model().to(self.device)

    def train(self):
        for epoch in range(1, self.args.epochs + 1):
            self.train_epoch(epoch)
            self.test()
            self.reference_model.lr_scheduler_step()

        if self.args.save_model:
            if not os.path.exists(self.args.path):
                os.makedirs(self.args.path)
            torch.save(
                self.reference_model.get_model().state_dict(),
                "checkpoints/mnist_cnn_ref.pt",
            )

    def train_epoch(self, epoch):
        self.reference_model.train(True)

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            step_info_ref = self.reference_model.step(data, target)
            ref_loss = step_info_ref["loss"]

            if batch_idx % self.args.log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tRef loss: {:.6f}\t".format(
                        epoch,
                        batch_idx * len(data),
                        len(self.train_loader.dataset),
                        100.0 * batch_idx / len(self.train_loader),
                        ref_loss,
                    )
                )

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

        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(self.test_loader.dataset),
                100.0 * correct / len(self.test_loader.dataset),
            )
        )


def parse_args(external_args=None):
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST model train")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=True,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--num-replicas",
        type=int,
        default=8,
        metavar="N",
        help="number of dataparallel replicas (default: 4)",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="checkpoints",
        metavar="N",
        help="directory for model saves",
    )
    if external_args is not None:
        args = parser.parse_args(external_args)
    else:
        args = parser.parse_args()
    return args


def main():
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
