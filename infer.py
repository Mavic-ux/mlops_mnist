import argparse

import torch
import torch.utils.data
from torchvision import datasets, transforms

from classifier.classifier import Net


class Evaluator:
    def __init__(self, args):
        self.args = args

        use_cuda = not args.no_cuda and torch.cuda.is_available()

        torch.manual_seed(args.seed)

        self.device = torch.device("cuda" if use_cuda else "cpu")

        test_kwargs = {"batch_size": args.test_batch_size}
        if use_cuda:
            cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
            test_kwargs.update(cuda_kwargs)

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.dataset_val = datasets.MNIST("../data", train=False, transform=transform)

        shrink_dataset = True
        if shrink_dataset:
            val_data_size = 1000

            def _shrink_dataset(dataset, size):
                dataset.data = dataset.data[:size]
                dataset.targets = dataset.targets[:size]

            _shrink_dataset(self.dataset_val, val_data_size)

        self.test_loader = torch.utils.data.DataLoader(self.dataset_val, **test_kwargs)

        self.inference_model = Net()
        self.inference_model.load_state_dict(torch.load(args.path))
        self.inference_model.to(self.device)

    def evaluate(self):
        self.inference_model.train(False)

        test_loss = 0
        correct = 0
        for data, target in self.test_loader:
            data, target = data.to(self.device), target.to(self.device)
            step_info_dp = self.inference_model(data)
            pred = step_info_dp.detach().argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)

        print(
            "\nTest set: Accuracy: {}/{} ({:.0f}%)\n".format(
                correct,
                len(self.test_loader.dataset),
                100.0 * correct / len(self.test_loader.dataset),
            )
        )


def parse_args(external_args=None):
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST model evaluate")
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--path",
        type=str,
        default="checkpoints/mnist_cnn_ref.pt",
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
    evaluator = Evaluator(args)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
