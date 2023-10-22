import csv
import logging

import hydra
import torch
import torch.utils.data
from torchvision import datasets, transforms

from classifier.classifier import Net

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, cfg):
        self.cfg = cfg

        use_cuda = not cfg.no_cuda and torch.cuda.is_available()

        torch.manual_seed(cfg.seed)

        self.device = torch.device("cuda" if use_cuda else "cpu")

        test_kwargs = {"batch_size": cfg.test_batch_size}
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
        self.inference_model.load_state_dict(
            torch.load(
                self.cfg.path + "/mnist_cnn_ref.pt", map_location=torch.device("cpu")
            )
        )
        self.inference_model.to(self.device)

    def evaluate(self):
        self.inference_model.train(False)

        test_loss = 0
        correct = 0
        preds = []
        for i, (data, target) in enumerate(self.test_loader):
            data, target = data.to(self.device), target.to(self.device)
            step_info_dp = self.inference_model(data)
            pred = step_info_dp.detach().argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            preds.append(pred.detach().cpu().numpy())
            correct += pred.eq(target.view_as(pred)).sum().item()

        with open(self.cfg.path + "/predictions.csv", "w+") as csvfile:
            writer = csv.writer(csvfile)
            for row in preds:
                for pred in row:
                    writer.writerow(pred)

        test_loss /= len(self.test_loader.dataset)

        logger.info(
            "\nTest set: Accuracy: {}/{} ({:.0f}%)\n".format(
                correct,
                len(self.test_loader.dataset),
                100.0 * correct / len(self.test_loader.dataset),
            )
        )


@hydra.main(config_path="conf", config_name="config", version_base="1.3.2")
def main(cfg):
    evaluator = Evaluator(cfg.inference)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
