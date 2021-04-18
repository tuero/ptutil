from ptu.module import PTUtilModule
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ptu.util.types import LoggingItem
from ptu.util.types import OptimizerInfo
from ptu.util.types import MetricsItem
from ptu.util.types import MetricFramework, MetricType, Mode

# -------------------------
import argparse
import gin
import torch
import torchvision
import torchvision.transforms as transforms
from ptu.trainer import Trainer
from ptu.callbacks.logger import Logger
from ptu.callbacks.checkpoint import Checkpoint
from ptu.callbacks.grad_clipping import GradClipping
from ptu.callbacks.tracker_tensorboard import TrackerTensorboard
from ptu.callbacks.early_stoppage import EarlyStoppage


class FCN(PTUtilModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # Set optimizers
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        self.optimizer_infos = [OptimizerInfo(optimizer=self.optimizer)]

    def __str__(self):
        return "test-fcn"

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def after_train_step(self):
        train_loss = self.trainer.epoch_loss_train / self.trainer.size_train_data
        logging_msg = "Epoch: {:>4},   Training Loss: {:>10.4f}".format(self.trainer.step_epoch, train_loss)
        self.trainer.logging_buffer.append(LoggingItem("INFO", logging_msg))
        self.trainer.metrics_buffer.append(
            MetricsItem(
                MetricFramework.tensorboard,
                MetricType.scalar,
                Mode.train,
                ("loss", train_loss, self.trainer.step_epoch),
            )
        )

    def after_val_step(self):
        val_loss = self.trainer.epoch_loss_val / self.trainer.size_val_data
        logging_msg = "Epoch: {:>4}, Validation Loss: {:>10.4f}".format(self.trainer.step_epoch, val_loss)
        self.trainer.logging_buffer.append(LoggingItem("INFO", logging_msg))
        self.trainer.metrics_buffer.append(
            MetricsItem(
                MetricFramework.tensorboard,
                MetricType.scalar,
                Mode.val,
                ("loss", val_loss, self.trainer.step_epoch),
            )
        )

    def calc_loss(self, outputs, labels):
        return F.cross_entropy(outputs, labels).mean()

    def step(self, batch):
        inputs, labels = batch
        inputs = inputs.to(self.trainer.device)
        labels = labels.to(self.trainer.device)
        # forward + backward + optimize
        outputs = self.__call__(inputs)
        return self.calc_loss(outputs, labels)

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        return loss


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)


if __name__ == "__main__":
    # Argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", dest="exp", required=False, type=str)
    parser.add_argument(
        "--gin_config", dest="gin_config", required=False, type=str, default="./config/default_config.gin"
    )
    parser.add_argument("--checkpoint", default=False, action="store_true")
    args = parser.parse_args()

    # rebind new/overwritten gin parameters
    gin.parse_config_file(args.gin_config)
    if args.exp is not None:
        gin.bind_parameter("%experiment_name", args.exp)

    trainer = Trainer(cbs=[Logger(), Checkpoint(), GradClipping(), TrackerTensorboard(), EarlyStoppage()])
    model = FCN()

    # Load from checkpoint
    if args.checkpoint:
        trainer.load_from_checkpoint(model)
    trainer.fit(model, trainloader, testloader)
