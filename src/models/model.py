import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms

import pytorch_lightning as pl

class Net(pl.LightningModule):
    def __init__(self, dataset, in_channels, hp, loss_func):
        super(Net, self).__init__()

        self.lr = hp['lr']
        self.dataset = dataset
        self.num_classes = self.dataset.get_num_classes()
        self.loss_func = loss_func

        self.channels = [64]
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=self.channels[0], 
            kernel_size=4, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=self.channels[0], out_channels=32, 
            kernel_size=4, stride=1, padding=2)

        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout1 = nn.Dropout2d(0.25)

        self.relu = nn.ReLU()

        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Linear(7200, self.num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.log_softmax(x)
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.loss_func(outputs, labels)

        self.log("ptl/loss", loss)

        return {'loss': loss}

    def training_step_end(self, outputs):
        self.log('ptl/train_loss_batch', outputs['loss'])

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        output = self(images)
        loss = self.loss_func(output, labels)

        return {'val_loss': loss}

    def validation_step_end(self, outputs):
        self.log("ptl/val_loss_batch", outputs['val_loss'])

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log("ptl/val_loss", avg_loss)

    def test_step(self, batch, batch_idx):
        images, labels = batch
        output = self(images)
        _, predicted = torch.max(output,1)
        correct = (predicted == labels).sum()
        total = labels.size(0)

        return {'test_acc': correct/total}

    def test_epoch_end(self, outputs):
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        self.log("ptl/test_acc", avg_acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 
                            lr=self.lr)

