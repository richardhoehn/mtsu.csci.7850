#!/usr/bin/env python3

import time
start = time.time() # Start Timing

import numpy as np

import torch
import lightning.pytorch as pl
import torchmetrics
import torchvision
from torchinfo import summary

from typing import List, Optional

import pandas as pd


if torch.cuda.is_available():
    print(torch.cuda.get_device_name())
    print(torch.cuda.get_device_properties("cuda"))
    print("Number of devices:", torch.cuda.device_count())
    device = ("cuda")
else:
    print("Only CPU is available...")
    device = ("cpu")

# Config Section
cfg_data_folder = "datasets/cifar100"
cfg_batch_size  = 250
cfg_max_epochs  = 50
cfg_num_workers = 2

# Logger Config
cfg_logger_dir     = "logs"
cfg_logger_name    = "OL3"
cfg_logger_version = "gelu"

# CIFAR 10
training_dataset = torchvision.datasets.CIFAR100(root=cfg_data_folder, download=True, train=True)
testing_dataset = torchvision.datasets.CIFAR100(root=cfg_data_folder,  download=True, train=False)

# Training
x_train = torch.Tensor(training_dataset.data).permute(0, 3, 1, 2)
y_train = torch.Tensor(training_dataset.targets).to(torch.long)

# Validation / Testing Data
x_test = torch.Tensor(testing_dataset.data).permute(0, 3, 1 ,2)
y_test = torch.Tensor(testing_dataset.targets).to(torch.long)

xy_train = torch.utils.data.DataLoader(list(zip(x_train,
                                                y_train)),
                                       shuffle=True,
                                       batch_size=cfg_batch_size,
                                       num_workers=cfg_num_workers)

xy_val = torch.utils.data.DataLoader(list(zip(x_test,
                                              y_test)), 
                                     shuffle=False,
                                     batch_size=cfg_batch_size,
                                     num_workers=cfg_num_workers)

#
# *****************************************
# ********** Citation References **********
# *****************************************
#
# It should be noted that below code for the following 
# functions and classes was heaviliy adapted from the 
# following sources:
#
# -> https://pytorch.org/vision/main/_modules/torchvision/models/resnet.html#resnet50
#    Detailes on how to creare the underlying calls to Resnet
#
# -> https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
#    This is the source code for Resnet that is implmented in PyTorch. I 
#    used this for a large part of the below code. Heavy adapatons have taking place
#    to make my classes and functions a bit more readable for my educational
#    pruposes!


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    return torch.nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(torch.nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    # *** NOTES - By Richard Hoehn
    # This has been reduced to only work for Resnet50!

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[torch.nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1
    ) -> None:
        super().__init__()
        

        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = torch.nn.BatchNorm2d(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = torch.nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = torch.nn.BatchNorm2d(planes * self.expansion)
        self.gelu = torch.nn.GELU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.gelu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.gelu(out)

        return out


class ResNetGelu(torch.nn.Module):
    def __init__(
        self,
        block: Bottleneck,
        layers: List[int],
        num_classes: int = 10,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
    ) -> None:
        super().__init__()
        
        self.inplanes = 64
        self.dilation = 1
        
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = torch.nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(self.inplanes)
        self.gelu = torch.nn.GELU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.output_layer = torch.nn.Linear(512 * block.expansion, num_classes)


    def _make_layer(
        self,
        block: Bottleneck,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> torch.nn.Sequential:
        norm_layer = torch.nn.BatchNorm2d
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = torch.nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [] # Setup the Layers
        
        layers.append(
            # Setup Blocks for helper functions
            block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation)
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation
                )
            )

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        y = x
        
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.gelu(y)
        y = self.maxpool(y)

        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)

        y = self.avgpool(y)
        y = torch.flatten(y, 1)
        y = self.output_layer(y)

        return y

class ResNet50Gelu(pl.LightningModule):
    def __init__(self,
                 input_shape,
                 output_size,
                 **kwargs):
        super().__init__(**kwargs)
        
        # Needs to always be applied to any incoming image for this model. The Compose operation
        # takes a list of torchvision transforms and applies them in sequential order, similar
        # to neural layers...
        self.normalize = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda x: x / 255.0),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Besides just scaling, the images can also undergo augmentation using torchvision. Again, we compose
        # these operations together - ranges are provided for each of these augmentations.
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomAffine(degrees=(-10.0, 10.0),
                                                translate=(0.1, 0.1),
                                                scale=(0.9, 1.1),
                                                shear=(-10.0, 10.0)),
            torchvision.transforms.RandomHorizontalFlip(0.5),
        ])

        # Linear projection - learned upsampling
        self.projection = torch.nn.ConvTranspose2d(3,3,
                                                   (4,4), # 8x
                                                   (4,4)) # 8+

        self.resnet = ResNetGelu(Bottleneck, [3, 4, 6, 3], num_classes=output_size)
        
        self.model_acc = torchmetrics.classification.Accuracy(task='multiclass', num_classes=output_size)
        self.model_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        y = x
        # Always normalize
        y = self.normalize(y)
        # Only augment when training 
        if self.training:
            y = self.transform(y)
        y = self.projection(y)
        y = self.resnet(y)
        return y

    def predict(self, x):
        return torch.softmax(self(x), -1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-4)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y_true = train_batch
        y_pred = self(x)
        acc = self.model_acc(y_pred, y_true)
        loss = self.model_loss(y_pred, y_true)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx): 
        x, y_true = val_batch
        y_pred = self(x)
        acc = self.model_acc(y_pred, y_true)
        loss = self.model_loss(y_pred, y_true)
        self.log('val_acc',  acc,  on_step=False, on_epoch=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss
    

model = ResNet50Gelu(x_train.shape[1:], len(torch.unique(y_train)))

summary(model, input_size=(1,)+x_train.shape[1:], depth=4)

logger = pl.loggers.CSVLogger(cfg_logger_dir,
                              name=cfg_logger_name,
                              version=cfg_logger_version)

trainer = pl.Trainer(logger=logger,
                     max_epochs=cfg_max_epochs,
                     enable_progress_bar=True,
                     log_every_n_steps=0,
                     enable_checkpointing=False,
                     callbacks=[pl.callbacks.TQDMProgressBar(refresh_rate=50)])


trainer.fit(model, xy_train, xy_val)

results = pd.read_csv(logger.log_dir + "/metrics.csv")

# Hanlding Timing
end = time.time()
elapsed = end - start

print("")
print(f"Processing Time: {elapsed:.6f} seconds\n")
print("Validation accuracy:", *["%.8f"%(x) for x in results['val_acc'][np.logical_not(np.isnan(results["val_acc"]))]])
print("")
print("")
