import numpy as np
import torch
import lightning.pytorch as pl
import torchmetrics
import torchvision
from torchinfo import summary
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
cfg_data_folder = "datasets/cifar10"
cfg_batch_size  = 250
cfg_max_epochs  = 10;
cfg_num_workers = 2

# Logger Config
cfg_logger_dir     = "logs"
cfg_logger_name    = "OL3"
cfg_logger_version = "relu"

# CIFAR 10
training_dataset = torchvision.datasets.CIFAR10(root=cfg_data_folder, download=True, train=True)
testing_dataset = torchvision.datasets.CIFAR10(root=cfg_data_folder,  download=True, train=False)

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

class ResNet50(pl.LightningModule):
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

        self.resnet = torchvision.models.resnet50(weights=None, num_classes=output_size)
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


model = ResNet50(x_train.shape[1:], len(torch.unique(y_train)))

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

print("Validation accuracy:", *["%.8f"%(x) for x in results['val_acc'][np.logical_not(np.isnan(results["val_acc"]))]])
