#!/usr/bin/env python3

import sys
import os

import numpy as np
import torch
import lightning.pytorch as pl 
from torchinfo import summary 
from torchview import draw_graph 
import matplotlib.pyplot as plt 
import pandas as pd
import torchmetrics

class Config:
    def __init__(self):
        self._data_type = None
        self._optimizer_name = None
        self._use_standardization = None
        self.max_epochs = 100

    def parse_args(self):
        self._data_type = (sys.argv[1] or "").lower() # Make sure it is lower case
        self._optimizer_name = (sys.argv[2] or "").lower() # Make sure it is lower case
        self._use_standardization = (sys.argv[3] == "1") # Conver to bool
        self.device = "gpu" if torch.cuda.is_available() else "cpu" # Setup Device
    
    @property
    def data_url(self):
        match self._data_type:
            case "iris":
                return "https://www.cs.mtsu.edu/~jphillips/courses/CSCI7850/public/iris-data.txt"
            case "wdbc":
                return "https://www.cs.mtsu.edu/~jphillips/courses/CSCI7850/public/WDBC.txt"
            case _:
                return ""
    
    @property
    def result_filename(self):
        _filename = ""
        _filename += self._data_type
        _filename += "-"
        _filename += "standardized" if self._use_standardization else "unstandardized"
        _filename += "-"
        _filename += self._optimizer_name
        _filename += "-results.txt"
        return _filename
    



# Define Neural Netowrk Model
class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_size, output_size, **kwargs):
        # This is the contructor, where we typically make
        # layer objects using provided arguments. 
        super().__init__(**kwargs) # Call the super class constructor
        
        # This is an actual neural layer...
        self.output_layer = torch.nn.Linear(input_size, output_size)
    
    def forward(self, x):
        # Here is where we use the layers to compute something...
        y = x # Start with the input
        y = self.output_layer(y) # y replaces y (stateful parameters) 
        return y # Final calculation returned

    # Separate the final activation function out because
    # binary_cross_entropy assumes you are using a sigmoid
    # (outputs are considered logits - more later on this...) 
    def predict(self, x):
        # Here is where we use the layers to compute something...
        y = x
        y = self.forward(y) # Start with the input
        y = torch.sigmoid(y) # Apply the activation function (stateless)
        return y # Final calculation returned


# Define Trainable Module
class PLModel(pl.LightningModule):
    def __init__(self, module, **kwargs):
        # This is the contructor, where we typically make
        # layer objects using provided arguments. 
        super().__init__(**kwargs) # Call the super class constructor 
        self.module = module
        
        # This creates an accuracy function
        self.network_acc = torchmetrics.classification.Accuracy(task='binary') # This creates a loss function
        self.network_loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.module.forward(x)

    def predict(self, x):
        return self.module.predict(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001) 
        return optimizer

    def training_step(self, train_batch, batch_idx): 
        x, y_true = train_batch
        y_pred = self(x)
        acc = self.network_acc(y_pred, y_true)
        loss = self.network_loss(y_pred, y_true) 
        self.log('train_acc', acc, on_step=False, on_epoch=True) 
        self.log('train_loss', loss, on_step=False, on_epoch=True) 
        return loss

    def validation_step(self, val_batch, batch_idx): 
        x, y_true = val_batch
        y_pred = self(x)
        acc = self.network_acc(y_pred, y_true)
        loss = self.network_loss(y_pred, y_true) 
        self.log('val_acc', acc, on_step=False, on_epoch=True) 
        self.log('val_loss', loss, on_step=False, on_epoch=True) 
        return loss

    def testing_step(self, test_batch, batch_idx): 
        x, y_true = test_batch
        y_pred = self(x)
        acc = self.network_acc(y_pred,y_true)
        loss = self.network_loss(y_pred,y_true) 
        self.log('test_acc', acc, on_step=False, on_epoch=True) 
        self.log('test_loss', loss, on_step=False, on_epoch=True) 
        return loss



# Main Section of the Code Application
if __name__ == '__main__':
    cfg = Config()
    cfg.parse_args()
    
    data = np.loadtxt(cfg.data_url) 
    print(data[0:5])
    
    X = data[ : ,    : -1]
    Y = data[ : , -1 :   ]

    neural_net = NeuralNetwork(X.shape[-1],
                               Y.shape[-1]).to(cfg.device)
    
    predictions = neural_net.predict(torch.Tensor(X[:5,:])) 
    predictions = predictions.detach().numpy()
    print(predictions)

    model = PLModel(neural_net).to(cfg.device)
    print(model)

    # Define a permutation needed to shuffle both # inputs and targets in the same manner.... 
    shuffle = np.random.permutation(X.shape[0]) 
    X_shuffled = X[shuffle, :]
    Y_shuffled = Y[shuffle, :]

    # Keep 70% for training and remaining for validation
    split_point = int(X_shuffled.shape[0] * 0.7) 
    x_train = X_shuffled[:split_point]
    y_train = Y_shuffled[:split_point]
    x_val = X_shuffled[split_point:]
    y_val = Y_shuffled[split_point:]

    # The dataloaders handle shuffling, batching, etc...
    xy_train = torch.utils.data.DataLoader(
        list(
            zip(
                torch.Tensor(x_train).type(torch.float),
                torch.Tensor(y_train).type(torch.float)
            )
        ),
        shuffle=True, 
        batch_size=32)
    
    xy_val = torch.utils.data.DataLoader(
        list(
            zip(
                torch.Tensor(x_val).type(torch.float),
                torch.Tensor(y_val).type(torch.float)
            )
        ),
        shuffle=False, 
        batch_size=32)
    
    logger = pl.loggers.CSVLogger(
        "logs",
        name = "Single-Layer-Network",) 
    
    print(f"Logger Dir: {logger.log_dir}")
    
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        logger=logger,
        enable_progress_bar= True,
        log_every_n_steps=0,
        callbacks=[pl.callbacks.TQDMProgressBar(refresh_rate=50)])
    
    preliminary_result = trainer.validate(model, dataloaders=xy_val)
    print(preliminary_result)

    trainer.fit(model, train_dataloaders=xy_train, val_dataloaders=xy_val)
    final_result = trainer.validate(model, dataloaders=xy_val)
    print(final_result)


    results = pd.read_csv(f"{logger.log_dir}/metrics.csv", delimiter=',')
    print(results)
    print(results.columns)
    print("Validation accuracy:",*["%.8f"%(x) for x in results['val_acc'][np.logical_not(np.isnan(results["val_acc"]))][0::10]])
    print(cfg.result_filename)

    # Calculate the expression and store it in a variable
    result_acc = ["%.8f"%(x) for x in results['val_acc'][np.logical_not(np.isnan(results["val_acc"]))][0::10]]
    result_acc_string = " ".join(result_acc)

    # Open the file in append mode
    with open(cfg.result_filename, 'a') as file:
        # Append the result_string to a new line in the file
        file.write(result_acc_string + "\n")

    res = np.loadtxt(cfg.result_filename)
    print(res)
    print(res.shape)
