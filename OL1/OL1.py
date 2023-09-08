#!/usr/bin/env python3

import sys
import numpy as np
import torch
import lightning.pytorch as pl 
import pandas as pd
import torchmetrics

class Config:
    def __init__(self):
        self.data_type = None
        self.optimizer_name = None
        self.use_standardization = None
        self.max_epochs = 100
        self.num_classes = 0

    def parse_args(self):
        self.data_type = (sys.argv[1] or "").lower() # Make sure it is lower case
        self.optimizer_name = (sys.argv[2] or "").lower() # Make sure it is lower case
        self.use_standardization = (sys.argv[3] == "1") # Conver to bool
        self.device = "gpu" if torch.cuda.is_available() else "cpu" # Setup Device
    
    @property
    def is_multi_classification(self):
        return (self.num_classes > 2)

    @property
    def data_url(self):
        match self.data_type:
            case "iris":
                return "https://www.cs.mtsu.edu/~jphillips/courses/CSCI7850/public/iris-data.txt"
            case "wdbc":
                return "https://www.cs.mtsu.edu/~jphillips/courses/CSCI7850/public/WDBC.txt"
            case _:
                return ""
    
    @property
    def result_filename(self):
        _filename = ""
        _filename += self.data_type
        _filename += "-"
        _filename += "standardized" if self.use_standardization else "unstandardized"
        _filename += "-"
        _filename += self.optimizer_name
        _filename += "-results.txt"
        return _filename
    
    @property
    def to_string(self):
        return f'''
        *** Config Settings ***
        Data Type:\t{self.data_type}
        Optimizer:\t{self.optimizer_name}
        Class Cnt:\t{self.num_classes}
        Is Multi-Class:\t{self.is_multi_classification}
        Epoch Cnt:\t{self.max_epochs}
        Standarize:\t{self.use_standardization}
        Filename:\t{self.result_filename}
        '''
    



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
        y = torch.softmax(y, -1) if config.is_multi_classification else torch.sigmoid(y)
        return y # Final calculation returned


# Define Trainable Module
class PLModel(pl.LightningModule):
    def __init__(self, module, **kwargs):
        # This is the contructor, where we typically make
        # layer objects using provided arguments. 
        super().__init__(**kwargs) # Call the super class constructor 
        self.module = module
        
        # This creates an Accuracy and loss Functions
        # based on teh num_class being passed
        if(config.is_multi_classification):
            self.network_acc = torchmetrics.classification.Accuracy(task='multiclass', num_classes=config.num_classes)
            self.network_loss = torch.nn.CrossEntropyLoss()
        else:
            self.network_acc = torchmetrics.classification.Accuracy(task='binary')
            self.network_loss = torch.nn.BCEWithLogitsLoss()
        
    def forward(self, x):
        return self.module.forward(x)

    def predict(self, x):
        return self.module.predict(x)

    def configure_optimizers(self):
        match config.optimizer_name:
            case "adam":
                return torch.optim.Adam(self.parameters(), lr=0.01) 
            case "rmsprop":
                return torch.optim.RMSprop(self.parameters(), lr=0.01)
            case "sgd":
                return torch.optim.SGD(self.parameters(), lr=0.01) 
            case _:
                return None

    def training_step(self, train_batch, batch_idx): 
        x, y_true = train_batch
        # Handle Cross Entropy to Long vs. BCE is a Float
        y_true = torch.Tensor(y_true).type(torch.long) if config.is_multi_classification else y_true
        y_pred = self(x)
        acc = self.network_acc(y_pred, y_true)
        loss = self.network_loss(y_pred, y_true) 
        self.log('train_acc', acc, on_step=False, on_epoch=True) 
        self.log('train_loss', loss, on_step=False, on_epoch=True) 
        return loss

    def validation_step(self, val_batch, batch_idx): 
        x, y_true = val_batch
        # Handle Cross Entropy to Long vs. BCE is a Float
        y_true = torch.Tensor(y_true).type(torch.long) if config.is_multi_classification else y_true
        y_pred = self(x)
        acc = self.network_acc(y_pred, y_true)
        loss = self.network_loss(y_pred, y_true) 
        self.log('val_acc', acc, on_step=False, on_epoch=True) 
        self.log('val_loss', loss, on_step=False, on_epoch=True) 
        return loss

    def testing_step(self, test_batch, batch_idx): 
        x, y_true = test_batch
        # Handle Cross Entropy to Long vs. BCE is a Float
        y_true = torch.Tensor(y_true).type(torch.long) if config.is_multi_classification else y_true
        y_pred = self(x)
        acc = self.network_acc(y_pred,y_true)
        loss = self.network_loss(y_pred,y_true) 
        self.log('test_acc', acc, on_step=False, on_epoch=True) 
        self.log('test_loss', loss, on_step=False, on_epoch=True) 
        return loss



# Main Section of the Code Application
if __name__ == '__main__':
    config = Config()
    config.parse_args()
    
    data = np.loadtxt(config.data_url) 

    X = data[ : ,    : -1]
    Y = data[ : , -1 :   ]

    # Setup Number of Classes
    config.num_classes = len(np.unique(Y))

    # Display Config Settings
    print(config.to_string)

    def preprocess(x):
        return (x - np.mean(x)) / np.std(x)
    
    if (config.use_standardization):
        X = np.apply_along_axis(preprocess, 0, X)

    # Setting the Out Feature based onteh Classificaiotn Model used (Binary or Multi-Class)
    out_features = config.num_classes if config.is_multi_classification else Y.shape[-1]

    neural_net = NeuralNetwork(X.shape[-1], out_features).to(config.device)
   
    predictions = neural_net.predict(torch.Tensor(X[:5])) if config.is_multi_classification else neural_net.predict(torch.Tensor(X[:5,:])) 
    predictions.detach().numpy()

    model = PLModel(neural_net).to(config.device)

    # Define a permutation needed to shuffle both # inputs and targets in the same manner.... 
    shuffle = np.random.permutation(X.shape[0]) 
    X_shuffled = X[shuffle, :]
    Y_shuffled = Y[shuffle, :]

    # Keep 70% for training and remaining for validation
    split_point = int(X_shuffled.shape[0] * 0.7) 
    x_train = X_shuffled[:split_point]
    y_train = Y_shuffled[:split_point,0] if config.is_multi_classification else Y_shuffled[:split_point]

    x_val = X_shuffled[split_point:]
    y_val = Y_shuffled[split_point:,0] if config.is_multi_classification else Y_shuffled[split_point:]

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
    
    logger = pl.loggers.CSVLogger("logs", name = "Single-Layer-Network",) 
    
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=0,
        callbacks=[pl.callbacks.TQDMProgressBar(refresh_rate=50)])
    
    preliminary_result = trainer.validate(model, dataloaders=xy_val)

    trainer.fit(model, train_dataloaders=xy_train, val_dataloaders=xy_val)
    final_result = trainer.validate(model, dataloaders=xy_val)

    # Get Results from Logger
    results = pd.read_csv(f"{logger.log_dir}/metrics.csv", delimiter=',')

    # Calculate the expression and store it in a variable
    result_acc = ["%.8f"%(x) for x in results['val_acc'][np.logical_not(np.isnan(results["val_acc"]))][0::10]]
    result_acc_string = " ".join(result_acc)

    # Open the file in append mode
    with open(config.result_filename, 'a') as file:
        # Append the result_string to a new line in the file
        file.write(result_acc_string + "\n")

    print("Validation accuracy:",*["%.8f"%(x) for x in results['val_acc'][np.logical_not(np.isnan(results["val_acc"]))][0::10]])
