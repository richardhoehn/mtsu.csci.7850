#!/usr/bin/env python3

import time
start = time.time() # Start Timing

import numpy as np
import torch
import lightning.pytorch as pl
import torchmetrics
import torchvision
from torchinfo import summary

import pandas as pd


# Config Section
cfg_batch_size  = 20
cfg_max_epochs  = 200
cfg_num_workers = 2

# Array Building Config
cfg_array_count      = 1000
cfg_min_array_length = 10
cfg_max_array_length = 30
cfg_start_int = 1
cfg_stop_int = 2
cfg_empty_int = 0

# Logger Config
cfg_logger_dir     = "logs"
cfg_logger_name    = "OL4"
cfg_logger_version = "parity-lstm"

if torch.cuda.is_available():
    print(torch.cuda.get_device_name())
    print(torch.cuda.get_device_properties("cuda"))
    print("Number of devices:",torch.cuda.device_count())
    device = ("cuda")
else:
    print("Only CPU is available...")
    device = ("cpu")


# Create The data and parity array
# Code it with start and stop integers
def make_arrays():
    # Generate a random length between 10 and 30 (30 + 1)
    length = np.random.randint(cfg_min_array_length, cfg_max_array_length + 1)

    # Create a random array of 0s and 1s with the random length
    data_array = np.random.randint(2, size=length)

    # Initialize an array to store the cumulative sum of random_array
    cumulative_sum = np.cumsum(data_array)

    # Calculate parity for each cumulative sum and save to a separate array
    parity_array = cumulative_sum % 2

    # No We need to convert Array to Encoding 0s = 3 and 1s = 4
    enc_data_array = np.where(data_array == 0, 3, 4)
    enc_parity_array = np.where(parity_array == 0, 3, 4)
    
    enc_data_array = np.concatenate(([cfg_start_int], enc_data_array, [cfg_stop_int]))
    enc_parity_array = np.concatenate(([cfg_start_int], enc_parity_array, [cfg_stop_int]))

    # Add Padding at the end to make sure ALL Arrays are same length
    enc_data_array = np.pad(enc_data_array, (0, cfg_max_array_length - length), 'constant')
    enc_parity_array = np.pad(enc_parity_array, (0, cfg_max_array_length - length), 'constant')

    return np.array([np.array(enc_data_array), np.array(enc_parity_array)])


# Create an array to store 1000 arrays generated by make_arrays
data = np.array([make_arrays() for _ in range(cfg_array_count)])
print(f"data.shape: {data.shape}")

# Print Example of First Row
data_array = data[0][0]
parity_array = data[0][1]
print("\nArray Example:")
print("Encoded:\nKey: 1 = Start, 2 = Stop, 3 = Zero, 4 = One, 0 = Padding / Empty")
print("Data Array:   ", data_array)
print("Parity Array: ", parity_array)
print("")

# Set Split Point
split_point = int(data.shape[0] * 0.8) # Keep 80/20 split
print(f"split_point: {split_point}")


# Setup - Create X & Y
X = data[:,0]
Y = data[:,1]

enc_x_train = X[:split_point]
enc_x_val = X[split_point:]
enc_x_train

dec_x_train = Y[:,0:-1][:split_point]
dec_x_val = Y[:,0:-1][split_point:]
dec_x_train

dec_y_train = Y[:,1:][:split_point]
dec_y_val = Y[:,1:][split_point:]
dec_y_train

print(enc_x_train.shape)
print(dec_x_train.shape)
print(dec_y_train.shape)

print("----")

print(enc_x_val.shape)
print(dec_x_val.shape)
print(dec_y_val.shape)

# Start, Stop, One, Zero, Empty
token_count = 5
print(f"Token Count: {token_count}")


class Lstm(torch.nn.Module):
    def __init__(self,
                 latent_size = 64,
                 bidirectional = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.layer_norm = torch.nn.LayerNorm(latent_size)
        self.lstm_layer = torch.nn.LSTM(latent_size,
                                      latent_size // 2 if bidirectional else latent_size,
                                      bidirectional=bidirectional,
                                      batch_first=True)
    def forward(self, x):
        return x + self.lstm_layer(self.layer_norm(x))[0]

class EncoderNetwork(torch.nn.Module):
    def __init__(self,
                 num_tokens,
                 latent_size = 64, # Use something divisible by 2
                 n_layers = 4,
                 **kwargs):
        super().__init__(**kwargs)
        self.embedding = torch.nn.Embedding(num_tokens,
                                            latent_size,
                                            padding_idx=0)
        self.dropout = torch.nn.Dropout1d(0.1) # Whole token dropped
        self.lstm_layers = torch.nn.Sequential(*[
            Lstm(latent_size, True) for _ in range(n_layers)
        ])
        
    def forward(self, x):
        y = x
        y = self.embedding(y)
        y = self.dropout(y)
        y = self.lstm_layers(y)[:,-1]
        return y


enc_net = EncoderNetwork(num_tokens=token_count)
summary(enc_net,input_data=torch.Tensor(enc_x_train[0:5]).long())


# Decoder Component
class DecoderNetwork(torch.nn.Module):
    def __init__(self,
                 num_tokens,
                 latent_size = 64, # Use something divisible by 2
                 n_layers = 4,
                 **kwargs):
        super().__init__(**kwargs)
        self.embedding = torch.nn.Embedding(num_tokens,
                                            latent_size,
                                            padding_idx=0)
        self.dropout = torch.nn.Dropout1d(0.1) # Whole token dropped
        self.linear = torch.nn.Linear(latent_size*2, latent_size)
        self.lstm_layers = torch.nn.Sequential(*[
            Lstm(latent_size,False) for _ in range(n_layers)
        ])
        self.output_layer = torch.nn.Linear(latent_size,
                                            num_tokens)
    
    def forward(self, x_enc, x_dec):
        y_enc = x_enc.unsqueeze(1).repeat(1,x_dec.shape[1],1)
        y_dec = self.embedding(x_dec)
        y_dec = self.dropout(y_dec)
        y = y_enc
        y = torch.concatenate([y_enc,y_dec],-1)
        y = self.linear(y)
        y = self.lstm_layers(y)
        y = self.output_layer(y)
        return y

dec_net = DecoderNetwork(num_tokens=token_count)
summary(dec_net,input_data=[enc_net(torch.Tensor(enc_x_train[0:5]).long()).cpu(), torch.Tensor(dec_x_train[0:5]).long()])


class EncDecLightningModule(pl.LightningModule):
    def __init__(self,
                 output_size,
                 **kwargs):
        super().__init__(**kwargs)
        self.mc_acc = torchmetrics.classification.Accuracy(task='multiclass',
                                                           num_classes=output_size,
                                                           ignore_index=0)
        self.cce_loss = torch.nn.CrossEntropyLoss(ignore_index=0)

    def predict(self, x):
        return torch.softmax(self(x),-1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x_enc, x_dec, y_dec = train_batch
        y_pred = self(x_enc, x_dec)
        perm = (0,-1) + tuple(range(y_pred.ndim))[1:-1]
        acc = self.mc_acc(y_pred.permute(*perm),y_dec)
        loss = self.cce_loss(y_pred.permute(*perm),y_dec)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss
    
    # Validate used for Teacher Forcing
    def validation_step(self, val_batch, batch_idx):
        x_enc, x_dec, y_dec = val_batch
        y_pred = self(x_enc, x_dec)
        perm = (0,-1) + tuple(range(y_pred.ndim))[1:-1]
        acc = self.mc_acc(y_pred.permute(*perm),y_dec)
        loss = self.cce_loss(y_pred.permute(*perm),y_dec)
        self.log('val_acc', acc, on_step=False, on_epoch=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss
    
    # Test used for Non-Teacher Forcing
    def test_step(self, test_batch, batch_idx):
        x_enc, x_dec, y_dec = test_batch
        context = self.enc_net(x_enc)
        tokens = torch.zeros_like(x_dec).long()
        tokens[:,0] = 1
        for i in range(y_dec.shape[1]-1):
            tokens[:,i+1] = self.dec_net(context, tokens).argmax(-1)[:,i]
        y_pred = self(x_enc, tokens)
        perm = (0,-1) + tuple(range(y_pred.ndim))[1:-1]
        acc = self.mc_acc(y_pred.permute(*perm),y_dec)
        loss = self.cce_loss(y_pred.permute(*perm),y_dec)
        self.log('test_acc', acc, on_step=False, on_epoch=True)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        return loss

class EncDecNetwork(EncDecLightningModule):
    def __init__(self,
                 num_enc_tokens,
                 num_dec_tokens,
                 latent_size = 64, # Use something divisible by 2 
                 n_layers = 4,
                 **kwargs):
        super().__init__(output_size=num_dec_tokens, **kwargs)
        self.enc_net = EncoderNetwork(num_enc_tokens,latent_size,n_layers)
        self.dec_net = DecoderNetwork(num_dec_tokens,latent_size,n_layers)
    
    def forward(self, x_enc, x_dec):
        return self.dec_net(self.enc_net(x_enc), x_dec)


enc_dec_net = EncDecNetwork(num_enc_tokens=token_count,
                            num_dec_tokens=token_count)

summary(enc_dec_net,input_data=[torch.Tensor(enc_x_train[0:1]).long(),
                                torch.Tensor(dec_x_train[0:1]).long()])



xy_train = torch.utils.data.DataLoader(list(zip(torch.Tensor(enc_x_train).long(),
                                                torch.Tensor(dec_x_train).long(),
                                                torch.Tensor(dec_y_train).long())), 
                                       shuffle=True, 
                                       batch_size=cfg_batch_size, 
                                       num_workers=cfg_num_workers)

xy_val = torch.utils.data.DataLoader(list(zip(torch.Tensor(enc_x_val).long(), 
                                              torch.Tensor(dec_x_val).long(),
                                              torch.Tensor(dec_y_val).long())), 
                                     shuffle=False, 
                                     batch_size=cfg_batch_size, 
                                     num_workers=cfg_num_workers)


logger = pl.loggers.CSVLogger(cfg_logger_dir,
                              name=cfg_logger_name,
                              version=cfg_logger_version)


trainer = pl.Trainer(logger=logger,
                     max_epochs=cfg_max_epochs,
                     enable_progress_bar=True,
                     log_every_n_steps=0,
                     enable_checkpointing=False,
                     callbacks=[pl.callbacks.TQDMProgressBar(refresh_rate=50)])


# Train Model
trainer.fit(enc_dec_net, xy_train, xy_val) 

# Test Model
results = trainer.test(enc_dec_net, xy_val)

# Hanlding Timing
end = time.time()
elapsed = end - start

print("")
print(f"Processing Time: {elapsed:.6f} seconds\n")
print("Test Accuracy:", results[0]['test_acc'])
print("")
print("")