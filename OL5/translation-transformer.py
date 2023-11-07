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

import urllib


if torch.cuda.is_available():
    print(torch.cuda.get_device_name())
    print(torch.cuda.get_device_properties("cuda"))
    print("Number of devices:", torch.cuda.device_count())
    device = ("cuda")
else:
    print("Only CPU is available...")
    device = ("cpu")

# Config Section
cfg_embedding_len = 100
cfg_batch_size    = 128
cfg_max_epochs    = 200
cfg_num_workers   = 2
cfg_url           = "https://raw.githubusercontent.com/luisroque/deep-learning-articles/main/data/eng-por.txt"

# Logger Config
cfg_logger_dir     = "logs"
cfg_logger_name    = "OL5"
cfg_logger_version = "translation-transformer"

# Setup Data Objects
data = []
with urllib.request.urlopen(cfg_url) as raw_data:
    for line in raw_data:
        data.append(line.decode("utf-8").split('\t')[0:2])
        
data = np.array(data)

# Subset? - All of the data will take some time...
n_seq = data.shape[0]
n_seq = 10000
data = data[0:n_seq]
split_point = int(data.shape[0] * 0.8) # Keep 80/20 split
np.random.shuffle(data) # In-place modification
max_length = np.max([len(i) for i in data.flatten()]) + 2 # Add start/stop

print(f"max_length = ${max_length}")
print(f"data[0] => ${data[0]}")

# Setup
i_to_c_eng = ['','<START>','<STOP>'] + list({char for word in data[:,0] for char in word})
c_to_i_eng = {i_to_c_eng[i]:i for i in range(len(i_to_c_eng))}
i_to_c_eng[1] = i_to_c_eng[2] = ''

i_to_c_por = ['','<START>','<STOP>'] + list({char for word in data[:,1] for char in word})
c_to_i_por = {i_to_c_por[i]:i for i in range(len(i_to_c_por))}
i_to_c_por[1] = i_to_c_por[2] = ''


def encode_seq(x, mapping, max_length=0):
    # String to integer
    return [mapping['<START>']] + \
    [mapping[i] for i in list(x)] + \
    [mapping['<STOP>']] + \
    [0]*(max_length-len(list(x))-2)
    
def decode_seq(x, mapping):
    # Integer-to-string
    try:
        idx = list(x).index(2) # Stop token?
    except:
        idx = len(list(x)) # No stop token found
    return ''.join([mapping[i] for i in list(x)[0:idx]])



# Setup
X = np.vstack([encode_seq(x, c_to_i_eng, max_length) for x in data[:,0]])
Y = np.vstack([encode_seq(x, c_to_i_por, max_length) for x in data[:,1]])

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


class TransformerBlock(torch.nn.Module):
    def __init__(self,
                 latent_size = 64,
                 num_heads = 4,
                 dropout = 0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.layer_norm1 = torch.nn.LayerNorm(latent_size)
        self.layer_norm2 = torch.nn.LayerNorm(latent_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.GELU()
        self.linear = torch.nn.Linear(latent_size,
                                      latent_size)
        self.mha = torch.nn.MultiheadAttention(latent_size,
                                               num_heads, 
                                               dropout=dropout,
                                               batch_first=True)
    
    def forward(self, x):
        y = x
        y = self.layer_norm1(y)
        y = self.mha(y, y, y)[0]
        x=y=x+y
        y = self.layer_norm2(y)
        y = self.linear(y)
        y = self.activation(y)
        return x +  y

class EncoderNetwork(torch.nn.Module):
    def __init__(self,
                 num_tokens,
                 max_length,
                 latent_size = 64,
                 num_heads = 4,
                 n_layers = 8,
                 embedding_len = 100,
                 **kwargs):
        super().__init__(**kwargs)
        self.token_embedding = torch.nn.Embedding(num_tokens,
                                                  embedding_len,
                                                  padding_idx=0)
        self.position_embedding = torch.nn.Embedding(max_length+1,
                                                     embedding_len,
                                                     padding_idx=0)
        self.projection = torch.nn.Linear(embedding_len, latent_size)
        self.transformer_blocks = torch.nn.Sequential(*[
            TransformerBlock(latent_size=latent_size,
                             num_heads=num_heads) for _ in range(n_layers)
        ])
        
    def forward(self, x):
        y = x
        y = self.token_embedding(y) + \
            self.position_embedding(torch.arange(1,y.shape[1]+1).long().to(device) * torch.sign(y).long().to(device))
        y = self.projection(y)
        y = self.transformer_blocks(y)
        return y


enc_net = EncoderNetwork(num_tokens=len(i_to_c_eng), max_length=enc_x_train.shape[-1])
summary(enc_net,input_size=enc_x_train[0:5].shape,dtypes=[torch.long])


class MaskedTransformerBlock(torch.nn.Module):
    def __init__(self,
                 latent_size = 64, 
                 num_heads = 4, 
                 dropout = 0.1, 
                 **kwargs):
        super().__init__(**kwargs)
        self.layer_norm1 = torch.nn.LayerNorm(latent_size) 
        self.layer_norm2 = torch.nn.LayerNorm(latent_size) 
        self.layer_norm3 = torch.nn.LayerNorm(latent_size) 
        self.dropout = torch.nn.Dropout(dropout) 
        self.activation = torch.nn.GELU()
        self.linear = torch.nn.Linear(latent_size,
                                      latent_size)
        self.mha1 = torch.nn.MultiheadAttention(latent_size,
                                                num_heads, 
                                                dropout=dropout, 
                                                batch_first=True)
        self.mha2 = torch.nn.MultiheadAttention(latent_size, 
                                                num_heads,
                                                dropout=dropout, 
                                                batch_first=True)
    
    def make_causal_mask(self, sz: int):
        return torch.triu(torch.full((sz, sz), True), diagonal=1).to(device) 
        # return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def forward(self, x):
        x_enc, x_dec = x
        x = y = x_dec
        y = self.layer_norm1(y)
        y = self.mha1(y,y,y,
                      attn_mask=self.make_causal_mask(y.shape[1]))[0]
        x = y = x + y
        y = self.layer_norm2(y)
        y = self.mha2(y, x_enc, x_enc)[0] 
        x = y = x + y
        y = self.layer_norm3(y)
        y = self.linear(y)
        y = self.activation(y)
        return x_enc, x + y


class DecoderNetwork(torch.nn.Module):
    def __init__(self,
                 num_tokens,
                 max_length,
                 latent_size = 64, 
                 num_heads = 4, 
                 n_layers = 8,
                 embedding_len = 100,
                 **kwargs):
        super().__init__(**kwargs)
        self.token_embedding = torch.nn.Embedding(num_tokens,
                                                  embedding_len,
                                                  padding_idx=0) 
        self.position_embedding = torch.nn.Embedding(max_length+1,
                                                     embedding_len,
                                                     padding_idx=0) 
        self.projection = torch.nn.Linear(embedding_len, latent_size)
        self.transformer_blocks = torch.nn.Sequential(*[
            MaskedTransformerBlock(latent_size=latent_size,
                                   num_heads=num_heads) for _ in range(n_layers)
        ])
        self.output_layer = torch.nn.Linear(latent_size,
                                            num_tokens)

    def forward(self, x_enc, x_dec):
        y = x_dec
        y = self.token_embedding(y) + \
        self.position_embedding(torch.arange(1,y.shape[1]+1).long().to(device) * torch.sign(y).long().to(device))
        y = self.projection(y)
        y = self.transformer_blocks((x_enc,y))[1]
        y = self.output_layer(y)
        return y


dec_net = DecoderNetwork(num_tokens=len(i_to_c_por), max_length=dec_x_train.shape[-1])
summary(dec_net,input_size=[enc_x_train[0:5].shape[0:2]+(64,),dec_x_train[0:5].shape], dtypes=[torch.float32, torch.long])


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
                 max_enc_length,
                 num_dec_tokens, 
                 max_dec_length,
                 latent_size = 64,
                 num_heads = 4, 
                 n_layers = 8,
                 embedding_len = 100,
                 **kwargs):
        super().__init__(output_size=num_dec_tokens, 
                         **kwargs)
        self.enc_net = EncoderNetwork(num_enc_tokens, max_enc_length, latent_size, num_heads, n_layers, embedding_len) 
        self.dec_net = DecoderNetwork(num_dec_tokens, max_dec_length, latent_size, num_heads, n_layers, embedding_len)

    def forward(self, x_enc, x_dec):
        return self.dec_net(self.enc_net(x_enc), x_dec)


enc_dec_net = EncDecNetwork(num_enc_tokens=len(i_to_c_eng), 
                            max_enc_length=enc_x_train.shape[-1],
                            num_dec_tokens=len(i_to_c_por), 
                            max_dec_length=dec_x_train.shape[-1], 
                            latent_size=64,
                            n_layers=4,
                            embedding_len=100)

summary(enc_dec_net, input_size=[enc_x_train[0:1].shape, 
                                 dec_x_train[0:1].shape], 
        dtypes=[torch.long, torch.long])





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
