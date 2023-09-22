#!/usr/bin/env python3

import numpy as np
import torch
import torchvision
import torchmetrics
import lightning.pytorch as pl 
from torchinfo import summary 
from torchview import draw_graph 
import matplotlib.pyplot as plt 
import pandas as pd

if (torch.cuda.is_available()):
    device = ("cuda")
else:
    device = ("cpu")
print(torch.cuda.is_available())