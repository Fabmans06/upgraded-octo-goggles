import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split


data = pd.read_csv("archive/DatsetFraud2.csv", delimiter=",") 
k=0
for i in range(len(data)):
    if data.isFraud[i] ==1:
        k+=1

print(k)