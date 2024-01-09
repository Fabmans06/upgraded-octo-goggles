import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

dataPath = "archive/DatsetFraud.csv"

dataset = pd.read_csv(dataPath)
dataset = pd.DataFrame(dataset, columns=["step",
                                         "type",
                                         "amount",
                                         "nameOrig",
                                         "oldbalanceOrig",
                                         "newbalanceOrig",
                                         "nameDest",
                                         "oldbalanceDest",
                                         "newbalanceDest",
                                         "isFraud"
                                         ]) 
print(dataset)

output =  pd.DataFrame(dataset, columns=["step",
                                         "type",
                                         "amount",
                                         "nameOrig",
                                         "oldbalanceOrig",
                                         "newbalanceOrig",
                                         "nameDest",
                                         "oldbalanceDest",
                                         "newbalanceDest",
                                         ]) 
input = pd.DataFrame(dataset, columns=["isFraud"])

print(output)
print(input)

input = torch.tensor(input, dtype=torch.float32)
output = torch.tensor(output, dtype=torch.float32).reshape(-1, 1)