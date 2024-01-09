#Import utils
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

#Path to csv file
dataPath = "archive/DatsetFraud.csv"

#Read dataset, delete IsFlaggedFraud
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

#Split into input to the network and labels
inputData =  pd.DataFrame(dataset, columns=["step",
                                         "type",
                                         "amount",
                                         "nameOrig",
                                         "oldbalanceOrig",
                                         "newbalanceOrig",
                                         "nameDest",
                                         "oldbalanceDest",
                                         "newbalanceDest",
                                         ]) 
Labels = pd.DataFrame(dataset, columns=["isFraud"])

print(output)
print(input)

#Convert to tensor
input = torch.tensor(input, dtype=torch.float32)
output = torch.tensor(output, dtype=torch.float32).reshape(-1, 1)

model = nn.Sequential(
    nn.Linear(9, 16),
    nn.ReLU(),
    nn.Linear(16, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid())