#Import utils
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimizers
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

print(Labels)
print(input)

#Convert to tensor
input = torch.tensor(input, dtype=torch.float32)
output = torch.tensor(Labels, dtype=torch.float32).reshape(-1, 1)

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

loss_fn = nn.BCELoss()  #Binary cross entropy loss
optimizer = optimizers.Adam(model.parameters(), lr=0.001)

n_epochs = 15
batch_size = 150

for epoch in range(n_epochs):
    for i in range(0, len(input), batch_size):
        inputBatch = input[i:i+batch_size]
        output_prediction = model(inputBatch)
        outputBatch = output[i:i+batch_size]
        loss = loss_fn(output_prediction, outputBatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Finished epoch {epoch}, latest loss {loss}')