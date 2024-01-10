import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda as cuda
import os

BATCH_SIZE = 256 if torch.cuda.is_available() else 64
NUM_WORKERS = int(os.cpu_count()/2)
# load the dataset, split into input (X) and output (y) variables

dataset = np.loadtxt('archive/creditcard.csv', delimiter=',')
X = dataset[:,0:10]
y = dataset[:,10]

X = torch.tensor(X, dtype=torch.float32, device=cuda)
y = torch.tensor(y, dtype=torch.float32, device=cuda).reshape(-1, 1)

# define the model
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 128),
    nn.ReLU(),
    nn.Linear(128, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)
print(model)

# train the model
loss_fn   = nn.BCELoss()  # binary cross entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 100
batch_size = 100
#TODO: Fix training, testing, and validation cycle
for epoch in range(n_epochs):
    for i in range(0, len(X), batch_size):
        Xbatch = X[i:i+batch_size]
        y_pred = model(Xbatch)
        ybatch = y[i:i+batch_size]
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Finished epoch {epoch}, latest loss {loss}')

# compute accuracy (no_grad is optional)
with torch.no_grad():
    y_pred = model(X)
accuracy = (y_pred.round() == y).float().mean()
print(f"Accuracy {accuracy}")

#Continue next:
#https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html 
#https://www.datacamp.com/tutorial/pytorch-tutorial-building-a-simple-neural-network-from-scratch