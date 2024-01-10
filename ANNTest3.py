import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import Dataloader, TensorDataset, Dataset, random_split
import pytorch_lightning as L

csv_file_path = "" #Set file path to dataset
BATCH_SIZE = 256 if torch.cuda.is_available() else 64
NUM_WORKERS = int(os.cpu_count()/2)
MAX_EPOCHS = 5 #Temporarily low for testing purposes
LATENT_DIM = 10
LEARNING_RATE = 0.002 #Can be tinkered with


class Data(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read(csv_file, delimiter=",") 
        
        
    #Needs to return tensor of input data
    def __getitem__(self, index):
        pass
    #Convert data to tensor
    #Return tensor
    
class DataModule(L.LightningDataModule):
    def __init__(
        self,
        csv_file,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
    ):
        super(DataModule, self).__init()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.csv_file = csv_file
        
    def prepare_data(self):
        pass
    
    def setup(self, stage="none"):
        dataset = Data(self.csv_file)
        train_size = int(80)
        val_size = int(10)
        test_size = int(10)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset[train_size, val_size, test_size])


    def train_dataloader(self):
        return Dataloader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return Dataloader(self.val_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return Dataloader(self.test_dataset, batch_size=self.batch_size, shuffle=True)
        
    
class ANN(nn.Module):
    def __init__(self, latent_dim, shape):
        super(ANN, self).__init__()
        self.model = nn.Sequential(
        nn.Linear(latent_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
        nn.ReLU(),
        nn.Linear(4, int(np.prod(shape))),
        nn.Sigmoid()
)
        def forward(self, z):
            return self.model(z)
    
class Training(L.LightningModule):
    def __init__(
        self, 
        latent_dim: int = LATENT_DIM,
        batch_size: int = BATCH_SIZE,
        **kwargs,
    ):
        super().__init__(),
        self.save_hyperparameters,
        self.automatic_optimization = False,
        data_shape = ()
        self.ann = ANN(latent_dim=self.hparams.latent_dim, shape = data_shape)
        
        def forward(self, z):
            return self.ann(z)
        
        def loss(self, y_hat ,y):
            return f.log_cross_entropy(y_hat, y)
        
        def training_step(self, batch):
            optimizer = self.optimizers()
            data = 
            labels = 
            loss = self.loss(self.ann(data), labels)
            self.log("loss", loss, prog_bar=True)
            self.manual_backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
        def validation_step(self, batch):
            #Need to figure this out
            pass
            
        def configure_optimizers(self):
            lr = LEARNING_RATE
            
            #https://www.kdnuggets.com/2022/12/tuning-adam-optimizer-parameters-pytorch.html
            opt = torch.optim.Adam(self.ann.parameters(), lr=lr)
            return[opt], []

def main():
    dm = DataModule(csv_file_path)
    model = Training(latent_dim=LATENT_DIM)
    trainer = L.trainer(
        accelerator = "auto",
        devices = "1",
        max_epochs = MAX_EPOCHS
    )
    trainer.fit(model, dm)         
