import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
import pytorch_lightning as L
from pytorch_lightning import loggers

csv_file_path = "archive/DatsetFraud2.csv" #Set file path to dataset
BATCH_SIZE = 256 if torch.cuda.is_available() else 64
NUM_WORKERS = max(int(os.cpu_count()/2), os.cpu_count()-3)
MAX_EPOCHS = 10 #Temporarily low for testing purposes
LATENT_DIM = 9
LEARNING_RATE = 0.02 #Can be tinkered with


class Data(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, delimiter=",") 

    def __len__(self):
        return len(self.data)    
    
    #Needs to return tensor of input data
    #def __getitem__(self, index):
    #    self.data_colums = pd.DataFrame(self.data, columns=["step", "type", "amount", "nameOrig", "oldbalanceOrig", "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest"])
    #    self.data_labels = pd.DataFrame(self.data, columns=["isFraud"])
    #    self.data_columtensor = torch.tensor(self.data_colums.values) 
    #    self.data_labeltensor = torch.tensor(self.data_labels.to_numpy().reshape(-1)).long()
    #    
    #    return self.data_colums[index], self.data_labels[index]
    
    def __getitem__(self, index):
        # Extract relevant columns for input features
        data_columns = self.data.iloc[index, [0, 1, 2, 4, 5, 7, 8, 3, 6]].values.astype(np.float32)

        # Extract label column
        data_label = self.data.iloc[index, 9].astype(np.float32)

        # Convert data to tensor
        data_tensor = torch.tensor(data_columns)

        # Convert label to tensor with correct shape
        label_tensor = torch.tensor(data_label).view(1, -1).float()

        return data_tensor, label_tensor
    
class DataModule(L.LightningDataModule):
    def __init__(
        self,
        csv_file,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
    ):
        super(DataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.csv_file = csv_file
        
    #This function needs to exist, and exists to download data. Our data however, is already downloaded
    def prepare_data(self):
        pass
    
    def setup(self, stage="none"):
        dataset = Data(self.csv_file)
        print(dataset)
        train_size = int(636260*8)
        val_size = int(636260)
        test_size = int(636260-1)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [train_size, val_size, test_size])


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=int(NUM_WORKERS/3))

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=int(NUM_WORKERS/3))

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=int(NUM_WORKERS/3))
        
    
class ANN(nn.Module):
    def __init__(self, latent_dim, shape=()):
        super(ANN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            #nn.Linear(512, 1024),
            #nn.ReLU(),
            #nn.Linear(1024, 512),
            #nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
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
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.ann = ANN(latent_dim=latent_dim)
        
    #def forward(self, z):
    #    return self.ann(z)
    
    def forward(self, z):
        return self.model(z).squeeze()

    def loss(self, y_hat ,y):
        y = y.view(-1, 1).float()
        return f.binary_cross_entropy(y_hat, y)
    
    #def loss(self, y_hat ,y):
        #return f.binary_cross_entropy(y_hat, y)
        
    #def training_step(self, batch):
    #    data, labels = batch
    #    optimizer = self.optimizers()
    #    loss = self.loss(self.ann(data), labels)
    #    self.log("loss", loss, prog_bar=True)
    #    self.manual_backward(loss)
    #    optimizer.step()
    #    optimizer.zero_grad()

    def training_step(self, batch, batch_idx):
        data, labels = batch
        optimizer = self.optimizers()
        loss = self.loss(self.ann(data), labels)
        self.log("loss", loss, prog_bar=True)
        self.manual_backward(loss)
        optimizer.step()
        optimizer.zero_grad(self)
        return loss        
    
    #def validation_step(self, batch):
    #    data, labels = batch
    #    predictions = self.ann(data)
    #    loss = self.loss(predictions, labels)
    #    self.log("Val_Loss", loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        predictions = self.ann(data)
        loss = self.loss(predictions, labels)
        self.log("Val_Loss", loss, prog_bar=True)
            
    def configure_optimizers(self):
        lr = LEARNING_RATE
            
        #https://www.kdnuggets.com/2022/12/tuning-adam-optimizer-parameters-pytorch.html
        opt = torch.optim.Adam(self.ann.parameters(), lr=lr)
        return[opt], []
        
dm = DataModule(csv_file_path)
model = Training(latent_dim=LATENT_DIM)
logger = loggers.TensorBoardLogger("tb_logs", name="my_model")
trainer = L.Trainer(
    accelerator = "auto",
    devices = "1",
    max_epochs = MAX_EPOCHS,
    enable_progress_bar=True,
    #num_sanity_val_steps=0,
    logger=logger
)
def main():
    trainer.fit(model, dm)         
main()