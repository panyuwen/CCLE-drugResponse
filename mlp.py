import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import os
import numpy as np
import pandas as pd
from tqdm import tqdm


class ImportDataSet(Dataset):
    def __init__(self, filenameX, filenameY):
        self.X = torch.tensor(pd.read_csv(filenameX, sep='\t', index_col=['cell']).values, dtype=torch.float32)
        self.Y = torch.tensor(pd.read_csv(filenameY, header=None)[0], dtype=torch.float32)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]

        return x, y


def build_data(filenameX, filenameY, batch_size, num_workers):
    ## dataloader
    data = ImportDataSet(filenameX, filenameY)
    train_data, valid_data = random_split(dataset=data, lengths=[0.9,0.1])
    
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_dataloader, valid_dataloader


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(9497, 5000),
            nn.ReLU(),
            nn.Dropout(0.15), 
            nn.Linear(5000, 2000),
            nn.ReLU(),
            nn.Dropout(0.15), 
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Dropout(0.15), 
            nn.Linear(500, 50),
            nn.ReLU(),
            nn.Dropout(0.15), 
            nn.Linear(50, 1)
        )

    def forward(self, x):
        out = self.model(x)
        return out


def train(dataloader, model, loss_fn, optimizer, device):
    train_loss = 0

    model.train()
    for batch, (X, y) in enumerate(tqdm(dataloader, leave=False)):
        X, y = X.to(device), y.to(device)

        # predict
        pred = model(X)
        loss = loss_fn(pred.reshape(y.shape), y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * len(y)
    
    return train_loss


def valid(dataloader, model, loss_fn, device):
    valid_loss = 0
    
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            
            valid_loss += loss_fn(pred.reshape(y.shape), y).item() * len(y)
    
    return valid_loss


def main():
    batch_size = 128
    num_workers = 16
    learning_rate = 1e-4
    nepoch = 200
    outputprefix = 'output'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    filenameX = 'EXP_MUT_SNP.scale9000.X.txt.gz'
    filenameY = 'EXP_MUT_SNP.scale9000.Y.txt'
    outputprefix = 'EXP_MUT_SNP.scale9000'

    ## device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device; torch version {torch.__version__}")

    ## model
    model = MLP().to(device)

    ## optimization
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_dataloader, valid_dataloader = build_data(filenameX, filenameY, batch_size=batch_size, num_workers=num_workers)
    
    ## training
    with open(outputprefix+'.loss_log.txt', 'w') as log:
        log.write('epoch\ttrain_loss\tvalid_loss\n')
    
    min_loss = float('inf')
    for epo in range(1, nepoch+1):
        train_loss = train(train_dataloader, model, loss_fn, optimizer, device)
        valid_loss = valid(valid_dataloader, model, loss_fn, device)

        print(f"epoch: {epo}; train_loss: {train_loss:>7f}; valid_loss: {valid_loss:>7f}")

        with open(outputprefix+'.loss_log.txt', 'a') as log:
            log.write(f"{epo}\t{train_loss}\t{valid_loss}\n")

        if valid_loss < min_loss:
            # checkpoint = {
            #     "epoch": epo,
            #     "train_loss": train_loss,
            #     "valid_loss": valid_loss,
            #     "model": model.state_dict(),
            #     'optim': optimizer.state_dict(),
            # }
            # torch.save(checkpoint,outputprefix+'.check_point.pth')

            checkpoint = {
                "epoch": epo,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "model": model.state_dict(),
                'optim': optimizer.state_dict(),
            }
            torch.save(checkpoint, outputprefix+'.check_point.pth')

        # model = MLP().to(device)
        # model.load_state_dict(torch.load("model.pth"))


if __name__ == '__main__':
	main()

