import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import time
import socket


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
            nn.Linear(5000, 5000),
            nn.ReLU(),
            nn.Dropout(0.15), 
            nn.Linear(5000, 2000),
            nn.ReLU(),
            nn.Dropout(0.15), 
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Dropout(0.15), 
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Dropout(0.15), 
            nn.Linear(100, 1)
        )

    def forward(self, x):
        out = self.model(x)
        return out


def train(dataloader, model, loss_fn, optimizer, device):
    train_loss = 0
    count = 0

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
        count += len(y)
    
    return train_loss / count


def valid(dataloader, model, loss_fn, device):
    valid_loss = 0
    count = 0
    
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            
            valid_loss += loss_fn(pred.reshape(y.shape), y).item() * len(y)
            count += len(y)
    
    return valid_loss / count


def plot_loss(loss_file, fig_path):
    losses = pd.read_csv(loss_file, sep='\t')
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    ax1 = sns.lineplot(data=losses, x='epoch', y='train_loss', ax=axes[0])
    ax2 = sns.lineplot(data=losses, x='epoch', y='valid_loss', ax=axes[1])

    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax1.set_ylabel('loss')
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax2.set_ylabel('loss')

    axes[0].set_title("Train Loss")
    axes[1].set_title("Valid Loss")
    plt.savefig(fig_path)


def timer(start_time, end_time):
    t = float(end_time) - float(start_time)
    t_m,t_s = divmod(t, 60)   
    t_h,t_m = divmod(t_m, 60)
    r_t = str(int(t_h)).zfill(2) + ":" + str(int(t_m)).zfill(2) + ":" + str(int(t_s)).zfill(2)

    return r_t
    

def main(batch_size, num_workers, learning_rate, nepoch, outputprefix, cudadevice, filenameX, filenameY):
    start_time = time.perf_counter()
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cudadevice)

    with open(outputprefix + '.logfile', 'w') as log:
        log.write('Hostname: '+socket.gethostname()+'\n')
        log.write('Working directory: '+os.getcwd()+'\n')
        log.write(f'Using {device} device\n')
        log.write(f'torch version {torch.__version__}\n\n')       
        log.write('Start time: '+time.strftime("%Y-%m-%d %X",time.localtime())+'\n')

    ## model
    model = MLP().to(device)

    ## optimization
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.MSELoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    ## training
    with open(outputprefix+'.loss_log.txt', 'w') as log:
        log.write('epoch\ttrain_loss\tvalid_loss\n')

    train_dataloader, valid_dataloader = build_data(filenameX, filenameY, batch_size=batch_size, num_workers=num_workers)

    min_loss = float('inf')
    for epo in range(1, nepoch+1):
        train_loss = train(train_dataloader, model, loss_fn, optimizer, device)
        valid_loss = valid(valid_dataloader, model, loss_fn, device)

        print(f"epoch: {epo}; train_loss: {train_loss:>7f}; valid_loss: {valid_loss:>7f}")
        with open(outputprefix+'.loss_log.txt', 'a') as log:
            log.write(f"{epo}\t{train_loss}\t{valid_loss}\n")

        if valid_loss < min_loss:
            checkpoint = {
                "epoch": epo,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "model": model.state_dict(),
                'optim': optimizer.state_dict(),
            }
            torch.save(checkpoint, outputprefix+'.check_point.pth')

    plot_loss(outputprefix+'.loss_log.txt', outputprefix+'.loss_log.pdf')

    ## to reload model
    # model = MLP().to(device)
    # model.load_state_dict(torch.load("model.pth"))

    end_time = time.perf_counter()
    with open(outputprefix + '.logfile','a') as log:
        log.write('End time: '+time.strftime("%Y-%m-%d %X",time.localtime())+'\n')
        log.write('Lasting: '+timer(start_time, end_time)+'\n\n')
        log.write(f'Parameters: \n\tbatch_size:\t{batch_size} \n\tnum_workers:\t{num_workers} \n\tlearning_rate:\t{learning_rate} \n\tnum_epoch:\t{nepoch} \n\toutputprefix:\t{outputprefix} \n\tcuda device:\t{cudadevice} \n\tfilenameX:\t{filenameX} \n\tfilenameY:\t{filenameY} \n\n')
        log.write(f'output: \n\t{outputprefix}.check_point.pth\n\t{outputprefix}.loss_log.txt\n\t{outputprefix}.loss_log.pdf\n\t{outputprefix}.logfile\n\n')
        log.write('Model structure:\n')
        log.write(str(model))


if __name__ == '__main__':
	main(
        batch_size = 128, 
        num_workers = 16, 
        learning_rate = 1e-5, 
        nepoch = 200, 
        outputprefix = 'EXP_MUT_SNP.scale9000.1', 
        cudadevice = '0', 
        filenameX = 'EXP_MUT_SNP.scale9000.X.txt.gz', 
        filenameY = 'EXP_MUT_SNP.scale9000.Y.txt'
    )

