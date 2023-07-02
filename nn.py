import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import time
import socket
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score


def plot_loss(loss_file, fig_path):
    losses = pd.read_csv(loss_file, sep='\t', index_col=['epoch'], usecols=['epoch','train_loss','valid_loss'])

    plt.clf()

    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    ax = sns.lineplot(data=losses)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_ylabel('loss')

    plt.savefig(fig_path)


class ImportDataSet(Dataset):
    def __init__(self, dataX, dataY):
        self.X = torch.tensor(dataX.values, dtype=torch.float32)
        self.Y = torch.tensor(dataY.values, dtype=torch.float32)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        return x, y
    
    def __featuresize__(self):
        return len(self.X[0])


def build_data(inputX, inputY, rankEXP, rankSNP, rankMUT, featuresize, labeltype, batch_size, num_workers):
    X = pd.read_csv(inputX, sep='\t', index_col=['cell'])
    Y = pd.read_csv(inputY, header=None)
    if labeltype == 'discrete':
        Y[1] = (Y[0] > 0) - 0
        Y[0] = 1 - Y[1]

    ## featuresize
    rankexp = pd.read_csv(rankEXP, sep='\t', usecols=['Unnamed: 0','tot_rank'])
    rankmut = pd.read_csv(rankMUT, sep='\t', usecols=['Unnamed: 0','tot_rank'])
    ranksnp = pd.read_csv(rankSNP, sep='\t', usecols=['Unnamed: 0','tot_rank'])

    featureprop = {'18K':0.4, '10K':0.2, '5K':0.1, '1K':0.02}[featuresize]
    rankexp = list(rankexp.head(n=int(rankexp.shape[0]*featureprop))['Unnamed: 0'])
    rankmut = list(rankmut.head(n=int(rankmut.shape[0]*featureprop))['Unnamed: 0'])
    ranksnp = list(ranksnp.head(n=int(ranksnp.shape[0]*featureprop))['Unnamed: 0'])

    cols = list(X.columns)
    cols = [x for x in cols if x.split('_')[-1] not in ['SNP','EXP','MUT']]

    X = X[cols + rankexp + rankmut + ranksnp]

    celltypelist = list(set(X.index))
    X.drop(celltypelist, axis=1, inplace=True)

    ## dataloader
    data = ImportDataSet(X, Y)
    train_data, valid_data = random_split(dataset=data, lengths=[0.9,0.1])

    train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)

    return train_dataloader, valid_dataloader, X.shape[1]


class MLP(nn.Module):
    def __init__(self, in_featuresize, out_featuresize, dropoutProb=0.15):
        super().__init__()
        
        in_dim, out_dim = in_featuresize, in_featuresize * 2 // 3
        self.block_init = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropoutProb)
        )

        in_dim, out_dim = out_dim, out_dim // 2
        self.block_2 = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropoutProb)
        )

        in_dim, out_dim = out_dim, out_dim // 2
        self.block_3 = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropoutProb)
        )

        in_dim, out_dim = out_dim, out_dim // 4
        self.block_4 = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropoutProb)
        )

        in_dim, out_dim = out_dim, out_dim // 8
        self.block_5 = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropoutProb)
        )

        in_dim, out_dim = out_dim, out_featuresize
        self.block_last = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        out = self.block_init(x)
        out = self.block_2(out)
        out = self.block_3(out)
        out = self.block_4(out)
        out = self.block_5(out)
        out = self.block_last(out)

        return out


class continuous_metrics():
    def __init__(self):
        self.R2_numerator_sum = 0
        self.R2_denominator_sum = 0
        self.MAE_sum = 0
        self.MSE_sum = 0
        self.count = 0
    
    def reset(self):
        self.R2_numerator_sum = 0
        self.R2_denominator_sum = 0
        self.MAE_sum = 0
        self.MSE_sum = 0
        self.count = 0

    def update(self, y_true, y_pred):
        y_true = y_true.cpu().detach().numpy()
        y_pred = y_pred.cpu().detach().numpy()

        y_true = y_true.reshape(y_pred.shape)
        # R2
        ybar = y_true.mean()
        self.R2_numerator_sum += ((y_true - y_pred)**2).sum()
        self.R2_denominator_sum += ((y_true - ybar)**2).sum()
        # mae
        self.MAE_sum += mean_absolute_error(y_true, y_pred) * len(y_true)
        # mse
        self.MSE_sum += mean_squared_error(y_true, y_pred) * len(y_true)
        # count
        self.count += len(y_true)
    
    def output(self):
        R2 = 1 - self.R2_numerator_sum / self.R2_denominator_sum
        MAE = self.MAE_sum / self.count
        MSE = self.MSE_sum / self.count

        return R2, MAE, MSE


class discrete_metrics():
    def __init__(self):
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.TN = 0
        self.AUC_sum = 0
        self.nbatch = 0

    def reset(self):
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.TN = 0
        self.AUC_sum = 0
        self.nbatch = 0

    def update(self, y_true, y_pred):
        y_pred_prob = F.softmax(y_pred, dim=1)
        y_pred_clas = y_pred_prob.argmax(1)

        y_true = y_true[:, 1]
        
        self.TP += sum((y_pred_clas == 1) & (y_true == 1))
        self.FP += sum((y_pred_clas == 1) & (y_true == 0))
        self.FN += sum((y_pred_clas == 0) & (y_true == 1))
        self.TN += sum((y_pred_clas == 0) & (y_true == 0))

        self.AUC_sum += roc_auc_score(y_true.cpu().detach().numpy(), y_pred_prob[:,1].cpu().detach().numpy())

        self.nbatch += 1
    
    def output(self):
        acc = (self.TP + self.TN) * 1.0 / (self.TP + self.TN + self.FP + self.FN)
        precision = self.TP * 1.0 / (self.TP + self.FP)
        recall = self.TP * 1.0 / (self.TP + self.FN)
        f1 = 2.0 * self.TP / (2 * self.TP + self.FP + self.FN)
        auc = self.AUC_sum / self.nbatch

        return auc, f1.item(), precision.item(), recall.item(), acc.item()


def MLP_train(dataloader, model, loss_fn, optimizer, device, labeltype):
    train_loss = 0
    count = 0
    if labeltype == 'continuous':
        metrics_calculator = continuous_metrics()
    else:
        metrics_calculator = discrete_metrics()

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

        # metrics
        metrics_calculator.update(y, pred)

    return train_loss / count, *metrics_calculator.output()


def MLP_valid(dataloader, model, loss_fn, device, labeltype):
    valid_loss = 0
    count = 0
    if labeltype == 'continuous':
        metrics_calculator = continuous_metrics()
    else:
        metrics_calculator = discrete_metrics()
    
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            
            valid_loss += loss_fn(pred.reshape(y.shape), y).item() * len(y)
            count += len(y)

            metrics_calculator.update(y, pred)

    return valid_loss / count, *metrics_calculator.output()


def MLP_train_valid(train_dataloader, valid_dataloader, featurecount, nepoch, labeltype, device, lr, output):
    if labeltype == 'continuous':
        model = MLP(featurecount, 1).to(device)
        loss_fn = nn.MSELoss()
    else:
        model = MLP(featurecount, 2).to(device)
        loss_fn = nn.CrossEntropyLoss()

    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    with open(output+'.metrics.txt', 'w') as log:
        if labeltype == 'continuous':
            log.write('epoch\ttrain_loss\ttrain_R2\ttrain_MAE\ttrain_MSE\tvalid_loss\tvalid_R2\tvalid_MAE\tvalid_MSE\n')
        else:
            log.write('epoch\ttrain_loss\ttrain_auc\ttrain_f1\ttrain_precision\ttrain_recall\ttrain_acc\tvalid_loss\tvalid_auc\tvalid_f1\tvalid_precision\tvalid_recall\tvalid_acc\n')

    min_loss = float('inf')
    for epo in range(1, nepoch+1):
        train_metrics_list = MLP_train(train_dataloader, model, loss_fn, optimizer, device, labeltype)
        valid_metrics_list = MLP_valid(valid_dataloader, model, loss_fn, device, labeltype)

        print(f"epoch: {epo}; train_loss: {train_metrics_list[0]:>7f}; valid_loss: {valid_metrics_list[0]:>7f}")
        with open(output+'.metrics.txt', 'a') as log:
            train_metrics_list = list(map(str, train_metrics_list))
            valid_metrics_list = list(map(str, valid_metrics_list))
            log.write('\t'.join([str(epo)] + train_metrics_list + valid_metrics_list) + '\n')

        if float(valid_metrics_list[0]) < min_loss:
            checkpoint = {
                "epoch": epo,
                "train_loss": train_metrics_list[0],
                "valid_loss": valid_metrics_list[0],
                "model": model.state_dict(),
                'optim': optimizer.state_dict(),
            }
            torch.save(checkpoint, output+'.check_point.pth')
            min_loss = float(valid_metrics_list[0])

    plot_loss(output+'.metrics.txt', output+'.metrics.pdf')

    return str(model)
    ## to reload model
    # model = MLP().to(device)
    # model.load_state_dict(torch.load("model.pth"))


def Attention_train_valid():
    return None


def timer(start_time, end_time):
    t = float(end_time) - float(start_time)
    t_m,t_s = divmod(t, 60)   
    t_h,t_m = divmod(t_m, 60)
    r_t = str(int(t_h)).zfill(2) + ":" + str(int(t_m)).zfill(2) + ":" + str(int(t_s)).zfill(2)

    return r_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputX", type=str, required=False, \
                        default="/glusterfs/home/local_pan_yuwen/research/20230608_CCLE2012/03.NN/EXP_MUT_SNP.scale18K.X.txt.gz")
    parser.add_argument("--inputY", type=str, required=False, \
                        default="/glusterfs/home/local_pan_yuwen/research/20230608_CCLE2012/03.NN/EXP_MUT_SNP.scale18K.Y.txt")
    parser.add_argument("--rankEXP", type=str, required=False, \
                        default="/glusterfs/home/local_pan_yuwen/research/20230608_CCLE2012/03.NN/cor/EXP_logIC50.cor_rank.txt")
    parser.add_argument("--rankSNP", type=str, required=False, \
                        default="/glusterfs/home/local_pan_yuwen/research/20230608_CCLE2012/03.NN/cor/SNP_logIC50.cor_rank.txt")
    parser.add_argument("--rankMUT", type=str, required=False, \
                        default="/glusterfs/home/local_pan_yuwen/research/20230608_CCLE2012/03.NN/cor/MUT_logIC50.cor_rank.txt")
    
    parser.add_argument("--batch-size", type=int, required=False, default=128)
    parser.add_argument("--num-workers", type=int, required=False, default=16, help="for DataLoader")
    parser.add_argument("--lr", type=float, required=False, default=5e-5)
    parser.add_argument("--nepoch", type=int, required=False, default=500)
    parser.add_argument("--device-id", type=str, required=False, choices=['0','1','2','3'], default='0')

    parser.add_argument("--feature-size", type=str, required=True, choices=['18K','10K','5K','1K'])
    parser.add_argument("--label-type", type=str, required=True, choices=['continuous', 'discrete'])
    parser.add_argument("--model-type", type=str, required=True, choices=['MLP', 'Attention'])
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()


    start_time = time.perf_counter()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    with open(args.out + '.logfile', 'w') as log:
        log.write('Hostname: '+socket.gethostname()+'\n')
        log.write('Working directory: '+os.getcwd()+'\n')
        log.write(f'Using {device} device\n')
        log.write(f'torch version {torch.__version__}\n\n')
        log.write('Start time: '+time.strftime("%Y-%m-%d %X",time.localtime())+'\n')

    ######################################
    
    train_dataloader, valid_dataloader, featurecount = build_data(args.inputX, args.inputY, args.rankEXP, args.rankSNP, args.rankMUT, args.feature_size, args.label_type, args.batch_size, args.num_workers)

    if args.model_type == 'MLP':
        modelstructure = MLP_train_valid(train_dataloader, valid_dataloader, featurecount, args.nepoch, args.label_type, device, args.lr, args.out)
    else:
        modelstructure = Attention_train_valid()
        print('not supported yet')

    ######################################

    end_time = time.perf_counter()
    with open(args.out + '.logfile', 'a') as log:
        log.write('End time: '+time.strftime("%Y-%m-%d %X",time.localtime())+'\n')
        log.write('Lasting: '+timer(start_time, end_time)+'\n\n')
        log.write(f'Parameters: \n\tbatch_size:\t{args.batch_size} \n\tnum_workers:\t{args.num_workers} \n\tlearning_rate:\t{args.lr} \n\tnum_epoch:\t{args.nepoch} \n\toutputprefix:\t{args.out} \n\tcuda device:\t{args.device_id} \n\tfeature size:\t{args.feature_size} \n\tlabel type:\t{args.label_type} model type:\t{args.model_type} \n\n')
        log.write('Model structure:\n')
        log.write(str(modelstructure))


if __name__ == '__main__':
	main()


# class argstmp():
#     def __init__(self):
#         self.inputX = "/glusterfs/home/local_pan_yuwen/research/20230608_CCLE2012/03.NN/EXP_MUT_SNP.scale18K.X.txt.gz"
#         self.inputY = "/glusterfs/home/local_pan_yuwen/research/20230608_CCLE2012/03.NN/EXP_MUT_SNP.scale18K.Y.txt"
#         self.rankEXP = "/glusterfs/home/local_pan_yuwen/research/20230608_CCLE2012/03.NN/cor/EXP_logIC50.cor_rank.txt"
#         self.rankSNP = "/glusterfs/home/local_pan_yuwen/research/20230608_CCLE2012/03.NN/cor/SNP_logIC50.cor_rank.txt"
#         self.rankMUT = "/glusterfs/home/local_pan_yuwen/research/20230608_CCLE2012/03.NN/cor/MUT_logIC50.cor_rank.txt"
#         self.batch_size = 128
#         self.num_workers = 16
#         self.lr = 5e-5
#         self.nepoch = 500
#         self.device_id = '0'
#         self.feature_size = '5K'
#         self.label_type = 'continuous'
#         self.model_type = 'MLP'
#         self.out = 'test'

# args = argstmp()
