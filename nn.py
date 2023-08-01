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
from typing import Optional, Tuple


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
        self.X = torch.tensor(dataX, dtype=torch.float32)
        self.Y = torch.tensor(dataY, dtype=torch.float32)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        return x, y
    
    def __featuresize__(self):
        return np.prod(self.X.shape) / self.X.shape[0]


def dim_pad(X, featurelist, d_model):
    if len(featurelist) % d_model == 0:
        return X[featurelist].values
    else:
        return np.pad(X[featurelist].values, ((0, 0), (0, d_model-len(featurelist)%d_model)), 'constant', constant_values=0)


def converf4atten(X, d_model):
    """
    Args:
        X (pd.DataFrame), #cell x #feature (indictors + SNP+EXP+MUT)
    Output:
        X (np.array), #cell * len x d_model
    """
    cols = list(X.columns)
    
    idxf = [x for x in cols if x.split('_')[-1] not in ['SNP','EXP','MUT']]
    snpf = [x for x in cols if x.split('_')[-1] == 'SNP']
    expf = [x for x in cols if x.split('_')[-1] == 'EXP']
    mutf = [x for x in cols if x.split('_')[-1] == 'MUT']
    
    Xpad = np.concatenate([dim_pad(X, idxf, d_model), dim_pad(X, snpf, d_model), dim_pad(X, expf, d_model), dim_pad(X, mutf, d_model)], axis=1).reshape([X.shape[0], -1, d_model])

    return Xpad


def build_data(modeltype, inputX, inputY, rankEXP, rankSNP, rankMUT, featuresize, labeltype, batch_size, num_workers, output, d_model):
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

    with open(output + '.featurelist.txt', 'w') as fin:
        for i in list(X.columns):
            fin.write(i + '\n')

    ## dataloader
    if modeltype == 'MLP':
        data = ImportDataSet(X.values, Y.values)
    else:
        data = ImportDataSet(converf4atten(X, d_model), Y.values)
        
    train_data, valid_data = random_split(dataset=data, lengths=[0.9,0.1])

    train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)

    return train_dataloader, valid_dataloader, int(data.__featuresize__())


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
    
    def check(self):
        return {'R2_numerator_sum': self.R2_numerator_sum, 
                'R2_denominator_sum': self.R2_denominator_sum, 
                'MAE_sum': self.MAE_sum, 
                'MSE_sum': self.MSE_sum, 
                'count': self.count}

    def output(self):
        R2 = 1 - self.R2_numerator_sum / self.R2_denominator_sum
        MAE = self.MAE_sum / self.count
        MSE = self.MSE_sum / self.count

        return R2, MAE, MSE


# class discrete_metrics():
#     """
#     For one-class
#     """
#     def __init__(self):
#         self.TP = 0; self.FP = 0; self.FN = 0; self.TN = 0; self.AUC_sum = 0; self.nbatch = 0

#     def reset(self):
#         self.TP = 0; self.FP = 0; self.FN = 0; self.TN = 0; self.AUC_sum = 0; self.nbatch = 0

#     def update(self, y_true, y_pred):
#         y_pred_prob = F.softmax(y_pred, dim=1)
#         y_pred_clas = y_pred_prob.argmax(1)
#         y_true = y_true[:, 1]
#         self.TP += sum((y_pred_clas == 1) & (y_true == 1))
#         self.FP += sum((y_pred_clas == 1) & (y_true == 0))
#         self.FN += sum((y_pred_clas == 0) & (y_true == 1))
#         self.TN += sum((y_pred_clas == 0) & (y_true == 0))
#         self.AUC_sum += roc_auc_score(y_true.cpu().detach().numpy(), y_pred_prob[:,1].cpu().detach().numpy())
#         self.nbatch += 1
    
#     def output(self):
#         acc = (self.TP + self.TN) * 1.0 / (self.TP + self.TN + self.FP + self.FN)
#         precision = self.TP * 1.0 / (self.TP + self.FP)
#         recall = self.TP * 1.0 / (self.TP + self.FN)
#         f1 = 2.0 * self.TP / (2 * self.TP + self.FP + self.FN)
#         auc = self.AUC_sum / self.nbatch
#         return auc, f1.item(), precision.item(), recall.item(), acc.item()


class discrete_metrics():
    """
    For multi-class
    Returns: 
        class1-metrics;class2-metrics;...;macro(average)-metrics
    """
    def __init__(self):
        self.TP = []
        self.FP = []
        self.FN = []
        self.TN = []
        self.AUC_sum = []
        self.nbatch = 0

    def reset(self):
        self.TP = []
        self.FP = []
        self.FN = []
        self.TN = []
        self.AUC_sum = []
        self.nbatch = 0

    def update(self, y_true, y_pred):
        y_pred_prob = F.softmax(y_pred, dim=1)
        y_pred_clas = y_pred_prob.argmax(1)
        
        for i in range(y_true.shape[1]):
            y_true_ = y_true[:, i].cpu().detach().numpy()
            y_pred_clas_ = (y_pred_clas == i).cpu().detach().numpy() - 0
            y_pred_prob_ = y_pred_prob[:, i].cpu().detach().numpy()

            if len(self.TP) <= i:
                self.TP.append(sum((y_pred_clas_ == 1) & (y_true_ == 1)))
                self.FP.append(sum((y_pred_clas_ == 1) & (y_true_ == 0)))
                self.FN.append(sum((y_pred_clas_ == 0) & (y_true_ == 1)))
                self.TN.append(sum((y_pred_clas_ == 0) & (y_true_ == 0)))
                self.AUC_sum.append(roc_auc_score(y_true_, y_pred_prob_))
            else:
                self.TP[i] += sum((y_pred_clas_ == 1) & (y_true_ == 1))
                self.FP[i] += sum((y_pred_clas_ == 1) & (y_true_ == 0))
                self.FN[i] += sum((y_pred_clas_ == 0) & (y_true_ == 1))
                self.TN[i] += sum((y_pred_clas_ == 0) & (y_true_ == 0))
                self.AUC_sum[i] += roc_auc_score(y_true_, y_pred_prob_)

        self.nbatch += 1
    
    def check(self):
        return {'TP': self.TP, 
                'FP': self.FP, 
                'FN': self.FN, 
                'TN': self.TN, 
                'AUC_sum': self.AUC_sum, 
                'nbatch': self.nbatch}

    def output(self):
        self.TP = np.array(self.TP)
        self.FP = np.array(self.FP)
        self.FN = np.array(self.FN)
        self.TN = np.array(self.TN)
        self.AUC_sum = np.array(self.AUC_sum)
        
        acc = (self.TP + self.TN) * 1.0 / (self.TP + self.TN + self.FP + self.FN)
        precision = self.TP * 1.0 / (self.TP + self.FP)
        recall = self.TP * 1.0 / (self.TP + self.FN)
        f1 = 2.0 * self.TP / (2 * self.TP + self.FP + self.FN)
        auc = self.AUC_sum / self.nbatch

        return ';'.join(map(str, auc)) +';'+ str(auc.mean()), \
                ';'.join(map(str, f1)) +';'+ str(f1.mean()), \
                ';'.join(map(str, precision)) +';'+ str(precision.mean()), \
                ';'.join(map(str, recall)) +';'+ str(recall.mean()), \
                ';'.join(map(str, acc)) +';'+ str(acc.mean())


def run_train(dataloader, model, loss_fn, optimizer, device, labeltype):
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


def run_valid(dataloader, model, loss_fn, device, labeltype):
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
        train_metrics_list = run_train(train_dataloader, model, loss_fn, optimizer, device, labeltype)
        valid_metrics_list = run_valid(valid_dataloader, model, loss_fn, device, labeltype)

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
    # model = MLP(in_dim, out_dim).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # checkpoint = torch.load(output + ".check_point.pth")
    # model.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optim'])


class ScaledDotProductAttention(nn.Module):
    """
    # https://github.com/sooftware/attentions
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values

    Args: dim, mask
        dim (int): dimention of attention, d_head: d_model // num_head
        mask (torch.Tensor): tensor containing indices to be masked

    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked

    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # batch_size·num_head x len x d_head for Q/K/V
        # d_head = d_model x num_head
        
        # attention scores
        # B·N x l x l
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        if mask is not None:
            score.masked_fill_(mask.view(score.size()), -float('Inf'))

        # attention weights
        # softmax activation applied along the last dimension, d_head
        # Batch·#head x len x len
        attn = F.softmax(score, -1)
        
        # Attended Value
        # Batch·#head x len x d_head
        context = torch.bmm(attn, value)
        
        return context, attn


class MultiHeadAttention(nn.Module):
    """
    # https://github.com/sooftware/attentions
    Multi-Head Attention proposed in "Attention Is All You Need"
    Instead of performing a single attention function with d_model-dimensional keys, values, and queries,
    project the queries, keys and values h times with different, learned linear projections to d_head dimensions.
    These are concatenated and once again projected, resulting in the final values.
    Multi-head attention allows the model to jointly attend to information from different representation
    subspaces at different positions.

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_o
        where head_i = Attention(Q · W_q, K · W_k, V · W_v)

    Args:
        d_model (int): The dimension of keys / values / quries (default: 512)
        num_heads (int): The number of attention heads. (default: 8)

    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): In transformer, three different ways:
            Case 1: come from previoys decoder layer
            Case 2: come from the input embedding
            Case 3: come from the output embedding (masked)

        - **key** (batch, k_len, d_model): In transformer, three different ways:
            Case 1: come from the output of the encoder
            Case 2: come from the input embeddings
            Case 3: come from the output embedding (masked)

        - **value** (batch, v_len, d_model): In transformer, three different ways:
            Case 1: come from the output of the encoder
            Case 2: come from the input embeddings
            Case 3: come from the output embedding (masked)

        - **mask** (-): tensor containing indices to be masked

    Returns: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features.
        - **attn** (batch * num_heads, v_len): tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, d_model: int, num_heads: int = 8):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model % num_heads should be zero."

        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.scaled_dot_attn = ScaledDotProductAttention(self.d_head)
        self.query_proj = nn.Linear(d_model, self.d_head * num_heads)
        self.key_proj = nn.Linear(d_model, self.d_head * num_heads)
        self.value_proj = nn.Linear(d_model, self.d_head * num_heads)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = value.size(0)

        # Linear transformations followed by Reshaping the inputs for multihead splitting
        # B: batch size; N: num_head; D: d_head
        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)  # B x Q_LEN x N x D
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head)        # B x K_LEN x N x D
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head)  # B x V_LEN x N x D

        query = query.permute(2, 0, 1, 3).contiguous().view(self.num_heads * batch_size, -1, self.d_head)  # B·N x Q_LEN x D
        key = key.permute(2, 0, 1, 3).contiguous().view(self.num_heads * batch_size, -1, self.d_head)      # B·N x K_LEN x D
        value = value.permute(2, 0, 1, 3).contiguous().view(self.num_heads * batch_size, -1, self.d_head)  # B·N x V_LEN x D

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # B x N x Q_LEN x K_LEN

        # attn:    attention weight, B·N x l x l
        # context: attended Value, B·N x l x d_head
        context, attn = self.scaled_dot_attn(query, key, value, mask)

        # -> N x B x l x d_head
        context = context.view(self.num_heads, batch_size, -1, self.d_head)

        # -> B x l x N x d_head -> B x l x N·d_head = B x l x d_model
        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.num_heads * self.d_head)  # B x T x ND

        return context, attn


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta

        return out


class FeedForward(nn.Module):
    def __init__(self, d_model, hidden, dropoutProb=0.15):
        super().__init__()
        self.linear = nn.Linear(d_model, hidden)
        self.activation = nn.ReLU()
        # self.dropout = nn.Dropout(dropoutProb)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        # x = self.dropout(x)
        return x


class Attention(nn.Module):
    def __init__(self, l, d_model, out_featuresize, dropoutProb=0.15):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(d_model)
        self.layernorm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropoutProb)
        self.feedforward = FeedForward(d_model, d_model)
        self.collapse_dim_d_model = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropoutProb),
            nn.Linear(d_model, 1),
            nn.ReLU()
        )
        self.collapse_dim_len = nn.Sequential(
            nn.Linear(l, l),
            nn.ReLU(),
            nn.Dropout(dropoutProb),
            nn.Linear(l, out_featuresize)
        )
        
    def forward(self, x):
        x_, attn = self.multi_head_attention(query=x, key=x, value=x)
        x_ = self.dropout(x_)
        x = self.layernorm(x + x_)
        
        x_ = self.feedforward(x)
        x_ = self.dropout(x_)
        x = self.layernorm(x + x_)

        x = self.collapse_dim_d_model(x)
        x = torch.squeeze(x, dim=-1)
        x = self.collapse_dim_len(x)

        return x


def Attention_train_valid(train_dataloader, valid_dataloader, l, d_model, nepoch, labeltype, device, lr, output):
    if labeltype == 'continuous':
        model = Attention(l, d_model, 1).to(device)
        loss_fn = nn.MSELoss()
    else:
        model = Attention(l, d_model, 2).to(device)
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
        train_metrics_list = run_train(train_dataloader, model, loss_fn, optimizer, device, labeltype)
        valid_metrics_list = run_valid(valid_dataloader, model, loss_fn, device, labeltype)

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


def timer(start_time, end_time):
    t = float(end_time) - float(start_time)
    t_m,t_s = divmod(t, 60)   
    t_h,t_m = divmod(t_m, 60)
    r_t = str(int(t_h)).zfill(2) + ":" + str(int(t_m)).zfill(2) + ":" + str(int(t_s)).zfill(2)

    return r_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputX", type=str, required=False, \
                        default="EXP_MUT_SNP.scale18K.X.txt.gz")
    parser.add_argument("--inputY", type=str, required=False, \
                        default="EXP_MUT_SNP.scale18K.Y.txt")
    parser.add_argument("--rankEXP", type=str, required=False, \
                        default="cor/EXP_logIC50.cor_rank.txt")
    parser.add_argument("--rankSNP", type=str, required=False, \
                        default="cor/SNP_logIC50.cor_rank.txt")
    parser.add_argument("--rankMUT", type=str, required=False, \
                        default="cor/MUT_logIC50.cor_rank.txt")
    
    parser.add_argument("--batch-size", type=int, required=False, default=128)
    parser.add_argument("--num-workers", type=int, required=False, default=16, help="for DataLoader")
    parser.add_argument("--lr", type=float, required=False, default=5e-5)
    parser.add_argument("--nepoch", type=int, required=False, default=500)
    parser.add_argument("--device-id", type=str, required=False, choices=['0','1','2','3'], default='0')
    parser.add_argument("--d_model", type=int, required=False, default=64)
    
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
    
    train_dataloader, valid_dataloader, featurecount = build_data(args.model_type, args.inputX, args.inputY, args.rankEXP, args.rankSNP, args.rankMUT, args.feature_size, args.label_type, args.batch_size, args.num_workers, args.out, args.d_model)

    with open(args.out + '.logfile', 'a') as log:
        log.write('Data building: '+time.strftime("%Y-%m-%d %X",time.localtime())+'\n')


    if args.model_type == 'MLP':
        modelstructure = MLP_train_valid(train_dataloader, valid_dataloader, featurecount, args.nepoch, args.label_type, device, args.lr, args.out)
    else:
        modelstructure = Attention_train_valid(train_dataloader, valid_dataloader, featurecount // args.d_model, args.d_model, args.nepoch, args.label_type, device, args.lr, args.out)

    ######################################

    end_time = time.perf_counter()
    with open(args.out + '.logfile', 'a') as log:
        log.write('End time: '+time.strftime("%Y-%m-%d %X",time.localtime())+'\n')
        log.write('Lasting: '+timer(start_time, end_time)+'\n\n')
        log.write(f'Parameters: \n\tbatch_size:\t{args.batch_size} \n\tnum_workers:\t{args.num_workers} \n\tlearning_rate:\t{args.lr} \n\tnum_epoch:\t{args.nepoch} \n\toutputprefix:\t{args.out} \n\tcuda device:\t{args.device_id} \n\tfeature size:\t{args.feature_size} \n\tlabel type:\t{args.label_type} \n\tmodel type:\t{args.model_type} \n\n')
        log.write('Model structure:\n')
        log.write(str(modelstructure))


if __name__ == '__main__':
	main()


# class argstmp():
#     def __init__(self):
#         self.inputX = "EXP_MUT_SNP.scale18K.X.txt.gz"
#         self.inputY = "EXP_MUT_SNP.scale18K.Y.txt"
#         self.rankEXP = "cor/EXP_logIC50.cor_rank.txt"
#         self.rankSNP = "cor/SNP_logIC50.cor_rank.txt"
#         self.rankMUT = "cor/MUT_logIC50.cor_rank.txt"
#         self.batch_size = 128
#         self.num_workers = 16
#         self.lr = 5e-5
#         self.nepoch = 500
#         self.device_id = '0'
#         self.feature_size = '18K'
#         self.label_type = 'discrete'
#         self.model_type = 'Attention'
#         self.d_model = 128
#         self.out = 'test'

# args = argstmp()
