import sys
import os
import argparse
import time
import socket
import pandas as pd
import numpy as np
from scipy import stats
import pickle
from sklearn import preprocessing
# from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score


############################################################################################
def classification_metrics(y_true, y_pred_prob):
    y_pred_clas = np.argmax(y_pred_prob, axis=1)

    auc = roc_auc_score(y_true, y_pred_prob[:,1])

    TP = sum((y_pred_clas == 1) & (y_true == 1))
    FP = sum((y_pred_clas == 1) & (y_true == 0))
    FN = sum((y_pred_clas == 0) & (y_true == 1))
    TN = sum((y_pred_clas == 0) & (y_true == 0))

    acc = (TP + TN) * 1.0 / (TP + TN + FP + FN)

    precision = TP * 1.0 / (TP + FP)
    recall = TP * 1.0 / (TP + FN)
    f1 = 2.0 * TP / (2 * TP + FP + FN)

    return auc, f1, precision, recall, acc


def BayesTrain(X, Y, cv_fold=10, cv_repeat=10):
    if len(set(Y)) == 1:
        model = GaussianNB()
        return model, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    
    idx1 = Y == 1; idx0 = Y == 0
    if idx1.sum() <= 12 or idx0.sum() <= 12:
        model = GaussianNB()
        return model, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    X1, X0, Y1, Y0 = X[idx1], X[idx0], Y[idx1], Y[idx0]

    X0_train, X0_valid, Y0_train, Y0_valid = train_test_split(X0, Y0, test_size=0.1, train_size=0.9, random_state=1)
    X1_train, X1_valid, Y1_train, Y1_valid = train_test_split(X1, Y1, test_size=0.1, train_size=0.9, random_state=1)

    X_train = np.concatenate([X0_train, X1_train], axis=0)
    Y_train = np.concatenate([Y0_train, Y1_train])

    X_valid = np.concatenate([X0_valid, X1_valid], axis=0)
    Y_valid = np.concatenate([Y0_valid, Y1_valid])

    ## cross-validation to search the best model
    rkf = RepeatedKFold(n_splits=cv_fold, n_repeats=cv_repeat, random_state=1)

    best_model = None
    best_auc = 0

    for (train0_index, cv0_index), (train1_index, cv1_index) in zip(rkf.split(X0_train), rkf.split(X1_train)):
        train_X = np.concatenate([X0_train[train0_index], X1_train[train1_index]], axis=0)
        cv_X = np.concatenate([X0_train[cv0_index], X1_train[cv1_index]], axis=0)

        train_Y = np.concatenate([Y0_train[train0_index], Y1_train[train1_index]])
        cv_Y = np.concatenate([Y0_train[cv0_index], Y1_train[cv1_index]])

        if len(set(train_Y)) == 1:
            continue

        model = GaussianNB()
        model.fit(train_X, train_Y)

        y_pred_prob = model.predict_proba(cv_X)

        tmp_auc, tmp_f1, tmp_precision, tmp_recall, tmp_acc = classification_metrics(cv_Y, y_pred_prob)
        if tmp_auc > best_auc:
            best_auc, best_model = tmp_auc, model
    
    ## evaluate training / validation metrics using the best model
    y_pred_prob = best_model.predict_proba(X_train)
    train_auc, train_f1, train_precision, train_recall, train_acc = classification_metrics(Y_train, y_pred_prob)

    y_pred_prob = best_model.predict_proba(X_valid)
    valid_auc, valid_f1, valid_precision, valid_recall, valid_acc = classification_metrics(Y_valid, y_pred_prob)

    return best_model, train_auc, train_f1, train_precision, train_recall, train_acc,   valid_auc, valid_f1, valid_precision, valid_recall, valid_acc


def ElasticNetTrain(X, Y, cv_fold=10, cv_repeat=10, ratios=np.arange(0, 1, 0.1), alphas=np.exp(np.arange(-6, 6, 0.5))):
    # https://machinelearningmastery.com/elastic-net-regression-in-python/
    
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.1, train_size=0.9, random_state=1)

    # define model evaluation method
    cv = RepeatedKFold(n_splits=cv_fold, n_repeats=cv_repeat, random_state=1)
    
    # define model
    # The parameter l1_ratio corresponds to alpha in the glmnet R package, scaling between l1 and l2 penalties
    # while alpha corresponds to the lambda parameter in glmne
    # n_jobs=-1 to use all the processors
    model = ElasticNetCV(l1_ratio=ratios, alphas=alphas, cv=cv, n_jobs=4)

    # fit model
    model.fit(X_train, Y_train)

    Y_train_pred = model.predict(X_train)
    Y_valid_pred = model.predict(X_valid)

    train_R2 = explained_variance_score(Y_train, Y_train_pred)
    train_MAE = mean_absolute_error(Y_train, Y_train_pred)
    train_MSE = mean_squared_error(Y_train, Y_train_pred)

    valid_R2 = explained_variance_score(Y_valid, Y_valid_pred)
    valid_MAE = mean_absolute_error(Y_valid, Y_valid_pred)
    valid_MSE = mean_squared_error(Y_valid, Y_valid_pred)

    # summarize chosen configuration
    # print('alpha: %f' % model.alpha_)
    # print('l1_ratio_: %f' % model.l1_ratio_)

    return model, train_R2, train_MAE, train_MSE, valid_R2, valid_MAE, valid_MSE


def train(X, Y, modeltype, output, compindex):
    if modeltype == 'elasticnet':
        train_model, train_R2, train_MAE, train_MSE, valid_R2, valid_MAE, valid_MSE = ElasticNetTrain(X, Y)
        
        with open(output+'_metrics.'+compindex+'.txt', 'w') as fin:
            fin.write('train_R2\ttrain_MAE\ttrain_MSE\tvalid_R2\tvalid_MAE\tvalid_MSE\n')
            fin.write(f'{train_R2}\t{train_MAE}\t{train_MSE}\t{valid_R2}\t{valid_MAE}\t{valid_MSE}\n')
    else:
        Y = (Y > 0) - 0
        train_model, train_auc, train_f1, train_precision, train_recall, train_acc,   valid_auc, valid_f1, valid_precision, valid_recall, valid_acc = BayesTrain(X, Y)

        with open(output+'_metrics.'+compindex+'.txt', 'w') as fin:
            fin.write('train_auc\ttrain_f1\ttrain_precision\ttrain_recall\ttrain_acc\tvalid_auc\tvalid_f1\tvalid_precision\tvalid_recall\tvalid_acc\n')
            fin.write(f'{train_auc}\t{train_f1}\t{train_precision}\t{train_recall}\t{train_acc}\t{valid_auc}\t{valid_f1}\t{valid_precision}\t{valid_recall}\t{valid_acc}\n')

    with open(output+'_model.'+compindex+'.pickle', 'wb') as fin:
        pickle.dump(train_model, fin)

    # #load model
    # with open('saved_model/rfc.pickle','rb') as f:
    #     model = pickle.load(f)


def data_prepare(inputX, inputY, rankEXP, rankSNP, rankMUT, featuresize, datatype):
    X = pd.read_csv(inputX, sep='\t', index_col=['cell'])
    Y = pd.read_csv(inputY, header=None)

    rankexp = pd.read_csv(rankEXP, sep='\t', usecols=['Unnamed: 0','tot_rank'])
    rankmut = pd.read_csv(rankMUT, sep='\t', usecols=['Unnamed: 0','tot_rank'])
    ranksnp = pd.read_csv(rankSNP, sep='\t', usecols=['Unnamed: 0','tot_rank'])

    featureprop = {'18K':0.4, '10K':0.2, '5K':0.1, '1K':0.02}[featuresize]
    rankexp = list(rankexp.head(n=int(rankexp.shape[0]*featureprop))['Unnamed: 0'])
    rankmut = list(rankmut.head(n=int(rankmut.shape[0]*featureprop))['Unnamed: 0'])
    ranksnp = list(ranksnp.head(n=int(ranksnp.shape[0]*featureprop))['Unnamed: 0'])

    genelist = []
    datatypelist = datatype.split('_')

    for dt in datatypelist:
        if dt == 'SNP':
            genelist.extend(ranksnp)
        elif dt == 'EXP':
            genelist.extend(rankexp)
        elif dt == 'MUT':
            genelist.extend(rankmut)
    
    cols = list(X.columns)
    cols = [x for x in cols if x.split('_')[-1] not in ['SNP','EXP','MUT']]

    X = X[cols + genelist]

    celltypelist = list(set(X.index))
    X.drop(celltypelist, axis=1, inplace=True)

    return X, Y


def timer(start_time, end_time):
    t = float(end_time) - float(start_time)
    t_m,t_s = divmod(t, 60)   
    t_h,t_m = divmod(t_m, 60)
    r_t = str(int(t_h)).zfill(2) + ":" + str(int(t_m)).zfill(2) + ":" + str(int(t_s)).zfill(2)

    return r_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputX", type=str, required=False, default="/glusterfs/home/local_pan_yuwen/research/20230608_CCLE2012/03.NN/EXP_MUT_SNP.scale18K.X.txt.gz")
    parser.add_argument("--inputY", type=str, required=False, default="/glusterfs/home/local_pan_yuwen/research/20230608_CCLE2012/03.NN/EXP_MUT_SNP.scale18K.Y.txt")
    parser.add_argument("--rankEXP", type=str, required=False, default="/glusterfs/home/local_pan_yuwen/research/20230608_CCLE2012/03.NN/cor/EXP_logIC50.cor_rank.txt")
    parser.add_argument("--rankSNP", type=str, required=False, default="/glusterfs/home/local_pan_yuwen/research/20230608_CCLE2012/03.NN/cor/SNP_logIC50.cor_rank.txt")
    parser.add_argument("--rankMUT", type=str, required=False, default="/glusterfs/home/local_pan_yuwen/research/20230608_CCLE2012/03.NN/cor/MUT_logIC50.cor_rank.txt")
    parser.add_argument("--featuresize", type=str, required=True, choices=['18K','10K','5K','1K'])
    parser.add_argument("--datatype", type=str, required=True, choices=['EXP','MUT','SNP','EXP_MUT','EXP_SNP','SNP_MUT','EXP_SNP_MUT'])
    parser.add_argument("--compound", type=str, required=True, choices=['combine', 'separate'])
    parser.add_argument("--compoundinfo", type=str, required=False, default="/glusterfs/home/local_pan_yuwen/research/20230608_CCLE2012/00.data/compound.list")
    parser.add_argument("--modeltype", type=str, required=True, choices=['elasticnet', 'bayes'])
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    start_time = time.perf_counter()
    with open(args.out + '.logfile', 'w') as log:
        log.write('Hostname: '+socket.gethostname()+'\n')
        log.write('Working directory: '+os.getcwd()+'\n')
        log.write('Start time: '+time.strftime("%Y-%m-%d %X",time.localtime())+'\n')

    X, Y = data_prepare(args.inputX, args.inputY, args.rankEXP, args.rankSNP, args.rankMUT, args.featuresize, args.datatype)
    
    if args.compound == 'combine':
        with open(args.out + '.featurelist.txt', 'w') as fin:
            for i in X.columns:
                fin.write(i+'\n')
        train(X.values, Y[0].values, args.modeltype, args.out, 'compoundAll')
    else:
        compoundlist = list(pd.read_csv(args.compoundinfo, header=None)[0])
        for compound in compoundlist:
            subX = X.copy()
            subX['phenotype'] = list(Y[0])

            subX = subX[subX[compound]==1].copy()
            subY = subX[['phenotype']].copy()

            subX.drop(compoundlist + ['phenotype'], axis=1, inplace=True)
            
            with open(args.out + '.featurelist.txt', 'w') as fin:
                for i in X.columns:
                    fin.write(i+'\n')
            train(subX.values, subY['phenotype'].values, args.modeltype, args.out, compound)

    end_time = time.perf_counter()
    with open(args.out + '.logfile', 'a') as log:
        log.write('End time: '+time.strftime("%Y-%m-%d %X",time.localtime())+'\n')
        log.write('Lasting: '+timer(start_time, end_time)+'\n\n')


if __name__ == '__main__':
    main()

# inputX = "/glusterfs/home/local_pan_yuwen/research/20230608_CCLE2012/03.NN/EXP_MUT_SNP.scale18K.X.txt.gz"
# inputY = "/glusterfs/home/local_pan_yuwen/research/20230608_CCLE2012/03.NN/EXP_MUT_SNP.scale18K.Y.txt"
# rankEXP = "/glusterfs/home/local_pan_yuwen/research/20230608_CCLE2012/03.NN/cor/EXP_logIC50.cor_rank.txt"
# rankSNP = "/glusterfs/home/local_pan_yuwen/research/20230608_CCLE2012/03.NN/cor/SNP_logIC50.cor_rank.txt"
# rankMUT = "/glusterfs/home/local_pan_yuwen/research/20230608_CCLE2012/03.NN/cor/MUT_logIC50.cor_rank.txt"

# featuresize = '10K'
# datatype = 'EXP_SNP'
# compound = 'separate'
# compoundinfo = "/glusterfs/home/local_pan_yuwen/research/20230608_CCLE2012/00.data/compound.list"

# modeltype = 'elasticnet'
# out = 'test'

