# usage: python bayes.py datatypelist compound
# python bayes.py SNP,MUT 17-AAG

import sys
import os
import pandas as pd
import numpy as np
from scipy import stats
import pickle
from sklearn import preprocessing
# from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from statsmodels.stats import multitest


def dif_con(list1, list2):
	df = pd.DataFrame({'d':list1, 'p':list2})
	df.dropna(inplace=True)

	d1 = df[df['p']==1].copy()
	d0 = df[df['p']==0].copy()

	if d1.empty or d0.empty:
		return np.nan
	else:
		return stats.ranksums(list(d1['d']), list(d0['d'])).pvalue


def dif_dis(list1, list2):
	df = pd.DataFrame({'d':list1, 'p':list2})
	df.dropna(inplace=True)

	a = df[(df['d']==1) & (df['p']==1)].shape[0]
	b = df[(df['d']==1) & (df['p']==0)].shape[0]
	c = df[(df['d']==0) & (df['p']==1)].shape[0]
	d = df[(df['d']==0) & (df['p']==0)].shape[0]

	return stats.fisher_exact([[a,b], [c,d]]).pvalue


def main_diff(data, phen, datakind, p_cut=0.25, ntop=50):
	dif = pd.DataFrame({'gene':list(data.columns)})
	if datakind == 'con':
		dif['dif'] = dif['gene'].apply(lambda x: dif_con(list(data[x]), list(phen['phenotype'])))
	else:
		dif['dif'] = dif['gene'].apply(lambda x: dif_dis(list(data[x]), list(phen['phenotype'])))

	# dif['fdr'] = multitest.multipletests(list(dif['dif']), method='fdr_bh')
	dif.sort_values('dif', ascending=True, inplace=True)
	dif = dif.head(n=ntop)

	return list(dif['gene'])


def bayes(data, phen, cv_fold=10, cv_repeat=5):
	X = data.values
	y = phen['phenotype'].values
	
	# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, train_size=0.95, random_state=1)


	rkf = RepeatedKFold(n_splits=cv_fold, n_repeats=cv_repeat, random_state=1)

	best_model = None
	best_auc, f1, precision, recall, accuracy = 0, 0, 0, 0, 0

	for train_index, test_index in rkf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]

		if len(set(y_train)) == 1 or len(set(y_test)) == 1:
			continue

		model = GaussianNB()
		model.fit(X_train, y_train)
		
		y_pred_prob = model.predict_proba(X_test)
		y_pred_clas = np.argmax(y_pred_prob, axis=1)

		auc = roc_auc_score(y_test, y_pred_prob[:,1])

		if auc > best_auc:
			best_auc = auc
			best_model = model
		
			TP = sum((y_pred_clas == 1) & (y_test == 1))
			FP = sum((y_pred_clas == 1) & (y_test == 0))
			FN = sum((y_pred_clas == 0) & (y_test == 1))
			TN = sum((y_pred_clas == 0) & (y_test == 0))

			accuracy = (TP + TN) * 1.0 / (TP + TN + FP + FN)

			precision = TP * 1.0 / (TP + FP)
			recall = TP * 1.0 / (TP + FN)
			f1 = 2.0 * TP / (2 * TP + FP + FN)

	return best_model, best_auc, f1, precision, recall, accuracy


def main(output, datatypelist, samplelist, compound, tumortype='all'):
	# EXP, MUT, SNP
	# IC50 > 1
	if tumortype == 'all':
		phen = pheno[(pheno['Compound']==compound) & (pheno['Primary Cell Line Name'].isin(samplelist))].copy()
	else:
		phen = pheno[(pheno['Compound']==compound) & (pheno['CCLE tumor type']==tumortype) & (pheno['Primary Cell Line Name'].isin(samplelist))].copy()
	phen.drop_duplicates(['Primary Cell Line Name'], inplace=True)


	phen = phen[['phenotype']]
	phen['phenotype'] = phen['phenotype'].replace({'sensitive':1, 'refractory':0})


	data = []
	for datatype in datatypelist:
		tmp = pd.read_csv('../00.data/' + datatype + '_data.T.txt.gz', sep='\t', index_col=['cell'])
		tmp = tmp.loc[samplelist]
		tmp.rename(columns=lambda x: x+'_'+datatype, inplace=True)
		tmp = tmp.loc[list(phen.index)]

		# select top 50 genes for each datatype for simplicity
		if datatype in ['EXP','SNP']:
			genelist = main_diff(tmp, phen, 'con')
		else:
			genelist = main_diff(tmp, phen, 'dis')

		tmp = tmp[genelist]
		data += [tmp]
	data = pd.concat(data, axis=1)


	# model fitting
	model, auc, f1, precision, recall, accuracy = bayes(data, phen)


	with open(output+'_genelist.txt', 'w') as f:
		for g in genelist:
			f.write(g + '\n')
	
	with open(output+'_model.pickle', 'wb') as f:
		pickle.dump(model, f)

	with open(output+'_metrics.txt', 'w') as f:
		f.write('AUC:\t' + str(auc) + '\n')
		f.write('F1:\t' + str(f1) + '\n')
		f.write('precision:\t' + str(precision) + '\n')
		f.write('recall:\t' + str(recall) + '\n')
		f.write('accuracy:\t' + str(accuracy) + '\n')


	# #load model
	# with open('rfc.pickle','rb') as f:
	# 	model = pickle.load(f)


############################################################################################
## infos
samplelist = list(pd.read_csv('../00.data/cell_line.info.overlap.list', header=None)[0])
info = pd.read_csv('../00.data/cell_line.info.overlap.txt', sep='\t')
info.index = info['Cell line primary name']

# 2015-Pharmacogenomic agreement between two cancer cell line data sets
# Without the waterfall method, we used a fixed threshold of 1 μ M for each drug,
# in order to distinguish between sensitive and resistant cell-lines. This was much
# simpler and faster than the previous approach, while generating similar results
pheno = pd.read_csv('../00.data/drug_sensitivity.txt', sep='\t')
pheno.index = pheno['Primary Cell Line Name']
pheno['phenotype'] = pheno['IC50 (µM)'].apply(lambda x: 'sensitive' if x < 1 else 'refractory')


datatypelist, compound = sys.argv[1:4]

datatypelist = datatypelist.split('_')
output = '_'.join(datatypelist) + '_' + compound

main(output, datatypelist, samplelist, compound)

