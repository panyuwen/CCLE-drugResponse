# usage: python elasticnet.py datatypelist phenotype compound
# python elasticnet.py SNP,MUT ActArea 17-AAG

import sys
import os
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


############################################################################################
def cor_one(list1, list2):
	df = pd.DataFrame({'l1':list1, 'l2':list2})
	df.dropna(inplace=True)

	if df.empty:
		return np.nan
	else:
		return abs(stats.pearsonr(list(df['l1']), list(df['l2'])).statistic)


def main_cor(data, phen, cor_cut=0.1, ntop=500):
	# corr
	cor = pd.DataFrame({'gene':list(data.columns)})
	cor['cor'] = cor['gene'].apply(lambda x: cor_one(list(data[x]), list(phen['phenotype'])))

	cor = cor[cor['cor']>cor_cut]

	return list(cor['gene'])


def elasticnet(data, phen, cv_fold=10, cv_repeat=10, ratios=np.arange(0, 1, 0.05), alphas=np.exp(np.arange(-6, 6, 0.05))):
	# https://machinelearningmastery.com/elastic-net-regression-in-python/
	
	X = data.values
	y = phen['phenotype'].values
	
	# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, train_size=0.95, random_state=1)

	# define model evaluation method
	cv = RepeatedKFold(n_splits=cv_fold, n_repeats=cv_repeat, random_state=1)
	
	# define model
	# The parameter l1_ratio corresponds to alpha in the glmnet R package, scaling between l1 and l2 penalties
	# while alpha corresponds to the lambda parameter in glmne
	# n_jobs=-1 to use all the processors
	model = ElasticNetCV(l1_ratio=ratios, alphas=alphas, cv=cv, n_jobs=-1)

	# fit model
	model.fit(X, y)

	y_pred = model.predict(X)
	R2 = explained_variance_score(y, y_pred)
	MAE = mean_absolute_error(y, y_pred)
	RMSE = mean_squared_error(y, y_pred, squared=False)

	# yhat = model.predict(X_test)
	
	# summarize chosen configuration
	# print('alpha: %f' % model.alpha_)
	# print('l1_ratio_: %f' % model.l1_ratio_)

	return model, R2, MAE, RMSE


def main(output, datatypelist, phenotype, samplelist, compound, tumortype='all'):
	# EXP, MUT, SNP
	# Amax, ActArea, logIC50
	data = []
	for datatype in datatypelist:
		tmp = pd.read_csv('../00.data/' + datatype + '_data.T.txt.gz', sep='\t', index_col=['cell'])
		tmp = tmp.loc[samplelist]
		tmp.rename(columns=lambda x: x+'_'+datatype, inplace=True)
		data += [tmp]
	data = pd.concat(data, axis=1)


	if tumortype == 'all':
		phen = pheno[(pheno['Compound']==compound) & (pheno['Primary Cell Line Name'].isin(samplelist))].copy()
	else:
		phen = pheno[(pheno['Compound']==compound) & (pheno['CCLE tumor type']==tumortype) & (pheno['Primary Cell Line Name'].isin(samplelist))].copy()
	phen.drop_duplicates(['Primary Cell Line Name'], inplace=True)
	
	phen = phen[[phenotype]]
	phen.columns = ['phenotype']

	data = data.loc[list(phen.index)]
	

	# norm
	preprocessing.scale(data, axis=0, with_mean=True, with_std=True, copy=False)
	# data.mean(axis=0, skipna=True).describe()

	# cor
	genelist = main_cor(data, phen)

	data = data[genelist]


	# model fitting
	model, R2, MAE, RMSE = elasticnet(data, phen)


	with open(output+'_genelist.txt', 'w') as f:
		for g in genelist:
			f.write(g + '\n')
	
	with open(output+'_model.pickle', 'wb') as f:
		pickle.dump(model, f)

	with open(output+'_metrics.txt', 'w') as f:
		f.write('R2:\t' + str(R2) + '\n')
		f.write('MAE:\t' + str(MAE) + '\n')
		f.write('RMSE:\t' + str(RMSE) + '\n')



	# #load model
	# with open('saved_model/rfc.pickle','rb') as f:
	# 	model = pickle.load(f)



############################################################################################
## infos
samplelist = list(pd.read_csv('../00.data/cell_line.info.overlap.list', header=None)[0])
info = pd.read_csv('../00.data/cell_line.info.overlap.txt', sep='\t')
info.index = info['Cell line primary name']

pheno = pd.read_csv('../00.data/drug_sensitivity.txt', sep='\t')
pheno.index = pheno['Primary Cell Line Name']

# compoundlist = list(pheno['Compound'].unique())


datatypelist, phenotype, compound = sys.argv[1:4]

datatypelist = datatypelist.split('_')
output = '_'.join(datatypelist) + '_' + phenotype + '_' + compound

main(output, datatypelist, phenotype, samplelist, compound)


