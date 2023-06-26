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
		return np.nan, np.nan
	else:
		res = stats.pearsonr(list(df['l1']), list(df['l2']))
		return abs(res.statistic), res.pvalue


def main_cor(data, phen, ntype, cor_cut=0.1, ntop=500):
	# corr
	cor = pd.DataFrame({'gene':list(data.columns)})

	cor['res'] = cor['gene'].apply(lambda x: cor_one(list(data[x]), list(phen['phenotype'])))
	cor['cor'] = cor['res'].apply(lambda x: x[0])
	cor['p'] = cor['res'].apply(lambda x: x[1])

	cor.sort_values(['p','cor'], ascending=[True, False], inplace=True)
	cor = cor.head(n=int(cor.shape[0] * 0.2 / ntype))

	return list(cor['gene'])


def elasticnet(data, phen, cv_fold=10, cv_repeat=5, ratios=np.arange(0, 1, 0.1), alphas=np.exp(np.arange(-6, 6, 0.5))):
	# https://machinelearningmastery.com/elastic-net-regression-in-python/
	
	X = data.values
	y = phen['phenotype'].values
	
	X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.05, train_size=0.95, random_state=1)

	# define model evaluation method
	cv = RepeatedKFold(n_splits=cv_fold, n_repeats=cv_repeat, random_state=1)
	
	# define model
	# The parameter l1_ratio corresponds to alpha in the glmnet R package, scaling between l1 and l2 penalties
	# while alpha corresponds to the lambda parameter in glmne
	# n_jobs=-1 to use all the processors
	model = ElasticNetCV(l1_ratio=ratios, alphas=alphas, cv=cv, n_jobs=4)

	# fit model
	model.fit(X_train, y_train)

	y_train_pred = model.predict(X_train)
	y_valid_pred = model.predict(X_valid)

	train_R2 = explained_variance_score(y_train, y_train_pred)
	train_MAE = mean_absolute_error(y_train, y_train_pred)
	train_MSE = mean_squared_error(y_train, y_train_pred)

	valid_R2 = explained_variance_score(y_valid, y_valid_pred)
	valid_MAE = mean_absolute_error(y_valid, y_valid_pred)
	valid_MSE = mean_squared_error(y_valid, y_valid_pred)

	# summarize chosen configuration
	# print('alpha: %f' % model.alpha_)
	# print('l1_ratio_: %f' % model.l1_ratio_)

	return model, train_R2, train_MAE, train_MSE, valid_R2, valid_MAE, valid_MSE


def main(output, datatypelist, phenotype, samplelist, compound, tumortype='all'):
	if tumortype == 'all':
		phen = pheno[(pheno['Compound']==compound) & (pheno['Primary Cell Line Name'].isin(samplelist))].copy()
	else:
		phen = pheno[(pheno['Compound']==compound) & (pheno['CCLE tumor type']==tumortype) & (pheno['Primary Cell Line Name'].isin(samplelist))].copy()
	phen.drop_duplicates(['Primary Cell Line Name'], inplace=True)
	
	phen = phen[[phenotype]]
	phen.columns = ['phenotype']

	# EXP, MUT, SNP
	# Amax, ActArea, logIC50
	data = []
	for datatype in datatypelist:
		tmp = pd.read_csv('../../00.data/' + datatype + '_data.T.txt.gz', sep='\t', index_col=['cell'])
		tmp = tmp.loc[samplelist]
		tmp.rename(columns=lambda x: x+'_'+datatype, inplace=True)
		tmp = tmp.loc[list(phen.index)]

		# norm
		preprocessing.scale(tmp, axis=0, with_mean=True, with_std=True, copy=False)

		# cor
		genelist = main_cor(tmp, phen, len(datatypelist))
		tmp = tmp[genelist]

		data += [tmp]
	data = pd.concat(data, axis=1)

	# data.mean(axis=0, skipna=True).describe()
	tmp = pd.concat([phen, data], axis=1)
	tmp.insert(0, 'cell', list(tmp.index))
	tmp.to_csv(output+'_data.txt.gz', sep='\t', index=None, compression='gzip')


	# model fitting
	model, train_R2, train_MAE, train_MSE, valid_R2, valid_MAE, valid_MSE = elasticnet(data, phen)


	with open(output+'_featurelist.txt', 'w') as f:
		for g in list(data.columns):
			f.write(g + '\n')
	
	with open(output+'_model.pickle', 'wb') as f:
		pickle.dump(model, f)

	with open(output+'_metrics.txt', 'w') as f:
		f.write('datatype\tphenotype\tcompound\ttrain_R2\ttrain_MAE\ttrain_MSE\tvalid_R2\tvalid_MAE\tvalid_MSE\n')
		f.write('_'.join(datatypelist) + f'\t{phenotype}\t{compound}\t{train_R2}\t{train_MAE}\t{train_MSE}\t{valid_R2}\t{valid_MAE}\t{valid_MSE}\n')

	# #load model
	# with open('saved_model/rfc.pickle','rb') as f:
	# 	model = pickle.load(f)



############################################################################################
## infos
samplelist = list(pd.read_csv('../../00.data/cell_line.info.overlap.list', header=None)[0])

pheno = pd.read_csv('../../00.data/drug_sensitivity.txt', sep='\t')
pheno.index = pheno['Primary Cell Line Name']
pheno = pheno[pheno['Primary Cell Line Name'].isin(samplelist)]

info = pd.read_csv('../../00.data/cell_line.info.overlap.txt', sep='\t')
info.index = info['Cell line primary name']
info = dict(zip(list(info.index), list(info['CCLE tumor type'])))

pheno['CCLE tumor type'] = pheno['Primary Cell Line Name'].apply(lambda x: info[x])


# compoundlist = list(pheno['Compound'].unique())


datatypelist, phenotype, compound = sys.argv[1:4]

datatypelist = datatypelist.split('_')
output = '_'.join(datatypelist) + '_' + phenotype + '_' + compound

main(output, datatypelist, phenotype, samplelist, compound)


