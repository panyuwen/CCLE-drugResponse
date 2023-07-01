# CCLE-drugResponse (developing)
drug response prediction based on the CCLE data


## ElasticNet regression / NaiveBayes classification 

```python
# usage:
# e.g., 
python elasticnet_bayes.py --featuresize 15K --datatype EXP_SNP --compound separate --modeltype elasticnet --out elasticnet.15K.EXP_SNP.separate
```

## Deep Neural Network   

MLP implemented,      
cell type, tumor type, and compound were one-hot encoded in the input data    

```python
# usgae:    
python mlp.py    
```


By: Yuwen Pan, 2023  
Contact: [panyuwen.x@gmail.com](mailto:panyuwen.x@gmail.com)


## Change log
### v0.0.3 beta

merge elasticnet & bayes; options for featuresize; define a dataset for benchmark analysis    


### v0.0.2

fix some bugs; more features included


### v0.0.1

the initial elasticNet & bayes model, following the instruction of the early CCLE paper (The Cancer Cell Line Encyclopedia enables predictive modelling of anticancer drug sensitivity. Nature. 2012)   





