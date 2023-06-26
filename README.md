# CCLE-drugResponse
drug response prediction based on the CCLE data


## ElasticNet regression

```python
# usage: 
python elasticnet.py datatypelist phenotype compound
# e.g., 
python elasticnet.py SNP,MUT ActArea 17-AAG
```

## NaiveBayes

```python
# usage: 
python bayes.py datatypelist compound
# e.g., 
python bayes.py SNP,MUT 17-AAG
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
### v0.0.1

the initial elasticNet & bayes model, following the instruction of the early CCLE paper (The Cancer Cell Line Encyclopedia enables predictive modelling of anticancer drug sensitivity. Nature. 2012)   


### v0.0.2

fix some bugs
more features included



