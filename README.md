# CCLE-drugResponse (developing)
drug response prediction based on the CCLE data


## ElasticNet regression / NaiveBayes classification 

```shell
# usage:
python elasticnet_bayes.py -h
```

```shell
# e.g., 
python elasticnet_bayes.py --featuresize 10K --datatype EXP_SNP --compound separate --modeltype elasticnet --out elasticnet.10K.EXP_SNP.separate
```

## Deep Neural Network   

```shell
# usage:
python nn.py -h
```

```shell
# e.g., 
python nn.py --device-id 0 --feature-size 10K --label-type continuous --model-type MLP --out MLP.10K.continuous
```

---
By: Yuwen Pan, 2023  
Contact: [panyuwen.x@gmail.com](mailto:panyuwen.x@gmail.com)    


## Change log
### v0.0.3 beta

merge elasticnet & bayes; options for featuresize; define a dataset for benchmark analysis    
for MLP, tumor type, and compound were one-hot encoded in the input data; continuous & discrete labels    
attention is under development

### v0.0.2

fix some bugs; more features included


### v0.0.1

the initial elasticNet & bayes model, following the instruction of the early CCLE paper (The Cancer Cell Line Encyclopedia enables predictive modelling of anticancer drug sensitivity. Nature. 2012)   





