# SEMI-CON project

***

## DATASET
### 1. LER

(1) Train : "2020_LER_20201008_V008.xlsx"
```
- input(3) : amp, corrx, corry, (alpha)
- num_in_cycle : 50
- num_of_cycle : 127
```

(2) Test : 2020_LER_20201102_testset_V04.xlsx
```
- input : amp, corrx, corry, (alpha)
- num_in_cycle : 250
- num_of_cycle : 10
```


###2. RDFWFV
(1) train : '2020_RDFWFV_20201222_V10.xlsx' (RDW&WFV, WFV, RDF)
```
- input(4+2) : AGS, SDdoping, RCDdoping, RWpeak + onehot
- num_in_cycle : 50
- num_of_cycle : 100(RDW&WFV) & 50(WFV) & 50(RDF)
```

(2) test : 2020_RDFWFV_20210107.xlsx (RDF&WFV, WFV, RDF)
```
- input(4+2) :  SDdoping, RCDdoping, RWpeak, AGS + onehot
- num_in_cycle : 250
- num_of_cycle : 6(RDW&WFV) & 5(WFV) & 5(RDF)
```

***

## MODEL SCRIPT
### 1. LER

```
python3 main-naive_gan.py --date test --dataset 2020_LER_20201008_V008.xlsx --dataset_test 2020_LER_20201102_testset_V04.xlsx --trainer gan --mean_model_type mlp --gan_model_type gan1 --batch_size 32 --g_lr 0.001 --d_lr 0.001 --noise_d 100 --gan_hidden_dim 50 --sample_num 250 --mode train --layer 1 --num_of_input 3
```


### 2. RDFWFV
```
python3 main-naive_gan.py --date test --dataset rdfwfv_wfv_rdf_train2020_RDFWFV_20201222_V10.xlsx --dataset_test 2021_RDFWFV_20210107.xlsx --trainer gan --mean_model_type mlp --gan_model_type gan1 --batch_size 32 --g_lr 0.001 --d_lr 0.001 --noise_d 100 --gan_hidden_dim 50 --sample_num 250 --mode train --layer 1 --num_of_input 4 --one_hot 2
```


