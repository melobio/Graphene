# Graphene
Pre-trained graph neural network and downstream tasks

## Download files
Download the following files respectively 

`gtex_js.csv`: https://doi.org/10.6084/m9.figshare.19550818.v1  to `./fig3-b/` folder;

`network.npy`: https://doi.org/10.6084/m9.figshare.19550902.v1  to `./network_data/` folder；

`label.npy`: https://doi.org/10.6084/m9.figshare.19551259.v1 to `./pathway_member_identification/reactom/checkpoint/` folder；

`test_mask.npy`: https://doi.org/10.6084/m9.figshare.19551439.v1 to `./pathway_member_identification/reactom/checkpoint/` folder;

`train_mask.npy`: https://doi.org/10.6084/m9.figshare.19551484.v1 to `./pathway_member_identification/reactom/checkpoint/` folder;

`disease_gen.npy`: https://doi.org/10.6084/m9.figshare.19551697.v1 to `./RR_predict/comorbidity/` folder;

`gen_feat.npy`: https://doi.org/10.6084/m9.figshare.19551790.v1 to `./RR_predict/comorbidity/` folder.

After downloading，the following downstream tasks can be run。

## Pathway member identification
### Requirements
- torch==1.5.1
- dgl==0.6.1
- pytorch-lightning==0.9.0
- scipy==1.5.1
### NCI
run `python evaluate_nci.py`，and load the trained checkpoint to evaluate。 
run  `python nci.py` to start training。
results are as the following：
```
ValAcc 0.3080 | ValROC 0.0000 | Pathway 0.3393/0.3107/0.3070
```

| pathway | score |  
|  ----  | ----  | 
| 3-10 | 0.3393 |  
| 11-30 | 0.3107  |  
| 31-1000 | 0.3070 |
| mean | 0.3080 |

### Reactome
run `python evaluate_reactome.py`，and load the trained checkpoint to evaluate。  
run  `python reactome.py` to start training。
results are as the following：
```
ValAcc 0.5008 | ValROC 0.0000 | Pathway 0.4210/0.6417/0.7581
```

| pathway | score |  
|  ----  | ----  | 
| 3-10 | 0.4210 |  
| 11-30 | 0.6417  |  
| 31-1000 | 0.7581 |
| mean | 0.5008 |


## Disease gene prioritization
Run `python disease.py` for training，After training, model will be stored in `./checkpoint/` folder.
Run `python evluate.py`for testing，the testing result will reach about `Roc 0.8584` for 3000 epochs and `Roc 0.8910` for 7000 epochs.

### result
uncomment the following to save results for disease prioritization according to the checkpoint you choose.
`#np.save("./result/logits203.npy", logits.t().detach().numpy())` for `eye_epoch07000_valacc0.0000_val_roc0.8910_checkpoint.pt`
`#np.save("./result/logits202.npy", logits.t().detach().numpy())` for `checkpoint.pt`

### noting
The checkpoint "eye_epoch07000_valacc0.0000_val_roc0.8910_checkpoint.pt" was trained using 203 diseases with label file
"gwas_cui_MAPPED_TRAIT_threshold_30_tab_2.txt", where "retinitis pigmentosa" was our recently included disease. 
To see the 202 diseases result, please load checkpoint "checkpoint.pt", and do the following changes:

in `process.py`:
(1) line 14:`disease_set_path = "gwas_cui_MAPPED_TRAIT_threshold_30_tab_2.txt"` change to
`disease_set_path = "gwas_cui_MAPPED_TRAIT_threshold_30_tab.txt"`
(2) line 33: `label = [0] * 203`  change to `label = [0] * 202`
(3) line 111: `node_label_list.append([0] * 203)` change to `node_label_list.append([0] * 202)`

in `evluate.py`:
line 59 `n_classes = 203` change to `n_classes = 202`

in `disease.py`:
ling 78 `n_classes = 203` change to `n_classes = 202`

## Comorbidity RR prediction
### Requirements
- tensorflow==1.14.0
- torch==1.5.1
- networkx==2.5
- scipy==1.5.1

Run `python comorbidity_train.py` to start training ：set `FLAGS.encoder` to gat or gcn to select encoder type.
Run `python comorbidity_predict.py` for testing.

## Pre-training

This pre-training code is based on the paper:

Weihua Hu*, Bowen Liu*, Joseph Gomes, Marinka Zitnik, Percy Liang, Vijay Pande, Jure Leskovec. Strategies for Pre-training Graph Neural Networks. ICLR 2020. 

### Requirements
- pytorch==1.0.1
- torch-cluster==1.2.4
- torch-geometric==1.0.3
- torch-scatter==1.1.2
- torch-sparse==0.2.4
- torch-spline-conv==1.0.6

### Context prediction
Run `pretrain_context_predict.py` to start context prediction pretraining task.
