# environment
Run `pip install -r requirements.txt` to setup environment. 
If you use GPU, please choose the GPU version of `pytorch` and `dgl`.

# training
Run `python disease.py` ：
```

```
After training, model will be stored in `./checkpoint/` folder.

# testing
Run `python evluate.py`，the testing result will reach about `Roc 0.8584` for 3000 epochs and `Roc 0.8910` for 7000 epochs.

# result
uncomment the following to save results for disease prioritization according to the checkpoint you choose.
`#np.save("./result/logits203.npy", logits.t().detach().numpy())` for `eye_epoch07000_valacc0.0000_val_roc0.8910_checkpoint.pt`
`#np.save("./result/logits202.npy", logits.t().detach().numpy())` for `checkpoint.pt`

# noting
The checkpoint "eye_epoch07000_valacc0.0000_val_roc0.8910_checkpoint.pt" was trained using 203 diseases with label file
"gwas_cui_MAPPED_TRAIT_threshold_30_tab_2.txt", where "retinitis pigmentosa" was our recently included. 
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