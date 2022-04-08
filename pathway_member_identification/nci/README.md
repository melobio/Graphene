

## testing proformance
run `python evaluate_nci.py`，and load the trained checkpoint to evaluate。  
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