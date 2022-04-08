

## 运行测试
run `python evaluate_reactome.py`，and load the trained checkpoint to evaluate。  
results are as the following：
```
ValAcc 0.5008 | ValROC 0.0000 | Pathway 0.4210/0.6417/0.7581
```

| pathway | score |  
|  ----  | ----  | 
| 3-10 | 0.4210 |  
| 11-30 | 0.6417  |  
| 31-1000 | 0.7581 |
| mean | 0.0.5008 |