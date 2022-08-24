# 配置环境
运行 `pip install -r requirements.txt` 完成环境的安装，需要安装相同版本的依赖包。  
`pytorch` 和 `dgl` 需要选择gpu版本

# 训练模型
运行如下代码，可以开始训练：
```
python disease.py \
    --gpu=1 \
    --epochs=20000 \
    --num-heads=8 \
    --num-out-heads=1 \
    --num-layers=2 \
    --num-hidden=128 \
    --attn-drop=0.4 \
    --lr=5e-3
```
训练结束后，模型文件存于`./data/pretrain/checkpoint.pt`中，预测的logist矩阵和ground truth矩阵分别存于
`./data/pretrain/predicts.npy` 和 `./data/pretrain/labels.npy` 中。

# 评估模型
运行 `python evluate.py`，可以加载训练好的模型(`./data/checkpoint.pt`)，进行evaluation。  
结果如下：
`TEST -- Roc 0.7632`