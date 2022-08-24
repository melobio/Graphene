#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from dgl.data import register_data_args
from pytorch_lightning.metrics.classification import Accuracy
from pytorch_lightning.metrics.functional import auroc
from sklearn import metrics

from gat import GAT
from process import get_network

metric = Accuracy(num_classes=2)


def cal_metric(logits, labels):
    accs = []
    predict_labels = logits.transpose(1, 0)
    acc_labels = labels.transpose(1, 0)
    # for i in range(predict_labels.shape[0]):
    #     accs.append(metric(predict_labels[i], acc_labels[i]))
    # acc = np.array(accs).mean().item()
    for i in range(predict_labels.shape[0]):
        accs.append(metrics.average_precision_score(acc_labels[i].cpu().detach().numpy(),
                                                    predict_labels[i].cpu().detach().numpy()))
    acc = np.nanmean(np.array([acc.item() for acc in accs]))

    predicts = logits.transpose(1, 0)
    labels = labels.transpose(1, 0)
    aurocs = []
    for i in range(labels.shape[0]):
        try:
            aurocs.append(auroc(predicts[i], labels[i]).cpu())
        except:
            aurocs.append(torch.tensor(0.))
    roc = torch.stack([auc for auc in aurocs]).sum() / labels.shape[0]
    return acc, roc


def cal_intersect(predicts, labels, head=13):
    over = []
    for i in range(head):
        predict = predicts[i]
        label = labels[i]
        label_indices = torch.where(label == 1)[0].tolist()
        nums = len(label_indices)
        pre_values, pre_indices = torch.topk(predict, nums)
        lab_values, lab_indices = torch.topk(label, nums)
        intersect = list(set(pre_indices.tolist()).intersection(set(lab_indices.tolist())))
        over.append(len(intersect) / nums)
    return over


def main(args):
    # load and preprocess dataset
    g = get_network()

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        g = g.int().to(args.gpu)

    features = g.ndata['feat']
    labels = g.ndata['label']
    num_feats = features.shape[1]
    n_classes = 202

    # add self loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)

    # create model
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
    model = GAT(g,
                args.num_layers,
                num_feats,
                args.num_hidden,
                n_classes,
                heads,
                F.elu,
                args.in_drop,
                args.attn_drop,
                args.negative_slope,
                args.residual,
                features)
    model.load_state_dict(torch.load('./data/checkpoint.pt',map_location=torch.device('cpu')))
    print(model)
    if cuda:
        model.cuda()

    # evaluate
    input_ids = torch.linspace(0, 17538, 17539).long()
    model.eval()
    logits = model(input_ids)

    acc, roc = cal_metric(logits, labels)
    predicts = np.load("./data/predicts.npy")
    labels = np.load("./data/labels.npy")
    overs = cal_intersect(torch.Tensor(predicts), torch.Tensor(labels))
    print(overs)
    print("TEST -- Acc {:.4f} | Roc {:.4f} ".format(acc, roc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT')
    register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=128,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=0,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=0.5,
                        help="attention dropout")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    args = parser.parse_args()
    print(args)

    main(args)
