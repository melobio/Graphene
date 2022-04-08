

import argparse
import os

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from dgl.data import register_data_args
from pytorch_lightning.metrics.classification import Accuracy
from pytorch_lightning.metrics.functional import auroc
from reactome_model import *
from reactome_process import get_network
from sklearn import metrics

metric = Accuracy(num_classes=2)


def cal_metric(logits, train_labels, test_labels, labels):
    predicts = logits.transpose(1, 0)
    labels = labels.transpose(1, 0)
    train_labels = train_labels.transpose(1, 0)
    test_labels = test_labels.transpose(1, 0)

    # 计算average precision
    auprcs = []
    for i in range(predicts.shape[0]):
        train_label = train_labels[i].cpu().detach().numpy()
        test_label = test_labels[i].cpu().detach().numpy()
        predict_logits = predicts[i].cpu().detach().numpy()
        predict_ind1 = set(np.where(train_label == 0)[0])
        predict_ind2 = set(np.where(test_label == 1)[0])
        predict_ind = list(predict_ind1.union(predict_ind2))
        predict_ind = np.array(predict_ind)
        truth = test_label[predict_ind]
        pred = predict_logits[predict_ind]
        auprc = metrics.average_precision_score(truth, pred)
        auprcs.append(auprc)
    auprc = np.nanmean(np.array([acc.item() for acc in auprcs]))

    # 计算auroc
    aurocs = []
    for i in range(labels.shape[0]):
        try:
            aurocs.append(auroc(predicts[i], labels[i]).cpu())
        except:
            aurocs.append(torch.tensor(0.))
    roc = torch.stack([auc for auc in aurocs]).sum() / labels.shape[0]
    return auprc, roc


def evaluate(model, features, train_labels, test_labels):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        predicts = logits.transpose(1, 0)
        train_labels = train_labels.transpose(1, 0)
        test_labels = test_labels.transpose(1, 0)
        auprcs = []
        auprc_path = [[], [], []]
        for i in range(predicts.shape[0]):
            train_label = train_labels[i].cpu().detach().numpy()
            test_label = test_labels[i].cpu().detach().numpy()
            predict_logits = predicts[i].cpu().detach().numpy()
            predict_ind1 = set(np.where(train_label == 0)[0])
            predict_ind2 = set(np.where(test_label == 1)[0])
            predict_ind = list(predict_ind1.union(predict_ind2))
            predict_ind = np.array(predict_ind)
            truth = test_label[predict_ind]
            pred = predict_logits[predict_ind]
            auprc = metrics.average_precision_score(truth, pred)
            auprcs.append(auprc)
            if 3 <= truth.sum() <= 10:
                auprc_path[0].append(auprc)
            elif 11 <= truth.sum() <= 30:
                auprc_path[1].append(auprc)
            elif 31 <= truth.sum() <= 1000:
                auprc_path[2].append(auprc)

        auprc = np.nanmean(np.array([acc.item() for acc in auprcs]))
        a = np.nanmean(np.array([acc.item() for acc in auprc_path[0]]))
        a1 = np.std(np.array(auprc_path[0])) / np.sqrt(len(auprc_path[0]))  # calculating standard error
        b = np.nanmean(np.array([acc.item() for acc in auprc_path[1]]))
        b1 = np.std(np.array(auprc_path[1])) / np.sqrt(len(auprc_path[1]))  # calculating standard error
        c = np.nanmean(np.array([acc.item() for acc in auprc_path[2]]))
        c1 = np.std(np.array(auprc_path[2])) / np.sqrt(len(auprc_path[2]))  # calculating standard error

        aurocs = []
        for i in range(test_labels.shape[0]):
            try:
                aurocs.append(auroc(predicts[i], test_labels[i]).cpu())
            except:
                aurocs.append(torch.tensor(0.))
        roc = torch.stack([auc for auc in aurocs]).sum() / test_labels.shape[0]
        return auprc, roc, a, a1, b, b1 ,c ,c1


def main(args):
    save_path = "checkpoint"

    # load and preprocess dataset
    g = get_network()

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        g = g.int().to(args.gpu)
    features = g.ndata['feat']
    labels = np.load(os.path.join(save_path, 'label.npy'))
    train_mask = np.load(os.path.join(save_path, 'train_mask.npy'))
    test_mask = np.load(os.path.join(save_path, 'test_mask.npy'))
    labels = torch.from_numpy(labels)
    train_mask = torch.from_numpy(train_mask)
    test_mask = torch.from_numpy(test_mask)

    num_feats = features.shape[1]
    train_labels = train_mask * labels
    test_labels = test_mask * labels
    in_feats = num_feats
    n_classes = 2408
    n_edges = g.edges()[0].shape[0]
    print("""----Data statistics------'
          #Edges %d
          #Classes %d
          #Train samples %d
          #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.int().sum().item(),
           test_mask.int().sum().item()))

    # add self loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()
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
    # model = FC(100, 1, features)
    #checkpoint = torch.load(os.path.join(save_path, "checkpoint.ckpt"))
    checkpoint = torch.load(os.path.join(save_path, "checkpoint.ckpt"),map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    # model = FC(100, 1, features)
    print(model)

    if cuda:
        model.cuda()

    #input_ids = torch.linspace(0, 18546, 18547).long().to(args.gpu)
    input_ids = torch.linspace(0, 18546, 18547).long() #.to(args.gpu)   cpu
    val_acc, val_roc,  a, a1, b, b1,c ,c1 = evaluate(model, input_ids, train_labels, test_labels)

    # print("ValAcc {:.4f} | ValROC {:.4f} | Pathway {:.4f}/{:.4f}/{:.4f}"
    #       .format(val_acc, val_roc, a, b, c))
    print("ValAcc {:.4f} | ValROC {:.4f} | Pathway {:.4f}_{:.4f}/{:.4f}_{:.4f}/{:.4f}_{:.4f}"
          .format(val_acc, val_roc, a, a1, b, b1, c, c1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT')
    register_data_args(parser)
    # parser.add_argument("--gpu", type=int, default=3,
    #                     help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=800000,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=128,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=0,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=0.5,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.002,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=0,
                        help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help="indicates whether to use early stop or not")
    parser.add_argument("--save_path", type=str, default="./data",
                        help="number of hidden units")
    parser.add_argument('--fastmode', action="store_true", default=False,
                        help="skip re-evaluate the validation set")
    args = parser.parse_args()
    print(args)

    main(args)
