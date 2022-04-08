

import argparse
import time

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from dgl.data import register_data_args
from gat import GAT
from pytorch_lightning.metrics.classification import Accuracy
from pytorch_lightning.metrics.functional import auroc
from sklearn import metrics
import numpy as  np
from process import get_network

metric = Accuracy(num_classes=2)


def cal_metric(logits, labels):
    predicts = logits.transpose(1, 0)
    labels = labels.transpose(1, 0)
    aurocs = []
    for i in range(labels.shape[0]):
        try:
            aurocs.append(auroc(predicts[i], labels[i]).cpu())
        except:
            aurocs.append(torch.tensor(0.))
    roc = torch.stack([auc for auc in aurocs]).sum() / labels.shape[0]
    return 0, roc


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
    print(over)


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        return cal_metric(logits, labels)


def main(args):
    # load and preprocess dataset
    g = get_network()

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        g = g.int().to(args.gpu)
    gen = False
    features = g.ndata['feat']
    if gen:
        emb = torch.nn.Embedding(features.shape[0], 100)
        new_feature = emb.weight.data.to(args.gpu)
        features = torch.cat((new_feature, features), dim=1)
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    num_feats = features.shape[1]
    in_feats = num_feats
    n_classes = 203
    n_edges = g.edges()[0].shape[0]
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
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
    print(model)
    if cuda:
        model.cuda()
    # loss_fcn = torch.nn.BCEWithLogitsLoss()
    loss_fcn = torch.nn.BCELoss(reduction='mean')

    # use optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    input_ids = torch.linspace(0, 18546, 18547).long().to(args.gpu)
    for epoch in range(args.epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(input_ids)
        # for error editing
        # print(train_mask)
        # print(train_mask.shape)
        #print(labels.shape)
        #
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        if epoch % 20 == 0:
            train_acc, train_roc = cal_metric(logits[train_mask], labels[train_mask])
            if args.fastmode:
                val_acc, val_roc = cal_metric(logits[val_mask], labels[val_mask])
            else:
                val_acc, val_roc = evaluate(model, input_ids, labels, val_mask)
            val_loss = loss_fcn(logits[val_mask], labels[val_mask])
            print("Epoch {:05d} | Loss {:.4f} | TrainAcc {:.4f} |"
                  " TrainRoc {:.4f} | ValLoss {:.4f} | ValAcc {:.4f} | ValROC {:.4f}".
                  format(epoch, loss.item(), train_acc, train_roc, val_loss, val_acc, val_roc))
        if epoch % 1000 == 0:
        #if epoch % 40 == 0:
            # 保存模型
            torch.save(model.state_dict(), './checkpoint/eye_epoch{:05d}_valacc{:.4f}_val_roc{:.4f}_checkpoint.pt'.format(epoch,val_acc,val_roc))
            # 保存矩阵
            predicts_nmupy = logits.transpose(1, 0).cpu().detach().numpy()
            labels_nmupy = labels.transpose(1, 0).cpu().detach().numpy()
            np.save("./data/eye_checkpoint/results/predicts.npy", predicts_nmupy)
            np.save("./data/eye_checkpoint/results/labels.npy", labels_nmupy)


    if args.early_stop:
        model.load_state_dict(torch.load('./checkpoint/eye_epoch07000_valacc0.0000_val_roc0.8910_checkpoint.pt'))
    acc, roc = evaluate(model, features, labels, test_mask)
    print("Test Accuracy {:.4f}".format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT')
    register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    # parser.add_argument("--gpu", type=int, default=-1,
    #                     help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=7100,
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
    parser.add_argument("--attn-drop", type=float, default=0.3,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=0,
                        help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--fastmode', action="store_true", default=False,
                        help="skip re-evaluate the validation set")
    args = parser.parse_args()
    print(args)

    main(args)
