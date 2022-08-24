

import argparse
import json
import os
import time
import uuid

import dgl
import numpy as np
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
        b = np.nanmean(np.array([acc.item() for acc in auprc_path[1]]))
        c = np.nanmean(np.array([acc.item() for acc in auprc_path[2]]))

        aurocs = []
        for i in range(test_labels.shape[0]):
            try:
                aurocs.append(auroc(predicts[i], test_labels[i]).cpu())
            except:
                aurocs.append(torch.tensor(0.))
        roc = torch.stack([auc for auc in aurocs]).sum() / test_labels.shape[0]
        return auprc, roc, a, b, c


def main(args):
    #random_uuid = str(uuid.uuid1())
    #save_path = os.path.join(args.save_path, random_uuid)
    #if not os.path.exists(save_path):
        #os.mkdir(save_path)

    #with open(os.path.join(save_path, "args.json"), 'w') as f:
        #args_dict = args.__dict__
        #json.dump(args_dict, f, indent=4)
    save_path = "./save_path"

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
    # val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    num_feats = features.shape[1]
    train_labels = g.ndata['train_label']
    test_labels = g.ndata['test_label']
    in_feats = num_feats
    n_classes = 2035
    n_edges = g.edges()[0].shape[0]

    np.save(os.path.join(save_path, 'train_mask'), train_mask.cpu().numpy())
    np.save(os.path.join(save_path, 'test_mask'), test_mask.cpu().numpy())
    np.save(os.path.join(save_path, 'label'), labels.cpu().numpy())
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
    print(model)
    # if args.early_stop:
    #     stopper = EarlyStopping(patience=100)
    if cuda:
        model.cuda()
    # loss_fcn = torch.nn.BCEWithLogitsLoss()
    loss_fcn = torch.nn.BCELoss(reduction='mean')
    # loss_fcn= nn.MSELoss()

    # use optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    best_train = 0
    best_val = 0
    best_a = 0
    best_b = 0
    best_c = 0
    input_ids = torch.linspace(0, 18546, 18547).long().to(args.gpu)
    for epoch in range(args.epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(input_ids)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        if epoch % 100 == 0:
            model.eval()
            train_acc, train_roc, _, _, _ = evaluate(model, input_ids, test_labels, train_labels)

            val_acc, val_roc, a, b, c = evaluate(model, input_ids, train_labels, test_labels)
            # if args.early_stop:
            #     if stopper.step(val_acc, model):
            #         break
            val_loss = loss_fcn(logits[test_mask], labels[test_mask])

            if train_acc > best_train:
                best_train = train_acc
            if val_acc > best_val:
                best_a = a
                best_b = b
                best_c = c
                best_val = val_acc
                with open(os.path.join(save_path, 'result'), 'w') as f:
                    f.write("train: {:.4f}  val:{:.4f}\n".format(train_acc, val_acc))

                torch.save(model.state_dict(), os.path.join(save_path, "checkpoint.ckpt"))
            print("Epoch {:05d} | Loss {:.4f} | TrainAcc {:.4f} |"
                  " TrainRoc {:.4f} | ValLoss {:.4f} | ValAcc {:.4f} | ValROC {:.4f} | Best {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}".
                  format(epoch, loss.item(), train_acc, train_roc, val_loss, val_acc, val_roc, best_train, best_val,
                         best_a, best_b, best_c))

    print()
    acc, roc = evaluate(model, input_ids, train_labels, test_labels)
    print("Test Accuracy {:.4f}".format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT')
    register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=3,
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
    parser.add_argument("--lr", type=float, default=0.001,
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
