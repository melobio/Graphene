
import json
import pickle

import dgl
import numpy as np
import torch
from scipy import sparse
from torch import nn


def get_noed_label():
    NCI_set_path = "Reactome_node_set_entrez.txt"
    with open(NCI_set_path, 'r') as fin:
        lines = fin.readlines()

    # 获取每个node的set
    node_entrez = []
    node2set = {}
    for line in lines:
        set_name, node_num = line.strip().split('\t')
        node_entrez.append(int(node_num))
        if node_num in node2set:
            node2set[node_num].append(set_name)
        else:
            node2set[node_num] = [set_name]
    print(len(node2set))

    # 转化为node的label
    node_label = {}
    for node in node2set:
        label = [0] * 2408
        # label = [0] * 211
        sets = node2set[node]
        for set in sets:
            label[int(set[3:]) - 1] = 1
        node_label[node] = label
    return node_label


def get_node_feature(mode=None):
    if mode == "set2Gus":
        node_embed = np.load(
            '../../network_data/output_embedg2g_node_emb.out.npy')
        with open('/../../network_data/map.json', 'r') as f:
            i2g, g2i = json.load(f)
        feature_dict = {}
        for i, node in enumerate(i2g.values()):
            feature_dict[node] = node_embed[i].tolist()
        print("entrez num:", len(feature_dict))
    elif mode == "pretrain":
        #node_embed = np.load('../pretrain/emb.txt.npy')
        node_embed = np.load(r'../../network_data/emb.txt.npy')
        #with open("../pretrain/nodeid_to_index.pickle", 'rb') as f:
        with open(r"../../network_data/nodeid_to_index.pickle", 'rb') as f:
            node_index = pickle.load(f)
        feature_dict = {}
        for i, node in enumerate(node_index.keys()):
            feature_dict[str(node)] = node_embed[i].tolist()
        print("entrez num:", len(feature_dict))
    else:
        feature_path = "../../network_data/feature_100.json"
        with open(feature_path, 'r') as f:
            feature_data = json.load(f)
        feature_dict = {}
        for data in feature_data['data']:
            feature_dict[data['entrez_id']] = data['feature']
        print("entrez num:", len(feature_dict))
    return feature_dict


def get_network():
    #network_numpy = np.load('/data/nfsdata2/sunzijun/bio/9606v11&set2gau_entrez/set2Gau_src/network.npy')
    network_numpy = np.load(r'../../network_data/network.npy')
    # 构造图
    network_scipy = sparse.csr_matrix(network_numpy)
    network = dgl.from_scipy(network_scipy)

    # 添加边权重
    weights = network_scipy.data
    # network.edata['w'] = weights

    # 添加node feature
    feature_dict = get_node_feature(mode='pretrain')
    #with open('/data/nfsdata2/sunzijun/bio/9606v11&set2gau_entrez/set2Gau_src/map.json', 'r') as f:
    with open(r'../../network_data/map.json', 'r') as f:
        i2g, g2i = json.load(f)
    entrezs = [value for value in i2g.values()]
    feature_list = []
    init_num = 0
    for ent in entrezs:
        if ent in feature_dict:
            feature_list.append(feature_dict[ent])
        else:
            init_num += 1
            feature = nn.init.kaiming_normal_(torch.empty(1, 100))
            feature_list.append(feature.tolist()[0])
    print(init_num / len(entrezs), " node are initialized.")

    # network.ndata['feat'] = torch.randn((18547, 100))
    features = torch.FloatTensor(feature_list)
    network.ndata['feat'] = features

    # 添加node label
    not_in_num = 0
    node_label = get_noed_label()
    node_label_list = []
    for ent in entrezs:
        if ent in node_label:
            node_label_list.append(node_label[ent])
        else:
            not_in_num += 1
            node_label_list.append([0] * 2408)
    print(not_in_num, " not in node label")
    labels = torch.FloatTensor(node_label_list)
    # 只取一个
    # labels = labels[:, 0].view(-1, 1)
    network.ndata['label'] = labels

    train_mask = torch.bernoulli(torch.full([labels.shape[0], labels.shape[1]], 0.5)).bool()
    train_labels = train_mask * labels
    true_train_mask = train_labels != labels
    network.ndata['train_label'] = train_labels

    test_mask = ~train_mask
    test_labels = test_mask * labels
    true_test_mask = test_labels != labels
    network.ndata['test_label'] = test_labels

    # 构造train val test的mask
    # node_num = labels.shape[0]
    # train_mask = torch.bernoulli(torch.full([labels.shape[0]], 0.5)).bool()
    # val_mask = torch.bernoulli(torch.full([labels.shape[0]], 1)).bool()
    # test_mask = torch.bernoulli(torch.full([labels.shape[0]], 1)).bool()

    network.ndata['train_mask'] = ~true_train_mask
    # network.ndata['val_mask'] = val_mask
    network.ndata['test_mask'] = ~true_test_mask

    # 返回构造的网络
    print(network)
    return network


def get_subnetwork():
    # network_numpy = np.load('/data/nfsdata2/sunzijun/bio/9606v11&set2gau_entrez/set2Gau_src/network.npy')
    # network_numpy = np.array([[0.5, 0.5, 0.5], [0.6, 0.5, 0.5], [0.5, 0.5, 0.5]])
    # network_numpy = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    # 构造图
    # network_scipy = sparse.csr_matrix(network_numpy)
    # network = dgl.from_scipy(network_scipy)

    # 初始化一个大图
    #
    source = torch.LongTensor((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18))
    target = torch.LongTensor((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19))
    network = dgl.graph((source, target))
    network = dgl.to_bidirected(network)

    # 添加边权重
    # weights = network_scipy.data
    # network.edata['w'] = weights

    # 添加node feature
    feature_dict = get_node_feature(mode='pretrain')
    with open('../../network_data/map.json', 'r') as f:
        i2g, g2i = json.load(f)
    entrezs = [value for value in i2g.values()]
    feature_list = []
    init_num = 0
    for ent in entrezs:
        if ent in feature_dict:
            feature_list.append(feature_dict[ent])
        else:
            init_num += 1
            feature = nn.init.kaiming_normal_(torch.empty(1, 100))
            feature_list.append(feature.tolist()[0])
    print(init_num / len(entrezs), " node are initialized.")

    features = torch.FloatTensor(feature_list)
    network.ndata['feat'] = torch.randn((20, 100))
    # network.ndata['feat'] = features[:20]

    # 添加node label
    not_in_num = 0
    node_label = get_noed_label()
    node_label_list = []
    for ent in entrezs:
        if ent in node_label:
            node_label_list.append(node_label[ent])
        else:
            not_in_num += 1
            node_label_list.append([0] * 211)
    print(not_in_num, " not in node label")
    # node_label_list = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 1, 0, 0, 0, 0, 0]]
    labels = torch.FloatTensor(node_label_list)
    network.ndata['label'] = labels[:20]

    # 构造train val test的mask
    node_num = labels[:20].shape[0]
    train_mask = torch.randint(low=0, high=1, size=(node_num,))
    val_mask = 1 - train_mask
    test_mask = 1 - train_mask
    # train_mask = torch.zeros(node_num)
    # train_mask[:10000] = 1
    # val_mask = torch.zeros(node_num)
    # val_mask[:3000] = 1
    # test_mask = torch.zeros(node_num)
    # test_mask[:3000] = 1

    network.ndata['train_mask'] = train_mask == 1
    network.ndata['val_mask'] = val_mask == 1
    network.ndata['test_mask'] = test_mask == 1

    # 返回构造的网络
    print(network)
    return network


if __name__ == '__main__':
    get_network()
    # get_node_feature(mode='deepwalk')
    get_noed_label()
    get_subnetwork()
