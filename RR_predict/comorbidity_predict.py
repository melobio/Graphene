# This code is based on decagon from
#
#   https://github.com/mims-harvard/decagon
#
# Our contribution is the incorporation of GAT
# encoder into the bipartite network.


from __future__ import division
from __future__ import print_function

import os
import pickle
import time
from operator import itemgetter

import h5py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp
import tensorflow.compat.v1 as tf
from scipy import sparse
from sklearn import metrics

from decagon.deep.minibatch import EdgeMinibatchIterator
from decagon.deep.model import DecagonModel
from decagon.deep.optimizer import DecagonOptimizer
from decagon.metrics import bedroc_score
from decagon.utility import rank_metrics, preprocessing

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

tf.disable_eager_execution()
# os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True

np.random.seed(0)


def tsne_visualization(matrix):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    plt.figure(dpi=300)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, random_state=0,
                n_iter=1000)
    tsne_results = tsne.fit_transform(matrix)
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def draw_graph(adj_matrix):
    G = nx.from_scipy_sparse_matrix(adj_matrix)
    pos = nx.spring_layout(G, iterations=100)
    d = dict(nx.degree(G))
    nx.draw(G, pos, node_color=range(3215), nodelist=d.keys(),
            node_size=[v * 20 + 20 for v in d.values()], cmap=plt.cm.Dark2)
    plt.show()


def get_accuracy_scores(edges_pos, edges_neg, edge_type, name=None):
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['batch_edge_type_idx']: minibatch.edge_type2idx[edge_type]})
    feed_dict.update({placeholders['batch_row_edge_type']: edge_type[0]})
    feed_dict.update({placeholders['batch_col_edge_type']: edge_type[1]})
    rec = sess.run(opt.predictions, feed_dict=feed_dict)

    def sigmoid(x):
        return 1. / (1 + np.exp(-x))

    preds = []
    actual = []
    predicted = []
    edge_ind = 0
    for u, v in edges_pos[edge_type[:2]][edge_type[2]]:
        score = sigmoid(rec[u, v])
        preds.append(score)

        assert adj_mats_orig[edge_type[:2]][edge_type[2]][u, v] == 1, 'Problem 1'

        actual.append(edge_ind)
        predicted.append((score, edge_ind))
        edge_ind += 1

    preds_neg = []
    for u, v in edges_neg[edge_type[:2]][edge_type[2]]:
        score = sigmoid(rec[u, v])
        preds_neg.append(score)
        assert adj_mats_orig[edge_type[:2]][edge_type[2]][u, v] == 0, 'Problem 0'

        predicted.append((score, edge_ind))
        edge_ind += 1

    preds_all = np.hstack([preds, preds_neg])
    preds_all = np.nan_to_num(preds_all)
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    predicted = list(zip(*sorted(predicted, reverse=True, key=itemgetter(0))))[1]

    #plot
    precision, recall, thresholds = metrics.precision_recall_curve(labels_all, preds_all)
    #plot

    roc_sc = metrics.roc_auc_score(labels_all, preds_all)
    aupr_sc = metrics.average_precision_score(labels_all, preds_all)
    apk_sc = rank_metrics.apk(actual, predicted, k=200)
    bedroc_sc = bedroc_score(labels_all, preds_all)
    if name != None:
        with open(name, 'wb') as f:
            pickle.dump([labels_all, preds_all], f)
    return roc_sc, aupr_sc, apk_sc, bedroc_sc, precision, recall


def construct_placeholders(edge_types):
    placeholders = {
        'batch': tf.placeholder(tf.int64, name='batch'), # 把原来的tf.int32改成64
        'batch_edge_type_idx': tf.placeholder(tf.int64, shape=(), name='batch_edge_type_idx'),
        'batch_row_edge_type': tf.placeholder(tf.int64, shape=(), name='batch_row_edge_type'),
        'batch_col_edge_type': tf.placeholder(tf.int64, shape=(), name='batch_col_edge_type'),
        'degrees': tf.placeholder(tf.int64),
        'dropout': tf.placeholder_with_default(0., shape=()),
    }
    placeholders.update({
        'adj_mats_%d,%d,%d' % (i, j, k): tf.sparse_placeholder(tf.float32)
        for i, j in edge_types for k in range(edge_types[i, j])})
    placeholders.update({
        'feat_%d' % i: tf.sparse_placeholder(tf.float32)
        for i, _ in edge_types})
    return placeholders


def network_edge_threshold(network_adj, threshold):
    edge_tmp, edge_value, shape_tmp = preprocessing.sparse_to_tuple(network_adj)
    preserved_edge_index = np.where(edge_value > threshold)[0]
    preserved_network = sp.csr_matrix(
        (edge_value[preserved_edge_index],
         (edge_tmp[preserved_edge_index, 0], edge_tmp[preserved_edge_index, 1])),
        shape=shape_tmp)
    return preserved_network


def get_prediction(edges_pos, edges_neg, edge_type):
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['batch_edge_type_idx']: minibatch.edge_type2idx[edge_type]})
    feed_dict.update({placeholders['batch_row_edge_type']: edge_type[0]})
    feed_dict.update({placeholders['batch_col_edge_type']: edge_type[1]})
    rec = sess.run(opt.predictions, feed_dict=feed_dict)

    return 1. / (1 + np.exp(-rec))


#gene_phenes_path = './data_prioritization/genes_phenes.mat'
#f = h5py.File(gene_phenes_path, 'r')

# tep 1: 获取【基因-基因】网络
gene_network_numpy = np.load(r'../network_data/network.npy')
gene_network_numpy = (gene_network_numpy > 0).astype(float)
gene_network_adj = sparse.csr_matrix(gene_network_numpy)
gene_network_adj = gene_network_adj.tocsr()  # shape=(18547, 18547)
gen_bias_adj = -1e9 * (1 - gene_network_numpy)
gen_bias_adj = sparse.csr_matrix(gen_bias_adj).tocsr()

# step 2: 获取【疾病-疾病】网络
disease_network_numpy = np.load('./comorbidity/disease_disease.npy')
disease_network_numpy = (disease_network_numpy > 1).astype(float)
disease_network_adj = sparse.csr_matrix(disease_network_numpy)
disease_network_adj = disease_network_adj.tocsr()  # shape=(299, 299)
disease_network_adj = network_edge_threshold(disease_network_adj, 0.9)
disease_bias_adj = -1e9 * (1 - disease_network_numpy)
disease_bias_adj = sparse.csr_matrix(disease_bias_adj).tocsr()

# step 3: 获取【基因-疾病】网络
gene_disease_network_numpy = np.load('./comorbidity/disease_gen.npy')
gene_disease_adj = sparse.csr_matrix(gene_disease_network_numpy)
gene_disease_adj = gene_disease_adj.tocsr()  # shape=(18547, 299)

# step 4: 获取基因 feature
gene_feat = np.load('./comorbidity/gen_feat.npy')
gene_feat = sp.csc_matrix(gene_feat)
gene_nonzero_feat, gene_num_feat = gene_feat.shape
gene_feat = preprocessing.sparse_to_tuple(gene_feat.tocoo())

# step 5: 获取疾病 feature
disease_feat = np.load('./comorbidity/disease_feat.npy')
dis_feat = sp.csc_matrix(disease_feat)
dis_nonzero_feat, dis_num_feat = dis_feat.shape
dis_feat = preprocessing.sparse_to_tuple(dis_feat.tocoo())

dis_dis_adj_list = list()
dis_dis_adj_list.append(disease_network_adj)

val_test_size = 0.1
# n_genes = 12331
# n_dis = 3215
n_dis_rel_types = len(dis_dis_adj_list)
gene_adj = gene_network_adj
gene_degrees = np.array(gene_adj.sum(axis=0)).squeeze()

gene_dis_adj = gene_disease_adj
dis_gene_adj = gene_dis_adj.transpose(copy=True)

dis_degrees_list = [np.array(dis_adj.sum(axis=0)).squeeze() for dis_adj in dis_dis_adj_list]

adj_mats_orig = {
    (0, 0): [gene_adj, gene_adj.transpose(copy=True)],
    (0, 1): [gene_dis_adj],
    (1, 0): [dis_gene_adj],
    (1, 1): dis_dis_adj_list + [x.transpose(copy=True) for x in dis_dis_adj_list],
}

adj_bias_orig = {
    0: [gen_bias_adj],
    1: [disease_bias_adj]
}

degrees = {
    0: [gene_degrees, gene_degrees],
    1: dis_degrees_list + dis_degrees_list,
}

num_feat = {
    0: gene_num_feat,
    1: dis_num_feat,
}
nonzero_feat = {
    0: gene_nonzero_feat,
    1: dis_nonzero_feat,
}
feat = {
    0: gene_feat,
    1: dis_feat,
}

edge_type2dim = {k: [adj.shape for adj in adjs] for k, adjs in adj_mats_orig.items()}

edge_type2decoder = {
    (0, 0): 'bilinear',
    (0, 1): 'bilinear',
    (1, 0): 'bilinear',
    (1, 1): 'bilinear',
}

edge_types = {k: len(v) for k, v in adj_mats_orig.items()}
num_edge_types = sum(edge_types.values())
print("Edge types:", "%d" % num_edge_types)

if __name__ == '__main__':
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_integer('neg_sample_size', 1, 'Negative sample size.')
    flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train.')
    flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
    flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
    flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
    flags.DEFINE_float('weight_decay', 0.001, 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_float('dropout', 0.1, 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('max_margin', 0.1, 'Max margin parameter in hinge loss')
    flags.DEFINE_integer('batch_size', 512, 'minibatch size.')
    flags.DEFINE_string('encoder', 'gat', 'encoder type： gcn or gat')
    flags.DEFINE_boolean('bias', True, 'Bias term.')
    PRINT_PROGRESS_EVERY = 10

    print("Defining placeholders")
    placeholders = construct_placeholders(edge_types)

    print("Create minibatch iterator")
    minibatch = EdgeMinibatchIterator(
        adj_mats=adj_mats_orig,
        adj_bias=adj_bias_orig,
        feat=feat,
        edge_types=edge_types,
        batch_size=FLAGS.batch_size,
        val_test_size=val_test_size
    )

    print("Create model")
    model = DecagonModel(
        placeholders=placeholders,
        num_feat=num_feat,
        nonzero_feat=nonzero_feat,
        edge_types=edge_types,
        decoders=edge_type2decoder,
        encoder=FLAGS.encoder
    )

    print("Create optimizer")
    with tf.name_scope('optimizer'):
        opt = DecagonOptimizer(
            embeddings=model.embeddings,
            latent_inters=model.latent_inters,
            latent_varies=model.latent_varies,
            degrees=degrees,
            edge_types=edge_types,
            edge_type2dim=edge_type2dim,
            placeholders=placeholders,
            batch_size=FLAGS.batch_size,
            margin=FLAGS.max_margin
        )

    print("Initialize session")
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    feed_dict = {}
    saver = tf.train.Saver()

    saver.restore(sess, r'./checkpoint/gat_encoder/gat_dis.ckpt')
    feed_dict = minibatch.next_minibatch_feed_dict(placeholders=placeholders)
    feed_dict = minibatch.update_feed_dict(
        feed_dict=feed_dict,
        dropout=FLAGS.dropout,
        placeholders=placeholders)

    roc_score, auprc_score, apk_score, bedroc , precision, recall = get_accuracy_scores(
        minibatch.test_edges, minibatch.test_edges_false, minibatch.idx2edge_type[4])

    #plot
    #color
    ALL_Color = [[1, 0, 0, 1], [0, 252 / 255, 0, 1], [0, 0, 1, 1], [46 / 255, 217 / 255, 1, 1],
                 [1, 156 / 255, 85 / 255, 1], [1, 51 / 255, 129 / 255, 1],
                 [186 / 255, 12 / 255, 1, 1], [0, 0, 0, 1]]
    #color
    #s_ab
    y_true2 = np.load(r"./Sab_Source_Code (1)/source/s_ab_result/s_ab_labels.npy")
    y_scores2 = np.load(r"./Sab_Source_Code (1)/source/s_ab_result/s_ab_preds.npy")
    precision2, recall2, thresholds2 = metrics.precision_recall_curve(y_true2, y_scores2)
    # s_ab

    plt.title('Precision-Recall Curve')  # give plot a title
    plt.xlabel('Recall')  # make axis labels
    plt.ylabel('Precision')
    plt.plot(recall,precision, color = ALL_Color[2])
    plt.plot(recall2,precision2,  color = ALL_Color[3])

    plt.legend(["Graphene" ,"S_ab"])
    plt.show()
    #plot
    print("Edge type=", "[%02d, %02d, %02d]" % minibatch.idx2edge_type[4])
    print("Edge type:", "%04d" % 4, "Test AUROC score", "{:.5f}".format(roc_score))
    print("Edge type:", "%04d" % 4, "Test AUPRC score", "{:.5f}".format(auprc_score))
    print("Edge type:", "%04d" % 4, "Test AP@k score", "{:.5f}".format(apk_score))
    print("Edge type:", "%04d" % 4, "Test BEDROC score", "{:.5f}".format(bedroc))
    print()

    # prediction = get_prediction(minibatch.test_edges, minibatch.test_edges_false,
    #                             minibatch.idx2edge_type[4])
    #
    # print('Saving result...')
    # np.save('./comorbidity/result/prediction_d_d_2.npy', prediction)
