#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random

import networkx as nx
import torch
from torch.utils.data import Dataset

from dataloader import DataLoaderSubstructContext
from loader import graph_data_obj_to_nx, nx_to_graph_data_obj


class NewBioDataset(Dataset):
    def __init__(self, l1, center):
        self.datasets = []
        self.center = center
        self.l1 = l1

    def __getitem__(self, item):
        var = self.datasets[item]
        return var

    def __len__(self):
        return len(self.datasets)

    def process(self, data, root_idx=None):
        num_atoms = data.x.size()[0]
        G = graph_data_obj_to_nx(data)

        if root_idx == None:
            if self.center == True:
                root_idx = data.center_node_idx.item()
            else:
                root_idx = random.sample(range(num_atoms), 1)[0]

        # in the PPI case, the subgraph is the entire PPI graph
        data.x_substruct = data.x
        data.edge_attr_substruct = data.edge_attr
        data.edge_index_substruct = data.edge_index
        data.center_substruct_idx = data.center_node_idx

        # Get context that is between l1 and the max diameter of the PPI graph
        l1_node_idxes = nx.single_source_shortest_path_length(G, root_idx,
                                                              self.l1).keys()
        # l2_node_idxes = nx.single_source_shortest_path_length(G, root_idx,
        #                                                       self.l2).keys()
        l2_node_idxes = range(num_atoms)
        context_node_idxes = set(l1_node_idxes).symmetric_difference(
            set(l2_node_idxes))
        if len(context_node_idxes) > 0:
            context_G = G.subgraph(context_node_idxes)
            context_G, context_node_map = reset_idxes(context_G)  # need to
            # reset node idx to 0 -> num_nodes - 1, other data obj does not
            # make sense
            context_data = nx_to_graph_data_obj(context_G, 0)  # use a dummy
            # center node idx
            data.x_context = context_data.x
            data.edge_attr_context = context_data.edge_attr
            data.edge_index_context = context_data.edge_index

        # Get indices of overlapping nodes between substruct and context,
        # WRT context ordering
        context_substruct_overlap_idxes = list(context_node_idxes)
        if len(context_substruct_overlap_idxes) > 0:
            context_substruct_overlap_idxes_reorder = [context_node_map[old_idx]
                                                       for
                                                       old_idx in
                                                       context_substruct_overlap_idxes]
            data.overlap_context_substruct_idx = \
                torch.tensor(context_substruct_overlap_idxes_reorder)

        return data


def reset_idxes(G):
    """
    Resets node indices such that they are numbered from 0 to num_nodes - 1
    :param G:
    :return: copy of G with relabelled node indices, mapping
    """
    mapping = {}
    for new_idx, old_idx in enumerate(G.nodes()):
        mapping[old_idx] = new_idx
    new_G = nx.relabel_nodes(G, mapping, copy=True)
    return new_G, mapping


datasets = NewBioDataset(l1=1, center=0)
loader = DataLoaderSubstructContext(datasets, batch_size=5, shuffle=False)
for batch in loader:
    print(batch)
