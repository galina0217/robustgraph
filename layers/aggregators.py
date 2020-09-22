import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import random
from collections import Counter

"""
Set of modules for aggregating embeddings of neighbors.
"""

class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, features, cuda=False, gcn=False, name=None):
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()

        self.name = name
        self.features = features
        self.cuda = cuda
        self.gcn = gcn

    def forward(self, nodes, to_neighs, num_sample=10, shuffle=False):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # print("agg:" + self.name)
        # print("nodes: {}".format(len(nodes)))

        if type(nodes).__name__ != 'list':
            nodes = nodes.tolist()
        # Local pointers to functions (speed hack)
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh,
                            num_sample,
                            )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        if self.gcn:
            # samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
            samp_neighs = [set(list(samp_neigh) + [nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}

        # # dense version
        # mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        # column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        # row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        # mask[row_indices, column_indices] = 1
        # num_neigh = mask.sum(1, keepdim=True)
        # mask = mask.div(num_neigh)
        # mask[mask != mask] = 0
        #
        # mask_self = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        # column_indices = [unique_nodes[n] for n in nodes]
        # row_indices = range(len(samp_neighs))
        # mask_self[row_indices, column_indices] = 1
        # if self.cuda:
        #     mask_self = mask_self.cuda()
        #     mask = mask.cuda()

        # sparse version
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        i = torch.LongTensor([row_indices, column_indices])
        v = torch.FloatTensor([1/row_indices.count(ele) for ele in row_indices])
        mask = torch.sparse.FloatTensor(i, v, torch.Size([len(samp_neighs), len(unique_nodes)]))

        column_indices = [unique_nodes[n] for n in nodes]
        row_indices = range(len(samp_neighs))
        i = torch.LongTensor([row_indices, column_indices])
        v = torch.FloatTensor([1 for ele in row_indices])
        mask_self = torch.sparse.FloatTensor(i, v, torch.Size([len(samp_neighs), len(unique_nodes)]))
        if self.cuda:
            mask_self = mask_self.cuda()
            mask = mask.cuda()
        if self.cuda:
            if self.name == 'l1':
                embed_matrix = self.features[torch.LongTensor(unique_nodes_list).cuda()]
            else:
                embed_matrix1, embed_matrix2 = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            if self.name == 'l1':
                embed_matrix = self.features[torch.LongTensor(unique_nodes_list)]
            else:
                embed_matrix1, embed_matrix2 = self.features(torch.LongTensor(unique_nodes_list))
        if self.name == 'l1':
            idx = np.random.permutation(embed_matrix.shape[0])
            shuf_embed_matrix = embed_matrix[idx, :]
            shuf_to_feats = mask.mm(shuf_embed_matrix)
            to_feats = mask.mm(embed_matrix)

            skip_feats = mask_self.mm(embed_matrix)
            shuf_skip_feats = mask_self.mm(shuf_embed_matrix)
            return to_feats, shuf_to_feats, skip_feats, shuf_skip_feats
        else:
            to_feats = mask.mm(embed_matrix1.t())
            shuf_to_feats = mask.mm(embed_matrix2.t())

            skip_feats = mask_self.mm(embed_matrix1.t())
            shuf_skip_feats = mask_self.mm(embed_matrix2.t())
            return to_feats, shuf_to_feats, skip_feats, shuf_skip_feats


class MeanAggregator_ml(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, features, cuda=False, gcn=False, name=None):
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator_ml, self).__init__()

        self.name = name
        self.features = features
        self.cuda = cuda
        self.gcn = gcn

    def forward(self, nodes, to_neighs, num_sample=10, shuffle=False):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # print("agg:" + self.name)
        # print("nodes: {}".format(len(nodes)))

        if type(nodes).__name__ != 'list':
            nodes = nodes.tolist()
        # Local pointers to functions (speed hack)
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh,
                            num_sample,
                            )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        # print("successfully get sample neigh "+self.name)
        if self.gcn:
            # samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
            samp_neighs = [set(list(samp_neigh) + [nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}

        # # dense version
        # mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        # column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        # row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        # mask[row_indices, column_indices] = 1
        # if self.cuda:
        #     mask = mask.cuda()
        # num_neigh = mask.sum(1, keepdim=True)
        # mask = mask.div(num_neigh)
        # mask[mask != mask] = 0
        #
        # mask_self = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        # column_indices = [unique_nodes[n] for n in nodes]
        # row_indices = range(len(samp_neighs))
        # mask_self[row_indices, column_indices] = 1
        # if self.cuda:
        #     mask_self = mask_self.cuda()

        # print('length of sample neighbor:{}'.format(len(samp_neighs)))
        # print('length of unique node{}'.format(len(unique_nodes_list)))
        # sparse version
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        # print('188'+self.name)
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        # print('190'+self.name)
        i = torch.LongTensor([row_indices, column_indices])
        # print(193)
        count=Counter(row_indices)
        v = torch.FloatTensor([1/count[ele] for ele in row_indices])
        # print(195)
        mask = torch.sparse.FloatTensor(i, v, torch.Size([len(samp_neighs), len(unique_nodes)]))
        # print(196)
        column_indices = [unique_nodes[n] for n in nodes]
        row_indices = range(len(samp_neighs))
        i = torch.LongTensor([row_indices, column_indices])
        v = torch.FloatTensor([1 for ele in row_indices])
        mask_self = torch.sparse.FloatTensor(i, v, torch.Size([len(samp_neighs), len(unique_nodes)]))
        # print(202)
        if self.cuda:
            mask_self = mask_self.cuda()
            mask = mask.cuda()
        # print(self.name+"%%%%%%")
        if self.cuda:
            if self.name == 'l1':
                # print('207'+self.name)
                embed_matrix = self.features[torch.LongTensor(unique_nodes_list).cuda()]
                # print('layer l1:'+self.name)
            else:
                embed_matrix1, embed_matrix2 = self.features(torch.LongTensor(unique_nodes_list).cuda())
                # print(self.name)
        else:
            if self.name == 'l1':
                embed_matrix = self.features[torch.LongTensor(unique_nodes_list)]
            else:
                embed_matrix1, embed_matrix2 = self.features(torch.LongTensor(unique_nodes_list))

        if self.name == 'l1':
            idx = np.random.permutation(embed_matrix.shape[0])
            shuf_embed_matrix = embed_matrix[idx, :]
            shuf_to_feats = mask.mm(shuf_embed_matrix)
            to_feats = mask.mm(embed_matrix)

            skip_feats = mask_self.mm(embed_matrix)
            shuf_skip_feats = mask_self.mm(shuf_embed_matrix)
            return to_feats, shuf_to_feats, skip_feats, shuf_skip_feats
        else:
            to_feats = mask.mm(embed_matrix1.t())
            shuf_to_feats = mask.mm(embed_matrix2.t())

            skip_feats = mask_self.mm(embed_matrix1.t())
            shuf_skip_feats = mask_self.mm(embed_matrix2.t())
            return to_feats, shuf_to_feats, skip_feats, shuf_skip_feats
