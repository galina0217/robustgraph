import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np

class Encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, features, feature_dim, embed_dim, adj_lists, aggregator,
                 num_sample=10, base_model=None, gcn=False, cuda=False,
                 feature_transform=False, activation=None, skip_connection=False, name=None):
        super(Encoder, self).__init__()

        self.name = name
        self.features = features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample
        self.act = nn.PReLU() if activation == 'prelu' else F.ReLU()
        self.skip_connection = skip_connection
        if base_model != None:
            self.base_model = base_model

        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.aggregator.cuda = cuda

        dim = self.feat_dim if self.gcn else 2 * self.feat_dim
        if skip_connection:
            dim = dim if self.name == 'l2' else 2 * dim
        self.weight = nn.Parameter(
                torch.FloatTensor(embed_dim, dim))
        self.weight_skip = nn.Parameter(
            torch.FloatTensor(embed_dim, dim))

        init.xavier_uniform_(self.weight)
        init.xavier_uniform_(self.weight_skip)

    def forward(self, nodes, shuffle=False):
        """
        Generates embeddings for a batch of nodes.

        nodes     -- list of nodes
        """
        # print("enc:" + self.name)
        # print "weights 1:", self.weight[0]

        neigh_feats, shuf_neigh_feats, skip_feats, shuf_skip_feats\
            = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes],
                                              self.num_sample, shuffle=shuffle)
        # print('pos neg feature complete!')
        if not self.gcn:
            if self.cuda:
                self_feats = self.features(torch.LongTensor(nodes).cuda())
            else:
                self_feats = self.features(torch.LongTensor(nodes))
            combined1 = neigh_feats
            combined2 = shuf_neigh_feats
        else:
            combined1 = neigh_feats
            combined2 = shuf_neigh_feats
        # print "weights 2:", self.weight[0]

        if self.skip_connection:
            combined1_skip = self.weight_skip.mm(skip_feats.t())
            combined2_skip = self.weight_skip.mm(shuf_skip_feats.t())

            combined1_mp = self.weight.mm(combined1.t())
            combined2_mp = self.weight.mm(combined2.t())

            # # print combined
            combined1 = self.act(torch.cat([combined1_skip, combined1_mp], 0))
            combined2 = self.act(torch.cat([combined2_skip, combined2_mp], 0))
        else:
            combined1 = self.act(self.weight.mm(combined1.t()))
            combined2 = self.act(self.weight.mm(combined2.t()))
        return combined1, combined2
