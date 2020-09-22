import numpy as np
import pickle as pkl
import networkx as nx
from networkx.readwrite import json_graph
import scipy.sparse as sp
import scipy.io as sio
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import torch
import torch.nn as nn
from collections import defaultdict
import json
import os
from time import time
from sklearn.model_selection import train_test_split
from scipy.sparse.linalg import svds


def parse_skipgram(fname):
    with open(fname) as f:
        toks = list(f.read().split())
    nb_nodes = int(toks[0])
    nb_features = int(toks[1])
    ret = np.empty((nb_nodes, nb_features))
    it = 2
    for i in range(nb_nodes):
        cur_nd = int(toks[it]) - 1
        it += 1
        for j in range(nb_features):
            cur_ft = float(toks[it])
            ret[cur_nd][j] = cur_ft
            it += 1
    return ret


# Process a (subset of) a TU dataset into standard form
def process_tu(data, nb_nodes):
    nb_graphs = len(data)
    ft_size = data.num_features

    features = np.zeros((nb_graphs, nb_nodes, ft_size))
    adjacency = np.zeros((nb_graphs, nb_nodes, nb_nodes))
    labels = np.zeros(nb_graphs)
    sizes = np.zeros(nb_graphs, dtype=np.int32)
    masks = np.zeros((nb_graphs, nb_nodes))

    for g in range(nb_graphs):
        sizes[g] = data[g].x.shape[0]
        features[g, :sizes[g]] = data[g].x
        labels[g] = data[g].y[0]
        masks[g, :sizes[g]] = 1.0
        e_ind = data[g].edge_index
        coo = sp.coo_matrix((np.ones(e_ind.shape[1]), (e_ind[0, :], e_ind[1, :])), shape=(nb_nodes, nb_nodes))
        adjacency[g] = coo.todense()

    return features, adjacency, labels, sizes, masks


def micro_f1(logits, labels):
    # Compute predictions
    preds = torch.round(nn.Sigmoid()(logits))

    # Cast to avoid trouble
    preds = preds.long()
    labels = labels.long()

    # Count true positives, true negatives, false positives, false negatives
    tp = torch.nonzero(preds * labels).shape[0] * 1.0
    tn = torch.nonzero((preds - 1) * (labels - 1)).shape[0] * 1.0
    fp = torch.nonzero(preds * (labels - 1)).shape[0] * 1.0
    fn = torch.nonzero((preds - 1) * labels).shape[0] * 1.0

    # Compute micro-f1 score
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = (2 * prec * rec) / (prec + rec)
    return f1


"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""


def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)


###############################################
# This section of code adapted from tkipf/gcn #
###############################################

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_npz(file_name):
    """Load a SparseGraph from a Numpy binary file.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    sparse_graph : gust.SparseGraph
        Graph in sparse matrix format.

    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                              loader['adj_indptr']), shape=loader['adj_shape'])

        if 'attr_data' in loader:
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                                   loader['attr_indptr']), shape=loader['attr_shape'])
        else:
            attr_matrix = None

        labels = loader.get('labels')

    return adj_matrix, attr_matrix, labels


def train_val_test_split_tabular(arrays, train_size=0.5, val_size=0.3,
                                 test_size=0.2, stratify=None, random_state=None):

    """
    Split the arrays or matrices into random train, validation and test subsets.

    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
            Allowed inputs are lists, numpy arrays or scipy-sparse matrices.
    train_size : float, default 0.5
        Proportion of the dataset included in the train split.
    val_size : float, default 0.3
        Proportion of the dataset included in the validation split.
    test_size : float, default 0.2
        Proportion of the dataset included in the test split.
    stratify : array-like or None, default None
        If not None, data is split in a stratified fashion, using this as the class labels.
    random_state : int or None, default None
        Random_state is the seed used by the random number generator;

    Returns
    -------
    splitting : list, length=3 * len(arrays)
        List containing train-validation-test split of inputs.

    """
    if len(set(array.shape[0] for array in arrays)) != 1:
        raise ValueError("Arrays must have equal first dimension.")
    idx = np.arange(arrays[0].shape[0])
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=random_state,
                                                   train_size=(train_size + val_size),
                                                   test_size=test_size,
                                                   stratify=stratify)
    if stratify is not None:
        stratify = stratify[idx_train_and_val]
    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=random_state,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)
    result = []
    for X in arrays:
        result.append(X[idx_train])
        result.append(X[idx_val])
        result.append(X[idx_test])
    return result


def load_data(dataset_str, mode=''):  # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    graph = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(graph)
    adj_lists = {i: set(list(graph.neighbors(i))) for i in range(adj.shape[0])}

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    if mode == 'ind':
        return adj, adj_lists, features, labels, idx_train, idx_val, idx_test
    return adj, features, labels, idx_train, idx_val, idx_test

def load_data_wiki(dataset):
    label_set = [1, 5, 10, 17, 2, 12, 6]
    label_dict = {l: i for i, l in enumerate(label_set)}
    old_label = []
    for l in open('data/wiki_label.txt'):
        old_label.append(int(l.strip().split('\t')[1]) - 1)

    node_remain = [i for i, l in enumerate(old_label) if l in label_set]
    node_dict = {node_remain[i]: i for i in range(len(node_remain))}

    N = len(node_dict)
    labels = np.zeros((N,7))
    for i, l in enumerate(old_label):
        if i in node_remain:
            labels[node_dict[i], int(label_dict[l])] = 1

    n_ori_features = 4973
    features = np.zeros((N, n_ori_features))
    for l in open('data/wiki_tfidf.txt'):
        d = l.strip().split('\t')
        if int(d[0]) in node_remain:
            features[node_dict[int(d[0])]][int(d[1])] = float(d[2])

    n_features = 200
    features = sp.csr_matrix(features, dtype=float)
    u, s, vt = svds(features, k=n_features)
    features = u * s

    edge_list = []
    with open('data/wiki.edgelist.n0', 'r') as f:
        for l in f:
            edge_list.append([int(l.strip().split()[0]), int(l.strip().split()[1])])

    g = nx.Graph()
    g.add_nodes_from([i for i in range(N)])
    g.add_edges_from(edge_list)
    adj = nx.adjacency_matrix(g, nodelist=sorted(g.nodes()))

    unlabeled_share = 0.8
    val_share = 0.1
    train_share = 1 - unlabeled_share - val_share

    idx_train, idx_val, idx_test = train_val_test_split_tabular(
        tuple(np.expand_dims(np.arange(N), 0)),
        train_size=train_share,
        val_size=val_share,
        test_size=unlabeled_share,
        stratify=labels)

    return adj, features, labels, idx_train, idx_val, idx_test

def load_data_ppi(dataset):
    mat_contents = sio.loadmat('data/PPI_subgraph.mat')
    labels = mat_contents["group"].toarray().astype('float32')
    adj = mat_contents["net"]
    adj.setdiag(0)

    label = np.where(labels)[1]
    _N = adj.shape[0]
    features = np.eye(_N)

    unlabeled_share = 0.8
    val_share = 0.1
    train_share = 1 - unlabeled_share - val_share

    idx_train, idx_val, idx_test = train_val_test_split_tabular(
        tuple(np.expand_dims(np.arange(_N), 0)),
        train_size=train_share,
        val_size=val_share,
        test_size=unlabeled_share,
        stratify=labels)

    return sp.csr_matrix(adj.astype(int)), sp.csr_matrix(features.astype(np.float32)), labels, idx_train, idx_val, idx_test

def load_data_blogcatalog(dataset):
    mat_contents = sio.loadmat('data/{}.mat'.format(dataset))
    features = mat_contents["Attributes"]
    label = mat_contents["Label"]
    adj = mat_contents["Network"]
    adj.setdiag(0)
    n, m = features.shape  # num of nodes

    _K = label.max()
    labels = np.eye(_K)[label-1]
    labels = labels.squeeze()

    # train_share = 0.1
    # val_share = 0.2
    # unlabeled_share = 1 - train_share - val_share
    unlabeled_share = 0.8
    val_share = 0.1
    train_share = 1 - unlabeled_share - val_share

    idx_train, idx_val, idx_test = train_val_test_split_tabular(
        tuple(np.expand_dims(np.arange(n), 0)),
        train_size=train_share,
        val_size=val_share,
        test_size=unlabeled_share,
        stratify=labels)

    return sp.csr_matrix(adj.astype(int)), sp.csr_matrix(features.astype(np.float32)), labels, idx_train, idx_val, idx_test

def load_data_polblogs(dataset):
    """Load data."""
    _A_obs, _X_obs, _z_obs = load_npz('data/polblogs.npz')
    _A_obs = _A_obs + _A_obs.T
    _A_obs[_A_obs > 1] = 1
    _A_obs.setdiag(0)

    # assert np.abs(_A_obs - _A_obs.T).sum() == 0, "Input graph is not symmetric"
    # assert _A_obs.max() == 1 and len(np.unique(_A_obs[_A_obs.nonzero()].A1)) == 1, "Graph must be unweighted"
    # assert _A_obs.sum(0).A1.min() > 0, "Graph contains singleton nodes"

    _N = _A_obs.shape[0]
    _K = _z_obs.max() + 1
    _Z_obs = np.eye(_K)[_z_obs]
    _X_obs = np.eye(_N)

    unlabeled_share = 0.8
    val_share = 0.1
    train_share = 1 - unlabeled_share - val_share

    idx_train, idx_val, idx_test = train_val_test_split_tabular(
        tuple(np.expand_dims(np.arange(_N), 0)),
        train_size=train_share,
        val_size=val_share,
        test_size=unlabeled_share,
        stratify=_z_obs)
    # print(split_train, split_val, split_unlabeled)

    return _A_obs, _X_obs, _Z_obs, idx_train, idx_val, idx_test

def load_data_reddit(prefix='data/reddit/reddit', normalize=True, load_walks=False, mode=None):
    # Save normalized version
    npz_file = prefix + '.npz'

    if os.path.exists(npz_file):
    # if False:
        start_time = time()
        print('Found preprocessed dataset {}, loading...'.format(npz_file))
        data = np.load(npz_file, allow_pickle=True)
        features = data['features']
        labels = data['labels']
        idx_train = data['idx_train']
        idx_val = data['idx_val']
        idx_test = data['idx_test']
        # adj = sp.csr_matrix((data['adj_data'], data['adj_indices'], data['adj_indptr']),
        #                     shape=data['adj_shape'])
        adj_lists = data['adj_lists'][()]
        print('Finished in {} seconds.'.format(time() - start_time))

    else:
        print('Loading data...')
        start_time = time()

        G_data = json.load(open(prefix + "-G.json"))
        G = json_graph.node_link_graph(G_data)

        if os.path.exists(prefix + "-feats.npy"):
            features = np.load(prefix + "-feats.npy")
        else:
            print("No features present.. Only identity features will be used.")
            features = None

        id_map = json.load(open(prefix + "-id_map.json"))
        if isinstance(list(G.nodes())[0], int):
            conversion = lambda n: int(n)
        else:
            conversion = lambda n: n
        id_map = {conversion(k): int(v) for k, v in id_map.items()}

        walks = []
        class_map = json.load(open(prefix + "-class_map.json"))
        if isinstance(list(class_map.values())[0], list):
            lab_conversion = lambda n: n
        else:
            lab_conversion = lambda n: int(n)
        class_map = {conversion(k): lab_conversion(v) for k, v in class_map.items()}

        # Remove all nodes that do not have val/test annotations
        # (necessary because of networkx weirdness with the Reddit data)
        broken_count = 0
        nodelist = list(G.nodes())
        for node in nodelist:
            if node not in id_map:
            # if 'val' not in G.nodes[node] or 'test' not in G.nodes[node]:
                G.remove_node(node)
                broken_count += 1
        print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

        # Construct adjacency matrix
        print("Loaded data ({} seconds).. now preprocessing..".format(time() - start_time))
        start_time = time()

        # Process edges
        edges = []
        for edge in G.edges():
            if edge[0] in id_map and edge[1] in id_map:
                edges.append((id_map[edge[0]], id_map[edge[1]]))
        print('{} edges'.format(len(edges)))
        print('{} features'.format(len(features)))
        print('{} labels'.format(len(class_map)))
        print('{} graph nodes'.format(len(list(G.nodes()))))
        print('{} all nodes'.format(len(id_map)))
        num_data = len(id_map)

        adj_lists = {i: set(list(G.neighbors(i))) for i in range(num_data)}

        # Split dataset
        idx_val = np.array([id_map[n] for n in G.nodes()
                            if 'val' in G.nodes[n] and G.nodes[n]['val']])
        idx_test = np.array([id_map[n] for n in G.nodes()
                             if 'test' in G.nodes[n] and G.nodes[n]['test']])
        is_train = np.ones(num_data, dtype=np.bool)
        is_train[idx_val] = False
        is_train[idx_test] = False
        idx_train = np.array([id_map[n] for n in G.nodes()
                             if ('val' in G.nodes[n] and 'test' in G.nodes[n]) and
                              not G.nodes[n]['val'] and not G.nodes[n]['test']])

        edges = np.array(edges)

        # Process labels
        if isinstance(list(class_map.values())[0], list):
            num_classes = len(list(class_map.values())[0])
            labels = np.zeros((num_data, num_classes))
            for k in class_map.keys():
                labels[id_map[k], :] = np.array(class_map[k])
        else:
            num_classes = len(set(class_map.values()))
            labels = np.zeros((num_data, num_classes))
            for k in class_map.keys():
                labels[id_map[k], class_map[k]] = 1

        # Make sure the graph has edge train_removed annotations
        # (some datasets might already have this..)
        for edge in G.edges():
            if ('val' not in G.nodes[edge[0]] or 'val' not in G.nodes[edge[1]] or
                    'test' not in G.nodes[edge[0]] or 'test' not in G.nodes[edge[1]]):
                G[edge[0]][edge[1]]['train_removed'] = True
            elif (G.nodes[edge[0]]['val'] or G.nodes[edge[1]]['val'] or
                    G.nodes[edge[0]]['test'] or G.nodes[edge[1]]['test']):
                G[edge[0]][edge[1]]['train_removed'] = True
            else:
                G[edge[0]][edge[1]]['train_removed'] = False

        if normalize and features is not None:
            from sklearn.preprocessing import StandardScaler
            # train_ids = np.array([id_map[n] for n in G.nodes() if not G.nodes[n]['val'] and not G.nodes[n]['test']])
            train_feats = features[idx_train]
            scaler = StandardScaler()
            scaler.fit(train_feats)
            features = scaler.transform(features)

        def _normalize_adj(edges):
            adj = sp.csr_matrix((np.ones((edges.shape[0]), dtype=np.float),
                                 (edges[:, 0], edges[:, 1])), shape=(num_data, num_data))
            adj += adj.transpose()

            rowsum = np.array(adj.sum(1)).flatten()
            d_inv = 1.0 / (rowsum + 1e-20)
            d_mat_inv = sp.diags(d_inv, 0)
            adj = d_mat_inv.dot(adj).tocoo()
            coords = np.array((adj.row, adj.col))
            return adj.data, coords

        full_v, full_coords = _normalize_adj(edges)

        def _get_adj(data, coords):
            adj = sp.csr_matrix((data, (coords[0, :], coords[1, :])),
                                shape=(num_data, num_data))
            return adj

        adj = _get_adj(full_v, full_coords)

        if load_walks:
            with open(prefix + "-walks.txt") as fp:
                for line in fp:
                    walks.append(map(conversion, line.split()))

        print("Done. {} seconds.".format(time() - start_time))

        with open(npz_file, 'wb') as fwrite:
            print('Saving {} edges'.format(adj.nnz))
            np.savez(fwrite,
                     adj_data=adj.data, adj_indices=adj.indices, adj_indptr=adj.indptr,
                     adj_shape=adj.shape, adj_lists=adj_lists,
                     features=features, labels=labels,
                     idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)

    return adj_lists, features, labels, idx_train, idx_val, idx_test


def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # standardize data
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f


def preprocess_features(features, dataset=''):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    if dataset == 'reddit' or dataset == 'polblogs' or dataset == 'wiki':
        return features, None
    return features.todense(), sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
