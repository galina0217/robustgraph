import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import networkx as nx
import random

from models import DGI, LogReg
from utils import process
from estimator.estimator import mi_loss, mi_loss_neg


from attacker.attacker import Attacker

import os, argparse

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--dataset', default='cora', help='dataset name')   # 'cora', 'citeseer', 'polblogs'
parser.add_argument('--alpha', type=float, default=0.2)
parser.add_argument('--epsilon', type=float, default=0.1)
parser.add_argument('--model', default='model.pkl', help='network name')
parser.add_argument('--critic', type=str, default="bilinear")
parser.add_argument('--dim', type=int, default=512)
parser.add_argument('--gpu', type=str, default="0")

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
dataset = args.dataset
attack_rate = args.alpha
attack_mode = 'both'

# training params
hid_units = args.dim
sparse = True
nonlinearity = 'prelu' # special name to separate parameters

if dataset == 'BlogCatalog':
    adj, features, labels, idx_train, idx_val, idx_test = process.load_data_blogcatalog(dataset)
elif dataset == 'polblogs':
    adj, features, labels, idx_train, idx_val, idx_test = process.load_data_polblogs(dataset)
elif dataset == 'wiki':
    adj, features, labels, idx_train, idx_val, idx_test = process.load_data_wiki(dataset)
else:
    adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)

A = adj.copy()
A.setdiag(0)
A.eliminate_zeros()
features, _ = process.preprocess_features(features, dataset=dataset)
# features_tack, _ = process.preprocess_features(sp.csr_matrix(features_tack))

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = labels.shape[1]
nb_edges = int(adj.sum() / 2)

adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))

if sparse:
    sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
    sp_A = process.sparse_mx_to_torch_sparse_tensor(A)
    # adj_tack = process.sparse_mx_to_torch_sparse_tensor(adj_tack)
else:
    adj = (adj + sp.eye(adj.shape[0])).todense()

features = torch.FloatTensor(features[np.newaxis])
# features_tack = torch.FloatTensor(features_tack[np.newaxis])
if not sparse:
    adj = torch.FloatTensor(adj[np.newaxis])
labels = torch.FloatTensor(labels[np.newaxis])
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

model = DGI(ft_size, hid_units, nonlinearity, critic=args.critic, dataset=dataset)
model.eval()

if torch.cuda.is_available():
    print('Using CUDA')
    model.cuda()
    features = features.cuda()
    # features_tack = features_tack.cuda()
    if sparse:
        sp_adj = sp_adj.cuda()
        sp_A = sp_A.cuda()
        # adj_tack = adj_tack.cuda()
    else:
        adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

sp_adj_ori = sp_adj.clone()
if attack_mode != 'A':
    features_ori = features.clone()

xent = nn.CrossEntropyLoss()
if args.gpu == "":
    model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))
else:
    model.load_state_dict(torch.load(args.model))
print("Load model")

embeds, _ = model.embed(features, sp_adj if sparse else adj, sparse, None)
train_embs = embeds[0, idx_train]
val_embs = embeds[0, idx_val]
test_embs = embeds[0, idx_test]

train_lbls = torch.argmax(labels[0, idx_train], dim=1)
val_lbls = torch.argmax(labels[0, idx_val], dim=1)
test_lbls = torch.argmax(labels[0, idx_test], dim=1)

tot = torch.zeros(1)
if torch.cuda.is_available():
    tot = tot.cuda()

accs = []

for _ in range(50):
    log = LogReg(hid_units, nb_classes)
    opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
    if torch.cuda.is_available():
        log.cuda()

    pat_steps = 0
    best_acc = torch.zeros(1)
    if torch.cuda.is_available():
        best_acc = best_acc.cuda()
    for _ in range(100):
        log.train()
        opt.zero_grad()

        logits = log(train_embs)
        loss = xent(logits, train_lbls)

        loss.backward()
        opt.step()

    logits = log(test_embs)
    preds = torch.argmax(logits, dim=1)
    acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
    accs.append(acc * 100)
    # print(acc)
    tot += acc

accs = torch.stack(accs)
natural_acc_mean = accs.mean().detach().cpu().numpy()
natural_acc_std = accs.std().detach().cpu().numpy()


print("== Start Attacking ==")
b_xent = nn.BCEWithLogitsLoss()
atm = Attacker(model, features, nb_nodes, attack_mode=attack_mode, gpu=torch.cuda.is_available())
if torch.cuda.is_available():
    atm = atm.cuda()
n_flips = int(attack_rate * nb_edges)
step_size = 20
iterations = 50
stepsize_x = 1e-3

acc_list = []
sp_A_ori = sp_A.clone()
for _ in range(10):
    adv = atm(sp_adj_ori.to_dense(), sp_A_ori.to_dense(), None, n_flips, eps_x=args.epsilon, step_size_x=stepsize_x,
              b_xent=b_xent, step_size=step_size, iterations=iterations, should_normalize=True,
              random_restarts=False, make_adv=True, return_a=False)
    if attack_mode == 'A':
        embeds, _ = model.embed(features, adv, sparse, None)
    elif attack_mode == 'X':
        embeds, _ = model.embed(adv, sp_adj, sparse, None)
    elif attack_mode == 'both':
        embeds, _ = model.embed(adv[1], adv[0], sparse, None)

    loss = mi_loss(model, adv[0], adv[1], nb_nodes, b_xent, 1, sparse)
    loss_ori = mi_loss(model, sp_adj_ori, features_ori, nb_nodes, b_xent, 1, sparse)
    RV = loss - loss_ori
    print("RV: {}; MI-nature: {}; MI-worst: {}".format(RV.detach().cpu().numpy(),
                                                       loss_ori.detach().cpu().numpy(),
                                                       loss.detach().cpu().numpy()))

    train_embs = embeds[0, idx_train]
    val_embs = embeds[0, idx_val]
    test_embs = embeds[0, idx_test]

    train_lbls = torch.argmax(labels[0, idx_train], dim=1)
    val_lbls = torch.argmax(labels[0, idx_val], dim=1)
    test_lbls = torch.argmax(labels[0, idx_test], dim=1)

    tot = torch.zeros(1)
    if torch.cuda.is_available():
        tot = tot.cuda()

    accs = []

    for _ in range(5):
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
        if torch.cuda.is_available():
            log.cuda()

        pat_steps = 0
        best_acc = torch.zeros(1)
        if torch.cuda.is_available():
            best_acc = best_acc.cuda()
        for _ in range(100):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc * 100)
        # print(acc)
        tot += acc

    accs = torch.stack(accs)
    print("accuracy: {} (std: {})".format(accs.mean().detach().cpu().numpy(), accs.std().detach().cpu().numpy()))
    acc_list += accs.detach().cpu().tolist()

print(acc_list)
print('Adversarial accuracy: {} (std: {})'.format(np.mean(acc_list), np.std(acc_list)))
print('Natural accuracy: {} (std: {})'.format(natural_acc_mean, natural_acc_std))