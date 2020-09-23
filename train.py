import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from tqdm import tqdm
import networkx as nx
import random
import math, os
from collections import defaultdict

from models import DGI, LogReg
from utils import process

from attacker.attacker import Attacker
from estimator.estimator import mi_loss, mi_loss_neg
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--dataset', type=str, default='cora', help='dataset')  # 'cora', 'citeseer', 'polblogs'
parser.add_argument('--alpha', type=float, default=0.4)
parser.add_argument('--epsilon', type=float, default=0.1)
parser.add_argument('--tau', type=float, default=0.01)
parser.add_argument('--critic', type=str, default="bilinear")  # 'inner product', 'bilinear', 'separable'
parser.add_argument('--hinge', type=bool, default=True)
parser.add_argument('--dim', type=int, default=512)
parser.add_argument('--gpu', type=str, default="0")
parser.add_argument('--save-model', type=bool, default=True)
parser.add_argument('--show-task', type=bool, default=True)
parser.add_argument('--show-attack', type=bool, default=True)

args = parser.parse_args()

dataset = args.dataset

print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

make_adv = True
attack_rate = args.alpha

# training params
batch_size = 1
nb_epochs = 10000
patience = 20
lr = 0.001
l2_coef = 0.0
drop_prob = 0.0
hid_units = args.dim
sparse = True
if dataset == 'polblogs':
    attack_mode = 'A'
else:
    attack_mode = 'both'
nonlinearity = 'prelu' # special name to separate parameters

if dataset == 'polblogs':
    adj, features, labels, idx_train, idx_val, idx_test = process.load_data_polblogs(dataset)
else:
    adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = labels.shape[1]
nb_edges = int(adj.sum() / 2)
n_flips = int(nb_edges * attack_rate)

A = adj.copy()
features, _ = process.preprocess_features(features, dataset=dataset)

adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))

if sparse:
    sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
    sp_A = process.sparse_mx_to_torch_sparse_tensor(A)
else:
    adj = (adj + sp.eye(adj.shape[0])).todense()

features = torch.FloatTensor(features[np.newaxis])
if not sparse:
    adj = torch.FloatTensor(adj[np.newaxis])
labels = torch.FloatTensor(labels[np.newaxis])
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)


if torch.cuda.is_available():
    print('Using CUDA')
    features = features.cuda()
    if sparse:
        sp_adj = sp_adj.cuda()
        sp_A = sp_A.cuda()
    else:
        adj = adj.cuda()
        A = A.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

sp_adj = sp_adj.to_dense()
sp_adj_ori = sp_adj.clone()
features_ori = features.clone()
sp_A = sp_A.to_dense()


encoder = DGI(ft_size, hid_units, nonlinearity, critic=args.critic)
atm = Attacker(encoder, features, nb_nodes, attack_mode=attack_mode,
               show_attack=args.show_attack, gpu=torch.cuda.is_available())
optimiser = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=l2_coef)

if torch.cuda.is_available():
    encoder.cuda()
    atm.cuda()

b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
cnt_wait = 0
best = 1e9
best_t = 0

step_size_init = 20
attack_iters = 10
stepsize_x = 1e-5
# attack_mode = 'both'

drop = 0.8
epochs_drop = 20

train_lbls = torch.argmax(labels[0, idx_train], dim=1)
val_lbls = torch.argmax(labels[0, idx_val], dim=1)
test_lbls = torch.argmax(labels[0, idx_test], dim=1)

def task(embeds):
    train_embs = embeds[0, idx_train]
    val_embs = embeds[0, idx_val]
    test_embs = embeds[0, idx_test]

    log = LogReg(hid_units, nb_classes)
    opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
    if torch.cuda.is_available():
        log.cuda()

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
    return acc.detach().cpu().numpy()

for epoch in range(nb_epochs):
    encoder.train()
    optimiser.zero_grad()

    if make_adv:
        # step_size = step_size_init * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        step_size = step_size_init
        step_size_x = stepsize_x
        adv = atm(sp_adj, sp_A, None, n_flips, b_xent=b_xent, step_size=step_size,
                  eps_x=args.epsilon, step_size_x=step_size_x,
                  iterations=attack_iters, should_normalize=True, random_restarts=False, make_adv=True)
        if attack_mode == 'A':
            sp_adj = adv
        elif attack_mode == 'X':
            features = adv
        elif attack_mode == 'both':
            sp_adj = adv[0]
            features = adv[1]

    loss = mi_loss(encoder, sp_adj, features, nb_nodes, b_xent, batch_size, sparse)
    if args.hinge:
        loss_ori = mi_loss(encoder, sp_adj_ori, features_ori, nb_nodes, b_xent, batch_size, sparse)
        RV = loss - loss_ori
        print("RV: {}; RV-tau: {}; MI-nature: {}; MI-worst: {}".format(RV.detach().cpu().numpy(),
                                                                       (RV - args.tau).detach().cpu().numpy(),
                                                                       loss_ori.detach().cpu().numpy(),
                                                                       loss.detach().cpu().numpy()))
        if RV - args.tau < 0:
            loss = loss_ori

    if args.show_task and epoch%5==0:
        adv = atm(sp_adj_ori, sp_A, None, n_flips, b_xent=b_xent, step_size=20,
                  eps_x=args.epsilon, step_size_x=1e-3,
                  iterations=50, should_normalize=True, random_restarts=False, make_adv=True)
        if attack_mode == 'A':
            embeds, _ = encoder.embed(features_ori, adv, sparse, None)
        elif attack_mode == 'X':
            embeds, _ = encoder.embed(adv, sp_adj_ori, sparse, None)
        elif attack_mode == 'both':
            embeds, _ = encoder.embed(adv[1], adv[0], sparse, None)
        acc_adv = task(embeds)

        embeds, _ = encoder.embed(features_ori, sp_adj_ori, sparse, None)
        acc_nat = task(embeds)

        print('Epoch:{} Step_size: {:.4f} Loss:{:.4f} Natural_Acc:{:.4f} Adv_Acc:{:.4f}'.format(
            epoch, step_size, loss.detach().cpu().numpy(), acc_nat, acc_adv))
    else:
        print('Epoch:{} Step_size: {:.4f} Loss:{:.4f}'.format(epoch, step_size, loss.detach().cpu().numpy()))

    if loss < best:
        best = loss
        best_t = epoch
        cnt_wait = 0
        if args.save_model:
            torch.save(encoder.state_dict(), 'model.pkl')
    else:
        cnt_wait += 1

    if cnt_wait == patience:
        print('Early stopping!')
        break

    loss.backward()
    optimiser.step()
