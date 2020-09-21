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

parser.add_argument('--name', default='1', help='random run')
parser.add_argument('--dataset', type=str, default='cora', help='dataset')  # 'cora', 'citeseer', 'pubmed', 'polblogs'
parser.add_argument('--rate', type=float, default=0.4)  # delta
parser.add_argument('--eps-x', type=float, default=0.1) # eplison
parser.add_argument('--stepsize-x', type=float, default=1e-5)
parser.add_argument('--critic', type=str, default="bilinear")  # 'inner product', 'bilinear', 'separable'
parser.add_argument('--save-model', type=bool, default=True)
parser.add_argument('--attack-mode', type=str, default='both')  # 'A', 'X', 'both'
parser.add_argument('--show-task', type=bool, default=True)
parser.add_argument('--show-attack', type=bool, default=True)
parser.add_argument('--gpu', type=str, default="1")
parser.add_argument('--decay', type=bool, default=False)
parser.add_argument('--patience', type=int, default=20)
parser.add_argument('--hinge', type=bool, default=False)
parser.add_argument('--tau', type=float, default=0.01)
parser.add_argument('--dim', type=int, default=512)

args = parser.parse_args()

dataset = args.dataset

print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

make_adv = True
attack_rate = args.rate
attack_mode = args.attack_mode

# training params
batch_size = 1
nb_epochs = 10000
patience = args.patience
lr = 0.001
l2_coef = 0.0
drop_prob = 0.0
if dataset == 'pubmed':
    hid_units = 256
else:
    hid_units = args.dim
sparse = True
nonlinearity = 'prelu' # special name to separate parameters

if dataset == 'polblogs':
    adj, features, labels, idx_train, idx_val, idx_test = process.load_data_polblogs(dataset)
elif dataset == 'wiki':
    adj, features, labels, idx_train, idx_val, idx_test = process.load_data_wiki(dataset)
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
# if attack_mode != 'A':
features_ori = features.clone()
sp_A = sp_A.to_dense()


encoder = DGI(ft_size, hid_units, nonlinearity, critic=args.critic)
# atm = Attacker(encoder, sp_adj, sp_A, sp_adj_ori, features, nb_nodes)
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

if dataset == 'pubmed':
    step_size_init = 40
    attack_iters = 5
else:
    step_size_init = 20
    attack_iters = 10

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

    # torch.cuda.empty_cache()
    if make_adv:
        # step_size = step_size_init * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        step_size = step_size_init
        if args.decay:
            step_size_x = args.stepsize_x * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        else:
            step_size_x = args.stepsize_x
        adv = atm(sp_adj, sp_A, None, n_flips, b_xent=b_xent, step_size=step_size,
                  eps_x=args.eps_x, step_size_x=step_size_x,
                  iterations=attack_iters, should_normalize=True, random_restarts=False, make_adv=True)
        if attack_mode == 'A':
            sp_adj = adv
        elif attack_mode == 'X':
            features = adv
        elif attack_mode == 'both':
            sp_adj = adv[0]
            features = adv[1]

    # loss = mi_loss_neg(encoder, sp_adj, sp_adj_ori, features, nb_nodes, b_xent, batch_size, sparse)
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
                  eps_x=args.eps_x, step_size_x=1e-3,
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
            if args.critic == 'bilinear':
                torch.save(encoder.state_dict(), 'logcite0817/eval_param/{}_{}_e{}_s{}_r{}_p{}_decay{}_dim{}_hinge{}_{}.pkl'.
                               format(dataset, args.attack_mode, args.eps_x, args.stepsize_x, args.rate, args.patience,
                                      args.decay, args.dim, args.hinge, args.name))
                # torch.save(encoder.state_dict(), dataset+'_'+args.attack_mode+'_best_encoder_'+args.name+'_'+str(args.rate)+'.pkl')
            elif args.critic == 'inner product':
                torch.save(encoder.state_dict(), 'logcite0817/eval_param/{}_fix_{}_e{}_s{}_r{}_p{}_decay{}_dim{}_hinge{}_{}.pkl'.
                               format(dataset, args.attack_mode, args.eps_x, args.stepsize_x, args.rate, args.patience,
                                      args.decay, args.dim, args.hinge, args.name))
                # torch.save(encoder.state_dict(), dataset+'_'+args.attack_mode+'fix_best_encoder_'+args.name+'_'+str(args.rate)+'.pkl')
        # torch.save(atm.state_dict(), 'best_attacker'+args.name+'.pkl')
    else:
        cnt_wait += 1

    if cnt_wait == patience:
        print('Early stopping!')
        break

    loss.backward()
    optimiser.step()

# print('Loading {}th epoch'.format(best_t))
# encoder.load_state_dict(torch.load('best_dgi.pkl'))

name = ['adversarial', 'natural']
for i, adj in enumerate([sp_adj, sp_adj_ori]):
    embeds, _ = encoder.embed(features, adj, sparse, None)
    train_embs = embeds[0, idx_train]
    val_embs = embeds[0, idx_val]
    test_embs = embeds[0, idx_test]

    train_lbls = torch.argmax(labels[0, idx_train], dim=1)
    val_lbls = torch.argmax(labels[0, idx_val], dim=1)
    test_lbls = torch.argmax(labels[0, idx_test], dim=1)

    tot = torch.zeros(1)
    tot = tot.cuda()

    accs = []

    for _ in range(50):
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)
        log.cuda()

        pat_steps = 0
        best_acc = torch.zeros(1)
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
        print(acc)
        tot += acc

    print('{} Average accuracy: {}'.format(name[i], tot / 50))

    accs = torch.stack(accs)
    print(accs.mean())
    print(accs.std())

