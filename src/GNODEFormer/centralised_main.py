import time
import yaml
import copy
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from sklearn.metrics import roc_auc_score, mean_absolute_error, accuracy_score, r2_score
from model import GNODEFormer
from utils import count_parameters, init_params, seed_everything, get_split


def parse_args():
    parser = argparse.ArgumentParser(description='GNODEFormer')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--cuda', type=int, help='CUDA device index')
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--nclass', type=int, help='Number of classes')
    parser.add_argument('--nlayer', type=int, help='Number of layers')
    parser.add_argument('--num_heads', type=int, help='Number of attention heads')
    parser.add_argument('--hidden_dim', type=int, help='Hidden dimension size')
    parser.add_argument('--epoch', type=int, help='Number of epochs')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, help='Weight decay')
    parser.add_argument('--tran_dropout', type=float, help='Transformer dropout rate')
    parser.add_argument('--feat_dropout', type=float, help='Feature dropout rate')
    parser.add_argument('--prop_dropout', type=float, help='Propagation dropout rate')
    parser.add_argument('--norm', type=str, help='Normalization type')
    parser.add_argument('--rk', type=int, help="The RK value for the experiment")

    return parser.parse_args()



def centralized_main(args):
    print(args)
    seed_everything(args.seed)

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")


    epoch = args.epoch
    lr = args.lr
    weight_decay = args.weight_decay
    nclass = args.nclass
    nlayer = args.nlayer
    hidden_dim = args.hidden_dim
    num_heads = args.num_heads
    tran_dropout = args.tran_dropout
    feat_dropout =  args.feat_dropout
    prop_dropout = args.prop_dropout
    norm = args.norm
    rk = args.rk

    e, u, x, y = torch.load('data/{}.pt'.format(args.dataset))
    e, u, x, y = e.to(device), u.to(device), x.to(device), y.to(device)

    if len(y.size()) > 1:
        if y.size(1) > 1:
            y = torch.argmax(y, dim=1)
        else:
            y = y.view(-1)

    train, valid, test = get_split(args.dataset, y, nclass, args.seed) 
    train, valid, test = map(torch.LongTensor, (train, valid, test))
    train, valid, test = train.to(device), valid.to(device), test.to(device)

    nfeat = x.size(1)
    net = GNODEFormer(nclass, nfeat, nlayer, hidden_dim, num_heads, tran_dropout, feat_dropout, prop_dropout, norm, rk)
    net = net.to(device)
    net.apply(init_params)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    res = []
    min_loss = 100.0
    max_acc = 0
    counter = 0
    test_accs = []
    evaluation = torchmetrics.Accuracy(task='multiclass', num_classes=nclass)

    for idx in range(epoch):

        net.train()
        optimizer.zero_grad()
        logits = net(e, u, x)

        loss = F.cross_entropy(logits[train], y[train]) 

        loss.backward()
        optimizer.step()

        net.eval()
        logits = net(e, u, x)

        val_loss = F.cross_entropy(logits[valid], y[valid]).item()

        val_acc = evaluation(logits[valid].cpu(), y[valid].cpu()).item()
        test_acc = evaluation(logits[test].cpu(), y[test].cpu()).item()
        res.append([val_loss, val_acc, test_acc])

        test_accs.append(test_acc)

        if val_loss < min_loss:
            min_loss = val_loss
            counter = 0
        else:
            counter += 1

        if counter == 200:
            break
    print("Highest Test Accuracy:", max(test_accs))


if __name__ == '__main__':
    args = parse_args()
    centralized_main(args)
