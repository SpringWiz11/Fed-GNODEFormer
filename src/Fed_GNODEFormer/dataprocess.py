import warnings
warnings.filterwarnings("ignore")

import pickle as pkl

import dgl
import networkx as nx
import numpy as np
import pandas as pd
import scipy as sp
import torch
from numpy.linalg import eigh
from ogb.nodeproppred.dataset_dgl import DglNodePropPredDataset
from scipy import io
from scipy.sparse import diags
from sklearn.preprocessing import label_binarize

from utils import get_split


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index



def normalize_graph(g):
    g = np.array(g)
    g = g + g.T
    g[g > 0.0] = 1.0
    try:
        deg = g.sum(axis=1).reshape(-1)
        deg[deg == 0.0] = 1.0
        deg =  np.diag(deg**-0.5)
    except Exception as e:
        deg = np.array(g.sum(axis=1)).flatten() 
        deg =  diags(deg**-0.5)
    
    adj = np.dot(np.dot(deg, g), deg)
    L = np.eye(g.shape[0]) - adj
    return L


def eigen_decomposition(g):
    g = normalize_graph(g)
    e, u = eigh(g)
    return e, u


def eig_dgl_adj_sparse(g, sm=0, lm=0):
    try:
        A = g.adj_external(scipy_fmt='csr')
    except Exception as e:
        print(e)
        raise ValueError(e)
    deg = np.array(A.sum(axis=0)).flatten()
    D_ = sp.sparse.diags(deg ** -0.5)

    A_ = D_.dot(A.dot(D_))
    L_ = sp.sparse.eye(g.num_nodes()) - A_

    if sm > 0:
        e1, u1 = sp.sparse.linalg.eigsh(L_, k=sm, which='SM', tol=1e-5)
        e1, u1 = map(torch.FloatTensor, (e1, u1))

    if lm > 0:
        e2, u2 = sp.sparse.linalg.eigsh(L_, k=lm, which='LM', tol=1e-5)
        e2, u2 = map(torch.FloatTensor, (e2, u2))

    if sm > 0 and lm > 0:
        return torch.cat((e1, e2), dim=0), torch.cat((u1, u2), dim=1)
    elif sm > 0:
        return e1, u1
    elif lm > 0:
        return e2, u2
    else:
        pass


def feature_normalize(x):
    x = np.array(x)
    rowsum = x.sum(axis=1, keepdims=True)
    rowsum = np.clip(rowsum, 1, 1e10)
    return x / rowsum



def load_data(dataset_str):
    if dataset_str in ["cora", "citeseer"]:
        names = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
        objects = []
        for i in range(len(names)):
            with open(
                "node_raw_data/{}/ind.{}.{}".format(dataset_str, dataset_str, names[i]),
                "rb",
            ) as f:
                objects.append(pkl.load(f, encoding="latin1"))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file(
            "node_raw_data/{}/ind.{}.test.index".format(dataset_str, dataset_str)
        )
        test_idx_range = np.sort(test_idx_reorder)

        if dataset_str == "citeseer":
            test_idx_range_full = range(
                min(test_idx_reorder), max(test_idx_reorder) + 1
            )
            tx_extended = sp.sparse.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.sparse.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)

        adj = adj.todense()
        features = torch.FloatTensor(features.todense())
        labels = torch.LongTensor(labels)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        if len(labels.size()) > 1:
            if labels.size(1) > 1:
                labels = torch.argmax(labels, dim=1)
            else:
                labels = labels.view(-1)


    elif dataset_str in ["chameleon", "squirrel", "actor"]:
        edge_df = pd.read_csv(
            "node_raw_data/{}/".format(dataset_str) + "out1_graph_edges.txt", sep="\t"
        )
        node_df = pd.read_csv(
            "node_raw_data/{}/".format(dataset_str) + "out1_node_feature_label.txt",
            sep="\t",
        )
        feature = node_df[node_df.columns[1]]
        y = node_df[node_df.columns[2]]

        num_nodes = len(y)
        adj = np.zeros((num_nodes, num_nodes))

        source = list(edge_df[edge_df.columns[0]])
        target = list(edge_df[edge_df.columns[1]])

        for i in range(len(source)):
            adj[source[i], target[i]] = 1.0
            adj[target[i], source[i]] = 1.0

        if dataset_str == 'actor':
            nfeat = 932
            x = np.zeros((len(y), nfeat))

            feature = list(feature)
            feature = [feat.split(',') for feat in feature]
            for ind, feat in enumerate(feature):
                for ff in feat:
                    x[ind, int(ff)] = 1.

            x = feature_normalize(x)
        else:
            feature = list(feature)
            feature = [feat.split(",") for feat in feature]
            new_feat = []

            for feat in feature:
                new_feat.append([int(f) for f in feat])

            x = np.array(new_feat)
        features = torch.FloatTensor(x)
        labels = torch.LongTensor(y)

        if len(labels.size()) > 1:
            if labels.size(1) > 1:
                labels = torch.argmax(labels, dim=1)
            else:
                labels = labels.view(-1)

        nclass = labels.max().item() + 1

        train_index, valid_index, test_index = get_split(labels, nclass)

        idx_train = torch.LongTensor(train_index)
        idx_val = torch.LongTensor(valid_index)
        idx_test = torch.LongTensor(test_index)
    
           
    else:
        raise ValueError(f"Invalid dataset. Can't Load {dataset_str}")

    return adj, features.float(), labels, idx_train, idx_val, idx_test
