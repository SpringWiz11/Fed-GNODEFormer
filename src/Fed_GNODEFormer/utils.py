import numpy as np
import torch
import torch.nn as nn


def intersect1d(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    combined = torch.cat((t1, t2))
    uniques, counts = combined.unique(return_counts=True)
    intersection = uniques[counts > 1]
    return intersection


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.1)
        if module.bias is not None:
            module.bias.data.zero_()


def get_split(y, nclass):
    percls_trn = int(round(0.6 * len(y) / nclass))
    val_lb = int(round(0.2 * len(y)))

    indices = []
    for i in range(nclass):
        index = (y == i).nonzero().view(-1)

        index = index[torch.randperm(index.size(0), device=index.device)]
        indices.append(index)

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)
    rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]
    valid_index = rest_index[:val_lb]
    test_index = rest_index[val_lb:]

    return train_index, valid_index, test_index


def label_dirichlet_partition(labels, tot_data, uniq_labels, num_clients, alpha):
    min_size = 0
    min_require_size = 20

    split_data_indexes = []

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(uniq_labels):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))

            proportions = np.array(
                [
                    p * (len(idx_j) < tot_data / num_clients)
                    for p, idx_j in zip(proportions, idx_batch)
                ]
            )

            proportions = proportions / proportions.sum()

            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

            idx_batch = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
            ]

            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(num_clients):
        np.random.shuffle(idx_batch[j])
        split_data_indexes.append(idx_batch[j])

    return split_data_indexes


def get_indexes(
    split_data_indexes: list,
    num_clients: int,
    idx_train: torch.Tensor,
    idx_test: torch.Tensor,
) -> tuple:
    train_data_indexes = []

    for i in range(num_clients):
        inter = intersect1d(split_data_indexes[i], idx_train)
        train_data_indexes.append(
            torch.searchsorted(split_data_indexes[i], inter).clone()
        )

    test_data_indexes = []
    for i in range(num_clients):
        inter = intersect1d(split_data_indexes[i], idx_test)
        test_data_indexes.append(
            torch.searchsorted(split_data_indexes[i], inter).clone()
        )
    return (
        train_data_indexes,
        test_data_indexes,
    )
