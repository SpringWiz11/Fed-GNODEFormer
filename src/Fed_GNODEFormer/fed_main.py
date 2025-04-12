import warnings
warnings.filterwarnings("ignore")

import argparse
import os
from datetime import datetime
from typing import Any

import numpy as np
import ray
import torch
from tqdm import tqdm

from dataprocess import load_data
from server import Server
from train_class import Trainer_General
from utils import get_indexes, label_dirichlet_partition


def parse_args():
    parser = argparse.ArgumentParser(description='Fed-GNODEFormer')
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-c", "--global_rounds", type=int)
    parser.add_argument("-i", "--local_step", type=int)
    parser.add_argument("-n", "--n_trainer", type=int)
    parser.add_argument("-a", "--alpha", type=float)
    parser.add_argument("-nl", "--nlayer", type=int)
    parser.add_argument("-hd", "--hidden_dim", type=int)
    parser.add_argument("-nh", "--num_heads", type=int)
    parser.add_argument("-rk", "--rk", type=int)
    parser.add_argument("-td", "--tran_dropout", type=float)
    parser.add_argument("-fd", "--feat_dropout", type=float)
    parser.add_argument("-pd", "--prop_dropout", type=float)
    parser.add_argument("-no", "--norm", type=str)
    parser.add_argument("-lr", "--lr", type=float)
    parser.add_argument("-wd", "--weight_decay", type=float)
    parser.add_argument("-r", "--repeat_time", type=int)
    parser.add_argument("-g", "--gpu", action="store_true")

    return parser.parse_args()



def setup_device(use_gpu):
    return torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")


def federated_train(args):
    ray.init()
    np.random.seed(42)
    torch.manual_seed(42)

    device = setup_device(args.gpu)
    print(f"Using device: {device}")

    adj, x, y, idx_train, idx_val, idx_test = load_data(args.dataset)
    class_num = y.max().item() + 1

    num_cpus = 0.1
    num_gpus = 0.1 if device.type == "cuda" else 0.0

    final_accs = []
    final_losses = []

    for repeat in range(args.repeat_time):
        print(f"Repeat {repeat + 1}/{args.repeat_time}")

        split_data_indexes = label_dirichlet_partition(y, len(y), class_num, args.n_trainer, args.alpha)
        split_data_indexes = [torch.tensor(sorted(i)) for i in split_data_indexes]

        in_com_train_data_indexes, in_com_test_data_indexes = get_indexes(
            split_data_indexes, args.n_trainer, idx_train, idx_test
        )

        @ray.remote(num_gpus=num_gpus, num_cpus=num_cpus, scheduling_strategy="SPREAD")
        class Trainer(Trainer_General):
            def __init__(self, *args: Any, **kwds: Any):
                super().__init__(*args, **kwds)

        trainers = [
            Trainer.remote(
                i, class_num,
                adj[split_data_indexes[i][:, None], split_data_indexes[i]],
                x[split_data_indexes[i]],
                y[split_data_indexes[i]],
                in_com_train_data_indexes[i],
                in_com_test_data_indexes[i],
                args.nlayer,
                args.hidden_dim,
                args.num_heads,
                args.tran_dropout,
                args.feat_dropout,
                args.prop_dropout,
                args.norm,
                args.lr,
                args.weight_decay,
                args.local_step,
                device,
                args.rk
            )
            for i in range(args.n_trainer)
        ]

        torch.cuda.empty_cache()

        server = Server(
            class_num, x.shape[1], args.nlayer, args.hidden_dim,
            args.num_heads, args.tran_dropout, args.feat_dropout,
            args.prop_dropout, args.norm, trainers, args.rk
        )

        for round_idx in tqdm(range(args.global_rounds), desc="Global Rounds"):
            server.train(round_idx)

        results = [trainer.get_all_loss_accuracy.remote() for trainer in server.trainers]
        results = np.array(ray.get(results))

        train_weights = [len(i) for i in in_com_train_data_indexes]
        test_weights = [len(i) for i in in_com_test_data_indexes]

        avg_train_loss = np.average([r[0] for r in results], weights=train_weights, axis=0)
        avg_train_acc = np.average([r[1] for r in results], weights=train_weights, axis=0)
        avg_test_loss = np.average([r[2] for r in results], weights=test_weights, axis=0)
        avg_test_acc = np.average([r[3] for r in results], weights=test_weights, axis=0)

        results = [trainer.local_test.remote() for trainer in server.trainers]
        results = np.array([ray.get(result) for result in results])

        average_final_test_loss = np.average(
            [row[0] for row in results], weights=test_weights, axis=0
        )
        average_final_test_accuracy = np.average(
            [row[1] for row in results], weights=test_weights, axis=0
        )   

        print(f"Average Test Loss: {average_final_test_loss}")
        print(f"Average Test Accuracy: {average_final_test_accuracy}")

        final_losses.append(average_final_test_loss)
        final_accs.append(average_final_test_accuracy)

    print(f"\n==== Final Aggregated Metrics Over {args.repeat_time} Repeats ====")
    print(f"Average Test Loss: {np.mean(final_losses):.4f}")
    print(f"Average Test Accuracy: {np.mean(final_accs):.4f}")
    ray.shutdown()


if __name__ == "__main__":
    args = parse_args()
    federated_train(args)
