import ray
import torch

from model import GNODEFormer
from train_class import Trainer_General

from typing import List


class Server:
    def __init__(
        self,
        class_num,
        feature_dim,
        nlayer,
        hidden_dim,
        num_heads,
        tran_dropout,
        feat_dropout,
        prop_dropout,
        norm,
        trainers: List[Trainer_General],
        rk=2
    ):
        self.feature_dim = feature_dim
        self.nlayer = nlayer
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.tran_dropout = tran_dropout
        self.feat_dropout = feat_dropout
        self.prop_dropout = prop_dropout
        self.norm = norm
        self.rk=rk

        self.model = GNODEFormer(
            class_num,
            self.feature_dim,
            self.nlayer,
            self.hidden_dim,
            self.num_heads,
            self.tran_dropout,
            self.feat_dropout,
            self.prop_dropout,
            self.norm,
            self.rk
        )

        self.trainers = trainers
        self.num_of_trainers = len(trainers)
        self.broadcast_params(-1)

    @torch.no_grad()
    def zero_params(self):
        for p in self.model.parameters():
            p.zero_()

    @torch.no_grad()
    def train(self, current_global_epoch):
        for trainer in self.trainers:
            trainer.train.remote(current_global_epoch)
        params = [trainer.get_params.remote() for trainer in self.trainers]
        self.zero_params()

        while True:
            ready, left = ray.wait(params, num_returns=1, timeout=None)
            if ready:
                for t in ready:
                    for p, mp in zip(ray.get(t), self.model.parameters()):
                        mp.data += p.cpu()
            params = left
            if not params:
                break

        for p in self.model.parameters():
            p /= self.num_of_trainers

        self.broadcast_params(current_global_epoch)

    def broadcast_params(self, current_global_epoch):
        for trainer in self.trainers:
            trainer.update_params.remote(
                tuple(self.model.parameters()), current_global_epoch
            )
