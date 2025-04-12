import numpy as np
import torch

from dataprocess import eigen_decomposition
from model import GNODEFormer
from train_func import test, train


class Trainer_General:
    def __init__(
        self,
        rank: int,
        class_num,
        adj,
        x,
        y,
        idx_train,
        idx_test,
        nlayer,
        hidden_dim,
        num_heads,
        tran_dropout,
        feat_dropout,
        prop_dropout,
        norm,
        lr,
        weight_decay,
        local_step,
        device='cpu',
        rk=2
    ):
        torch.manual_seed(rank)

        self.rank = rank

        self.train_losses = []
        self.train_accs = []

        self.test_losses = []
        self.test_accs = []

        self.x = x
        self.adj = adj
        self.e, self.u = eigen_decomposition(self.adj)
        self.e = torch.FloatTensor(self.e)
        self.u = torch.FloatTensor(self.u)
        self.y = y
        self.nlayer = nlayer
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.tran_dropout = tran_dropout
        self.feat_dropout = feat_dropout
        self.prop_dropout = prop_dropout
        self.norm = norm
        self.lr = lr
        self.weight_decay = weight_decay

        self.device = device

        self.idx_train = idx_train
        self.idx_test = idx_test
        self.rk = rk

        self.model = GNODEFormer(
            class_num,
            self.x.shape[1],
            self.nlayer,
            self.hidden_dim,
            self.num_heads,
            self.tran_dropout,
            self.feat_dropout,
            self.prop_dropout,
            self.norm,
            self.rk
        ).to(self.device)

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.local_step = local_step
        self.new_e = []

    @torch.no_grad()
    def update_params(self, params, current_global_epoch) -> None:
        self.model.to('cpu')
        for p, mp in zip(params, self.model.parameters()):
            mp.data = p
        self.model.to(self.device)

    def train(self, current_global_round) -> None:
        torch.cuda.empty_cache()
        for iteration in range(self.local_step):
            self.model.train()

            loss_train, acc_train, self.new_e = train(
                iteration,
                self.model,
                self.optimizer,
                self.e,
                self.u,
                self.x,
                self.y,
                self.idx_train,
                device=self.device
            )
            self.train_losses.append(loss_train)
            self.train_accs.append(acc_train)

            loss_test, acc_test = self.local_test()

            self.test_losses.append(loss_test)
            self.test_accs.append(acc_test)

    def local_test(self) -> list:
        local_test_loss, local_test_acc = test(
            self.model, self.e, self.u, self.x, self.y, self.idx_test, device=self.device
        )
        return [local_test_loss, local_test_acc]

    def get_params(self) -> tuple:
        self.optimizer.zero_grad(set_to_none=True)
        return tuple(self.model.parameters())

    def get_e(self):
        return self.e, self.new_e

    def get_all_loss_accuracy(self) -> list:
        return [
            np.array(self.train_losses),
            np.array(self.train_accs),
            np.array(self.test_losses),
            np.array(self.test_accs),
        ]

    def get_rank(self):
        return self.rank