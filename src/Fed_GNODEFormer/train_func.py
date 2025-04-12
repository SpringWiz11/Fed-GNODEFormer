
import torch
import torch.nn.functional as F


def accuracy(output: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def train(
    epoch,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    e,
    u,
    x,
    labels: torch.Tensor,
    idx_train: torch.Tensor,
    device=None
) -> tuple:
    model.train()
    optimizer.zero_grad()
    
    device = x.device if device is None else device
    labels = labels.to(device)
    
    logits, new_e = model(e, u, x)
    loss_train = F.cross_entropy(logits[idx_train], labels[idx_train])
    acc_train = accuracy(logits[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss_train.item(), acc_train.item(), new_e


def test(
    model: torch.nn.Module, e, u, x, labels: torch.Tensor, idx_test: torch.Tensor, device=None
) -> tuple:
    device = x.device if device is None else device
    labels = labels.to(device)
    model.eval()
    output, _ = model(e, u, x)
    loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])

    return loss_test.item(), acc_test.item()
