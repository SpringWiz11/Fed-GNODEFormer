
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, nheads, tran_dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, nheads, tran_dropout)
        self.normalize_before = True
        self.dropout = tran_dropout
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(2)])

    def forward(self, x):
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x, _ = self.self_attn(x,x,x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x
        
class BaseLayerHistory(nn.Module):
    def __init__(self, is_encoder,hidden_dim):
        super(BaseLayerHistory, self).__init__()
        self.is_encoder = is_encoder
        self.normalize_before = True

        layers = 6
        dim = hidden_dim
        self.layer_norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(layers))

    def add(self, layer):
        raise NotImplemented

    def pop(self):
        raise NotImplemented

    def clean(self):
        raise NotImplemented

class ResidualLayerHistory(BaseLayerHistory):
    def __init__(self, is_encoder,hidden_dim):
        super(ResidualLayerHistory, self).__init__(is_encoder,hidden_dim)
        self.count = 0
        self.x = None
        self.y = None

    def add(self, layer):
        layer = layer.to(next(self.layer_norms.parameters()).device)        
        if self.x is None:
            self.x = layer
            self.count += 1
            return
        self.count += 1
        if self.normalize_before:
            self.y = self.layer_norms[self.count - 2](layer)
        else:
            self.y = layer

    def pop(self):
        assert self.x is not None
        
        if self.y is None:
            return self.x
        ret = self.x + self.y
        if not self.normalize_before:
            ret = self.layer_norms[self.count - 2](ret)
        self.x = ret
        return ret

    def clean(self):
        self.x = None
        self.y = None
        self.count = 0

class RungeKuttaTransformerEncoder(nn.Module):
    def __init__(self, hidden_dim, nheads, dropout, steps=2):
        super().__init__()
        self.encoding = SineEncoding(hidden_dim)
        self.residual_history = ResidualLayerHistory(is_encoder=True, hidden_dim=hidden_dim)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, nheads, dropout) for _ in range(6)
        ])
        self.use_layer_norm = True
        self.layer_norm = nn.LayerNorm(hidden_dim) if self.use_layer_norm else None
        self.num_rk_steps = steps 

    def apply_runge_kutta(self, x_input, encoder_layer):
        step_outputs = []
        current = x_input

        for rk_step in range(self.num_rk_steps):
            out = encoder_layer(current)
            step_outputs.append(out)

            if self.num_rk_steps == 4:
                if rk_step in [0, 1]:
                    current = x_input + 0.5 * out
                elif rk_step == 2:
                    current = x_input + out
            elif self.num_rk_steps == 3:
                if rk_step == 0:
                    current = x_input + 0.5 * out
                elif rk_step == 1:
                    current = x_input - step_outputs[1] + 2 * out
            elif self.num_rk_steps == 2:
                current = x_input + out

        if self.num_rk_steps == 4:
            return x_input + (1 / 6) * (step_outputs[0] + 2 * step_outputs[1] + 2 * step_outputs[2] + step_outputs[3])
        elif self.num_rk_steps == 3:
            return x_input + (1 / 6) * (step_outputs[0] + 4 * step_outputs[1] + step_outputs[2])
        elif self.num_rk_steps == 2:
            return x_input + 0.5 * (step_outputs[0] + step_outputs[1])

    def forward(self, eigen_feats):
        device = next(self.parameters()).device
        eigen_feats = eigen_feats.to(device)

        self.residual_history.clean()

        hidden = self.encoding(eigen_feats)
        self.residual_history.add(hidden)

        for encoder_layer in self.encoder_layers:
            hidden = self.residual_history.pop()
            updated_hidden = self.apply_runge_kutta(hidden, encoder_layer)
            self.residual_history.add(updated_hidden)

        output = self.residual_history.pop()

        if self.use_layer_norm:
            output = self.layer_norm(output)

        return output


class SineEncoding(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.constant = 100
        self.hidden_dim = hidden_dim
        self.eig_w = nn.Linear(hidden_dim + 1, hidden_dim)

    def forward(self, e):
        device = next(self.parameters()).device 
        e = e.to(device)
        ee = e * self.constant
        div = torch.exp(
            torch.arange(0, self.hidden_dim, 2, device=device) * (-math.log(10000) / self.hidden_dim)
        )
        pe = ee.unsqueeze(1) * div
        eeig = torch.cat((e.unsqueeze(1), torch.sin(pe), torch.cos(pe)), dim=1)
        return self.eig_w(eeig)


class SpecLayer(nn.Module):
    def __init__(self, nbases, ncombines, prop_dropout=0.0, norm="none"):
        super().__init__()
        self.prop_dropout = nn.Dropout(prop_dropout)

        if norm == "none":
            self.weight = nn.Parameter(torch.ones((1, nbases, ncombines)))
        else:
            self.weight = nn.Parameter(torch.empty((1, nbases, ncombines)))
            nn.init.normal_(self.weight, mean=0.0, std=0.01)

        if norm == "layer":
            self.norm = nn.LayerNorm(ncombines)
        elif norm == "batch":
            self.norm = nn.BatchNorm1d(ncombines)
        else:
            self.norm = None

    def forward(self, x):
        device = next(self.parameters()).device 
        x = x.to(device)
        weight = self.weight.to(device)
        
        x = self.prop_dropout(x) * weight
        
        x = torch.sum(x, dim=1)

        if self.norm is not None:
            x = self.norm(x)
            x = F.relu(x)

        return x


class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        device = next(self.parameters()).device
        x = x.to(device)
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class GNODEFormer(nn.Module):
    def __init__(
        self,
        nclass,
        nfeat,
        nlayer=1,
        hidden_dim=128,
        nheads=1,
        tran_dropout=0.0,
        feat_dropout=0.0,
        prop_dropout=0.0,
        norm="none",
        rk=2
    ):
        super().__init__()
        self.norm = norm
        self.nfeat = nfeat
        self.nlayer = nlayer
        self.nheads = nheads
        self.hidden_dim = hidden_dim
        self.rk = rk

        self.feat_encoder = nn.Sequential(
            nn.Linear(nfeat, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nclass),
        )

        self.linear_encoder = nn.Linear(nfeat, hidden_dim)
        self.classify = nn.Linear(hidden_dim, nclass)

        self.eig_encoder = SineEncoding(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, nheads)

        self.mha_form = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.mha_dropout = nn.Dropout(tran_dropout)
        self.ffn_dropout = nn.Dropout(tran_dropout)
        self.mha = nn.MultiheadAttention(hidden_dim, nheads, tran_dropout)
        self.ffn = FeedForwardNetwork(hidden_dim, hidden_dim, hidden_dim)
        self.encoder = RungeKuttaTransformerEncoder(self.hidden_dim,self.nheads,tran_dropout,self.rk)
        self.feat_dp1 = nn.Dropout(feat_dropout)
        self.feat_dp2 = nn.Dropout(feat_dropout)

        if norm == "none":
            self.layers = nn.ModuleList(
                [
                    SpecLayer(nheads + 1, nclass, prop_dropout, norm=norm)
                    for _ in range(nlayer)
                ]
            )
        else:
            self.layers = nn.ModuleList(
                [
                    SpecLayer(nheads + 1, hidden_dim, prop_dropout, norm=norm)
                    for _ in range(nlayer)
                ]
            )

    def forward(self, e, u, x):
        device = next(self.parameters()).device
        e = e.to(device)
        u = u.to(device)
        x = x.to(device)
        N = e.size(0)
        ut = u.permute(1, 0).to(device)

        if self.norm == "none":
            h = self.feat_dp1(x)
            h = self.feat_encoder(h)
            h = self.feat_dp2(h)
        else:
            h = self.feat_dp1(x)
            h = self.linear_encoder(h)
        out = self.encoder(e)
        new_e = self.decoder(out)

        for conv in self.layers:
            basic_feats = [h]
            utx = ut @ h
            for i in range(self.nheads):
                basic_feats.append(u @ (new_e[:, i].unsqueeze(1) * utx))
            basic_feats = torch.stack(basic_feats, dim=1)
            h = conv(basic_feats)

        if self.norm == "none":
            return h, new_e
        else:
            h = self.feat_dp2(h)
            h = self.classify(h)
            return h, new_e
