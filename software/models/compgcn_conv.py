from .helper import *
from .message_passing import MessagePassing
import torch_scatter
import torch


class TARGELayer(MessagePassing):
    def __init__(self, in_feat, out_feat, bias_exist=None, activation=None, self_loop=False, dropout=0.0):
        super(TARGELayer, self).__init__()
        self.bias_exist = bias_exist
        self.activation = activation
        self.self_loop = self_loop

        if self.bias_exist:
            self.register_parameter('bias', Parameter(torch.zeros(out_feat)))

        # weight for self loop
        if self.self_loop:
            self.loop_weight = get_param((in_feat, out_feat))

        if dropout:
            self.drop = torch.nn.Dropout(dropout)
        else:
            self.drop = None

        self.bn = torch.nn.BatchNorm1d(out_feat)

class TARGEBLOCKLAYER(TARGELayer):
    def __init__(self, params, in_feat, out_feat, num_rels, num_bases=20, bias=None, activation=None, self_loop=False,
                 dropout=0.0):
        super(TARGEBLOCKLAYER, self).__init__(in_feat, out_feat, bias, activation, self_loop=self_loop,
                                                    dropout=dropout)
        self.p = params
        self.device = self.p.device
        self.opn = 'mult'
        self.num_rels = num_rels
        self.num_bases = num_bases
        assert self.num_bases > 0
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.act = activation

        self.submat_in = in_feat // self.num_bases
        self.submat_out = out_feat // self.num_bases

        # initialize node projector.
        self.node_proj = torch.nn.Linear(self.in_feat * 2, self.in_feat, bias=False)
        # the same initialization with get_param
        torch.nn.init.xavier_normal_(self.node_proj.weight)

        # W
        self.w = get_param((1, out_feat))

    def forward(self, x, edge_index, edge_type, ts_emb):

        num_e = x.size(0) - self.num_rels  # x raw embedding
        ent_emb = x[:num_e, :]
        rel_embed = x[num_e:, :]

        self.norm = self.compute_norm(edge_index, num_e)

        res = self.propagate('add', edge_index, edge_type=edge_type, rel_embed=rel_embed, x=ent_emb,
                             edge_norm=self.norm,
                             ts_emb=ts_emb, node_proj=self.node_proj)
        res = torch.cat((res,rel_embed),dim=0)
        out = self.drop(res)

        if self.p.bias: out = out + self.bias
        out = self.bn(out)

        return self.act(out)

    def forward_no_te(self, x, edge_index, edge_type):

        num_ent = x.size(0)  # x raw embedding

        self.norm = self.compute_norm(edge_index, num_ent)  # calculate the norm D_row^(-0.5)^D_col^(-0.5)
        res = self.propagate('add', edge_index, edge_type=edge_type, x=x, edge_norm=self.norm)
        out = self.drop(res)

        if self.p.bias: out = out + self.bias
        out = self.bn(out)

        return self.act(out)

    def rel_transform(self, ent_embed, rel_embed):
        if self.opn == 'corr':
            trans_embed = ccorr(ent_embed, rel_embed)
        elif self.opn == 'sub':
            trans_embed = ent_embed - rel_embed
        elif self.opn == 'mult':
            trans_embed = ent_embed * rel_embed
        else:
            raise NotImplementedError

        return trans_embed

    def message(self, x_j, edge_type, rel_embed, edge_norm):
        weight = self.w
        rel_emb = torch.index_select(rel_embed, 0, edge_type)
        xj_rel = self.rel_transform(x_j, rel_emb)
        out = xj_rel * weight

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out):
        return aggr_out

    def compute_norm(self, edge_index, num_ent):
        row, col = edge_index
        edge_weight = torch.ones_like(row).float()
        # length = dim_size, element values=freqeuncy of edge_index
        deg = torch_scatter.scatter_add(edge_weight, row, dim=0,
                                        dim_size=num_ent)  # Summing number of weights of the edges
        deg_inv = deg.pow(-0.5)  # D^{-0.5}
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[row] * edge_weight * deg_inv[col]  # D^{-0.5}

        return norm