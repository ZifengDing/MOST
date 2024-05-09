import torch.nn
from .compgcn_conv import TARGEBLOCKLAYER

class TARGELayerWrapper(torch.nn.Module):
    def __init__(self, edge_index, edge_type, num_e, num_rel, act, embsize, drop1, drop2, sub, rel, params=None,
                 time_encoder=None):
        super().__init__()
        self.nfe = 0
        self.p = params
        self.time_encoder = time_encoder
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.embsize = embsize
        self.num_e = num_e
        self.num_rel = num_rel
        self.device = self.p.device
        self.act = act
        self.drop_l1 = torch.nn.Dropout(drop1)
        self.drop_l2 = torch.nn.Dropout(drop2)
        self.sub = sub
        self.rel = rel

        # res
        if self.p.res:
            self.res = torch.nn.Parameter(torch.FloatTensor([0.1]))


        self.conv1 = TARGEBLOCKLAYER(self.p, self.embsize, self.embsize, self.num_rel, num_bases=20,
                                           bias=self.p.bias,
                                           activation=self.act, dropout=self.p.dropout)
        self.conv2 = TARGEBLOCKLAYER(self.p, self.embsize, self.embsize, self.num_rel,
                                           num_bases=20, bias=self.p.bias, activation=self.act,
                                           dropout=self.p.dropout) if self.p.core_layer == 2 else None

        self.register_parameter('bias', torch.nn.Parameter(torch.zeros(num_e)))


    def set_graph(self, edge_index, edge_type, edge_ts):
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.edge_ts = edge_ts


    def forward(self, emb):

        ts_emb = self.time_encoder(self.edge_ts)

        emb = emb + self.res * self.conv1(emb, self.edge_index, self.edge_type, ts_emb)
        emb = self.drop_l1(emb)
        emb = emb + self.res * self.conv2(emb, self.edge_index, self.edge_type,
                                          ts_emb) if self.p.core_layer == 2 else emb
        emb = self.drop_l2(emb) if self.p.core_layer == 2 else emb

        return emb