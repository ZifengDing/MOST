import torch.nn as nn
from models.compgcn_conv import *
from models.models import *
# from utils import grouper, timethis


class MOST(torch.nn.Module):
    def __init__(self, matcher, num_ent, num_rel, id2ts, o2srt, args):
        super().__init__()
        self.dataset = args.dataset
        self.matcher = matcher
        self.time_encoder = self.matcher.time_encoder
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.id2ts = id2ts
        self.is_rawts = args.is_rawts
        self.device = args.device
        self.embsize = args.embsize
        self.raw_embsize = int(args.embsize * args.rawportion)
        self.t_embsize = args.embsize - self.raw_embsize
        self.emb_e = self.get_param((self.num_ent, self.raw_embsize))
        self.emb_r = self.get_param((self.num_rel, self.embsize))
        self.act = args.activation
        self.drop = args.dropout
        self.loss = torch.nn.CrossEntropyLoss()
        self.o2srt = o2srt
        self.num_sample = args.num_sample
        self.max_support_ts_diff = 0

        # activation
        if args.activation.lower() == 'tanh':
            self.act = torch.tanh
        elif args.activation.lower() == 'relu':
            self.act = torch.nn.functional.relu
        elif args.activation.lower() == 'leakyrelu':
            self.act = torch.nn.functional.leaky_relu

        # Time-aware Relational Graph Encoder
        self.TARGE = TARGELayerWrapper(None, None, self.num_ent, self.num_rel, self.act, self.raw_embsize,
                                     drop1=self.drop, drop2=self.drop, sub=None, rel=None, params=args,
                                     time_encoder=self.time_encoder)
        self.TARGE.to(self.device)

    def get_param(self, shape):
        """a function to initialize embed"""
        param = torch.nn.Parameter(torch.empty(shape, requires_grad=True, device=self.device))
        nn.init.xavier_normal_(param.data)
        return param

    def o2srt_filter(self, pairs, filter_ts, o2srt, warm_up=0, ts_as_dis=False, ts_dis=None, query_ts=None):
        max_t = filter_ts
        max_t = max(warm_up, max_t)
        pairs = torch.tensor(pairs, device=self.device)
        # only support_pairs
        all_obj = set(torch.vstack([pairs])[:, [0, 1]].flatten().cpu().numpy())
        # copy to avoid inplace change
        self.o2srt = o2srt
        for obj in all_obj:
            obj_all_srt = o2srt[obj]
            if ts_as_dis:
                filter_obj_srt = [srt for srt in obj_all_srt if
                                  ((max(0, max_t - ts_dis) <= srt[2]) and (srt[2] < min(query_ts, max_t + ts_dis)))]
            else:
                filter_obj_srt = [srt for srt in obj_all_srt if srt[2] < max_t]
            self.o2srt[obj] = filter_obj_srt

    def o2srt_filter_until_test(self, pairs, last_valid_time, o2srt):
        pairs = torch.tensor(pairs, device=self.device)
        # only support_pairs
        all_obj = set(torch.vstack([pairs])[:, [0, 1]].flatten().cpu().numpy())
        # copy to avoid inplace change
        self.o2srt = o2srt
        for obj in all_obj:
            obj_all_srt = o2srt[obj]
            filter_obj_srt = [srt for srt in obj_all_srt if srt[2] < last_valid_time]
            self.o2srt[obj] = filter_obj_srt

    def forward(self, one_pair, is_random=None):

        sub = one_pair[:, 0]
        obj = one_pair[:, 1]
        ts = one_pair[:, 2]
        sub_srt = self.o2srt[int(sub)]
        obj_srt = self.o2srt[int(obj)]

        sub_edge_index, sub_edge_type, sub_edge_ts = self.sample_srt(sub_srt, sub, ts, num_sample=self.num_sample,
                                                                     is_random=is_random)
        obj_edge_index, obj_edge_type, obj_edge_ts = self.sample_srt(obj_srt, obj, ts, num_sample=self.num_sample,
                                                                     is_random=is_random)

        edge_index = torch.vstack((sub_edge_index, obj_edge_index))
        edge_type = torch.hstack((sub_edge_type, obj_edge_type))
        edge_ts = torch.hstack((sub_edge_ts, obj_edge_ts))

        self.TARGE.set_graph(edge_index.transpose(1, 0), edge_type, edge_ts)
        emb_e_agg = self.TARGE(torch.cat((self.emb_e, self.emb_r), dim=0))
        sub_embedding = torch.index_select(emb_e_agg, 0, sub)
        obj_embedding = torch.index_select(emb_e_agg, 0, obj)

        # support and query emb
        emb = torch.cat((sub_embedding, obj_embedding), dim=-1)
        return emb


    def sample_srt(self, srts, obj, true_ts, num_sample=None, is_random=None):

        if len(srts) == 0:
            return torch.empty((0, 2), dtype=torch.int64, device=self.device), \
                   torch.empty((0), dtype=torch.int64, device=self.device), \
                   torch.empty((0), dtype=torch.int64, device=self.device)

        if len(srts) > num_sample:
            if is_random:
                selected_srts = random.sample(srts, num_sample)
                selected_srts = torch.tensor(selected_srts).to(self.device)
                if self.is_rawts:
                    ts_diff = selected_srts[:, 2]
                else:
                    ts_diff = abs(selected_srts[:, 2] - true_ts.to(self.device))
            else:
                srts = torch.tensor(srts).to(self.device)
                ts_diff = abs(srts[:, 2] - true_ts.to(self.device))
                sort_index = torch.argsort(ts_diff)[0:num_sample]
                selected_srts = srts[sort_index, :]
                if self.is_rawts:
                    ts_diff = srts[sort_index, 2]
                else:
                    ts_diff = ts_diff[sort_index]
                    if self.max_support_ts_diff < int(max(ts_diff)):
                        self.max_support_ts_diff = int(max(ts_diff))
        else:
            srts = torch.tensor(srts).to(self.device)
            selected_srts = torch.tensor(srts).to(self.device)
            if self.is_rawts:
                ts_diff = srts[:, 2]
            else:
                ts_diff = abs(selected_srts[:, 2] - true_ts.to(self.device))
                if self.max_support_ts_diff < int(max(ts_diff)):
                    self.max_support_ts_diff = int(max(ts_diff))
        edge_index = torch.cat((selected_srts[:, :1], torch.tensor([obj] * len(selected_srts)).unsqueeze(-1).to(
            self.device)), dim=1)
        edge_type = selected_srts[:, 1]
        edge_ts = ts_diff

        return edge_index, edge_type, edge_ts

    def loss_comp(self, score, obj):
        return self.loss(score, obj)


class Decoder(torch.nn.Module):
    def __init__(self, embsize_model, embsize_inner, dropout=0.2, loss_fun=None, time_encoder=None, precess_step=1, is_rawts=None,
                 rel_reg = None):
        super().__init__()
        self.rel_reg = rel_reg
        self.is_rawts = is_rawts
        self.info = None
        self.norm_info = None
        self.time_encoder = time_encoder
        self.node_proj = nn.Linear(2 * embsize_model, embsize_model)
        self.obj_node_proj = nn.Linear(2 * embsize_model, embsize_model)
        self.proj1 = nn.Linear(2 * embsize_model, 4 * embsize_inner)
        self.proj2 = nn.Linear(4 * embsize_inner, 2 * embsize_model)
        self.rel_proj = nn.Linear(2* embsize_model,embsize_model)
        self.rel_proj_rotate = nn.Linear(2 * embsize_model, int(embsize_model / 2))
        self.layer_norm = LayerNormalization(2 * embsize_model)
        self.layer_norm_node = LayerNormalization(embsize_model)
        self.loss_fun = loss_fun
        self.res = torch.nn.Parameter(torch.FloatTensor([0.1]))
        nn.init.xavier_normal_(self.proj1.weight)
        nn.init.xavier_normal_(self.proj2.weight)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, support_emb, support_pairs,query_pairs, emb_e):

        # 1. query
        sub_query_index = query_pairs[:, 0]
        ts_query_index = query_pairs[:, 2]
        sub_query_emb = torch.index_select(emb_e, 0, sub_query_index)
        # obj_query_emb will not be used
        # obj_query_index = query_pairs[:, 1]
        # obj_query_emb = torch.index_select(emb_e, 0, obj_query_index)

        # 1.1 prepare time aware emb
        # two different type of ts
        if self.is_rawts:
            # 1.1.1 prepare time aware: absolute time
            if self.info == None:
                print('ABS_TIME')
                self.info = 'ABS_TIME'
            ts_emb = self.time_encoder(ts_query_index)
            ts_emb_repeat = self.time_encoder(ts_query_index).repeat(emb_e.shape[0], 1)
        else:
            # 1.1.2 prepare time aware: time difference
            if self.info == None:
                print('ABS_TIME_DIFF')
                self.info = 'ABS_TIME_DIFF'
            ts_support_pairs = support_pairs[0][2]
            ts_diff = abs(ts_query_index - ts_support_pairs)
            ts_emb = self.time_encoder(ts_diff)
            ts_emb_repeat = self.time_encoder(ts_diff).repeat(emb_e.shape[0], 1)

        # 1.1.3 node projection for time aware embedding
        sub_query_out = self.res * self.node_proj(torch.hstack((sub_query_emb, ts_emb)))
        sub_query = self.layer_norm_node(sub_query_out + sub_query_emb)
        obj_query_out = self.res * self.obj_node_proj(torch.hstack((emb_e, ts_emb_repeat)))
        obj_query = self.layer_norm_node(obj_query_out + emb_e)

        # 1.2  prepare time unware query embedding
        # real raw_embedding:
        # if self.info == None:
        #     print('query using raw emb')
        #     self.info = 'query raw'
        # sub_query = sub_query_emb
        # obj_query = emb_e

        # 2. support as relation
        support_1 = self.relu(self.proj1(support_emb))
        support_out = self.dropout(self.proj2(support_1))
        # layer_norm
        support_out = self.layer_norm(support_out + support_emb)

        if self.loss_fun == 'bce':
            relation = self.dropout(self.relu(self.rel_proj_rotate(support_out)))
            pi = 3.14159265358979323846
            relation_alt = False
            if relation_alt:
                relation = self.dropout(self.relu(self.rel_proj_rotate(support_emb)))

            # if self.norm_info == None:
            #     print('USING RAW RELATION')
            #     self.norm_info = 'RAW'
            # phase_relation = relation

            if self.norm_info == None:
                print(f'USING MAX ABS RELATION, SCALE:{self.rel_reg}')
                self.norm_info = 'ABS'
            phase_relation = relation / (torch.max(torch.abs(relation)) * self.rel_reg / pi)

            # if self.norm_info ==None:
            #     print('USING NORM RELATION')
            #     self.norm_info = 'NORM'
            # phase_relation = relation / (torch.norm(relation) / pi)

            re_relation = torch.cos(phase_relation)
            im_relation = torch.sin(phase_relation)
            re_head, im_head = torch.chunk(sub_query, 2, dim=1)
            re_obj_emb = re_head * re_relation - im_head * im_relation
            im_obj_emb = re_head * im_relation + im_head * re_relation
            obj_emb = torch.hstack([re_obj_emb, im_obj_emb])
            score = torch.mm(obj_emb, obj_query.transpose(1, 0))
            score = torch.sigmoid(score)


        else:
            raise NotImplementedError

        return score

class LayerNormalization(torch.nn.Module):

    def __init__(self, d_hid, eps=1e-3):
        super().__init__()

        self.eps = torch.tensor(eps)
        self.a_2 = nn.parameter.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.parameter.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out_1 = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out_1 * self.a_2.expand_as(ln_out_1) + self.b_2.expand_as(ln_out_1)

        return ln_out

class TimeEncoder(torch.nn.Module):

    def __init__(self, time_dim, device):
        super(TimeEncoder, self).__init__()
        self.time_dim = time_dim
        self.device = device
        self.register_parameter(name='basis_freq', param=torch.nn.Parameter(
            torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim)).float(), requires_grad=True))
        self.register_parameter(name='phase', param=torch.nn.Parameter(torch.zeros(self.time_dim).float(),
                                                                       requires_grad=True))

    def forward(self, ts):
        ts = torch.unsqueeze(ts, dim=1)

        map_ts = ts * self.basis_freq.view(1, -1)  # [batch_size, 1, time_dim]
        map_ts += self.phase.view(1, -1)
        harmonic = torch.cos(map_ts)
        return harmonic
