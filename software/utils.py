import time
import pickle
import json
from functools import wraps
from itertools import zip_longest
from collections import defaultdict
import logging
import os
import sys
import random

DataDir = os.path.join(os.path.dirname(__file__), 'dataset')

def get_logger(name: str):
    cur_dir = os.getcwd()
    if not os.path.exists(cur_dir + '/log/'):
        os.mkdir(cur_dir + '/log/')
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M',
        filename=cur_dir + '/log/' + name + '.log',
        filemode='a')
    logger = logging.getLogger(name)
    stdout_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stdout_handler)
    return logger

def grouper(iterable, n, fillvalue=None):
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)
def timethis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        r = func(*args, **kwargs)
        end = time.perf_counter()
        print('{}.{} : {}'.format(func.__module__, func.__name__, end - start))
        return r

    return wrapper


class MOSTData():
    def __init__(self, dataset=None, shuffle_first=True):
        self.data_path = data_path = DataDir + '/' + dataset + '/'

        if 'extrapolation' in dataset.lower():
            print('MOST_EXTRAPOLATION')
            self.mode = 'ext'
        elif 'interpolation' in dataset.lower():
            print('MOST_INTERPOLATION')
            self.mode = 'int'
        else:
            raise ValueError
        if 'icews' in dataset.lower():
            print('DATASET IS ICEWS0515')
        elif 'gdelt' in dataset.lower():
            print('DATASET IS GDELT')
        else:
            raise ValueError

        with open(f'{data_path}sparse_rel.pkl', 'rb') as f:
            self.sparse_rel = pickle.load(f)
        with open(f'{data_path}background_rel.pkl', 'rb') as f:
            self.background_rel = pickle.load(f)
        with open(f'{data_path}rel2quadruple.pkl', 'rb') as f:
            self.rel2quadruple = pickle.load(f)
        with open(f'{data_path}sparse_rel_train.pkl', 'rb') as f:
            self.sparse_rel_train = pickle.load(f)
        with open(f'{data_path}sparse_rel_valid.pkl', 'rb') as f:
            self.sparse_rel_valid = pickle.load(f)
        with open(f'{data_path}sparse_rel_test.pkl', 'rb') as f:
            self.sparse_rel_test = pickle.load(f)

        with open(f'{data_path}rel2id_noinv.json', 'rb') as f:
            self.rel2id_noinv = json.load(f)
        with open(f'{data_path}ent2id_noinv.json', 'rb') as f:
            self.ent2id_noinv = json.load(f)
        with open(f'{data_path}ts2id_noinv.json', 'rb') as f:
            self.ts2id_noinv = json.load(f)

        num_ent, num_rel = self.get_total_number(f'{data_path}stat.txt')

        self.background_rel = self.convert_list2id(self.background_rel, self.rel2id_noinv)
        self.sparse_rel = self.convert_list2id(self.sparse_rel, self.rel2id_noinv)
        self.sparse_rel_train = self.convert_list2id(self.sparse_rel_train, self.rel2id_noinv)
        self.sparse_rel_valid = self.convert_list2id(self.sparse_rel_valid, self.rel2id_noinv)
        self.sparse_rel_test = self.convert_list2id(self.sparse_rel_test, self.rel2id_noinv)
        self.rel2quadruple = self.convert_q2id(self.rel2quadruple, self.ent2id_noinv, self.rel2id_noinv,
                                               self.ts2id_noinv)
        self.id2ts = self.get_id2ts(self.ts2id_noinv)
        self.num_rel = len(self.rel2quadruple.keys())

        self.rel2quadruple = self.add_reverse(self.rel2quadruple)

        if shuffle_first:
            try:
                with open(f'{data_path}rel2quadruple_shuffle.pkl', 'rb') as f:
                    self.rel2quadruple = pickle.load(f)
                    print('LOADING SHUFFLED REL2QUADRUPLE')
            except FileNotFoundError:
                print('NO SHUFFLED REL2QUADRUPLE FOUND, CREATING A NEW ONE')
                for rel in self.rel2quadruple.keys():
                    random.shuffle(self.rel2quadruple[rel])
                with open(f'{data_path}rel2quadruple_shuffle.pkl', 'wb') as f:
                    pickle.dump(self.rel2quadruple, f)

        self.background_rel = self.add_reverse(self.background_rel)
        self.sparse_rel = self.add_reverse(self.sparse_rel)
        self.sparse_rel_train = self.add_reverse(self.sparse_rel_train)
        self.sparse_rel_valid = self.add_reverse(self.sparse_rel_valid)
        self.sparse_rel_test = self.add_reverse(self.sparse_rel_test)

        self.id2rel_inv = {}
        for key, value in self.rel2id_noinv.items():
            self.id2rel_inv[value] = key
            self.id2rel_inv[value + self.num_rel] = key + '_inv'

        self.num_rel *= 2

        entity_set = set()
        for rel in self.rel2quadruple.keys():
            for q in self.rel2quadruple[rel]:
                entity_set.update([q[0]])
                entity_set.update([q[2]])

        # check after add inverse
        assert len(entity_set) == len(self.ent2id_noinv) == num_ent
        assert self.num_rel == len(self.rel2quadruple.keys()) == num_rel * 2

        try:
            self.o2srt = pickle.load(open(self.data_path + 'o2srt.pkl', 'rb'))
            self.so2t2r = pickle.load(open(self.data_path + 'so2t2r.pkl', 'rb'))
            self.srt2o = pickle.load(open(self.data_path + 'srt2o.pkl', 'rb'))
            self.sr2o = pickle.load(open(self.data_path + 'sr2o.pkl', 'rb'))
        except:
            self.so2t2r = self._construct_so2t2r(self.rel2quadruple)
            self.o2srt = self._construct_o2srt(self.rel2quadruple)
            self.sr2o, self.srt2o = self._construct_srt2o(self.rel2quadruple)

            self.save_data(self.data_path)

    def get_total_number(self,path):
        with open(path, 'r') as file:
            for line in file:
                line_split = line.split()
                return int(line_split[0]), int(line_split[1])

    def get_data(self, rel, few, batch_size, is_candidate=False):
        train_and_test = self.rel2quadruple[rel]
        random.shuffle(train_and_test)
        support_quadruples = train_and_test[:few]
        support_pairs = [[q[0], q[2], q[3]] for q in support_quadruples]
        query_quadruples = train_and_test[few:]
        query_pairs = [[q[0], q[2], q[3]] for q in query_quadruples]
        # use all the rest quadruples as query in training
        if len(query_quadruples) == 0:
            return [], []

        if len(query_pairs) < batch_size:
            query_pairs = [random.choice(query_pairs) for _ in range(batch_size)]
        else:
            query_pairs = random.sample(query_pairs, batch_size)

        false_pairs = []
        for triple in query_pairs:
            sub = triple[0]
            obj = triple[1]
            ts = triple[2]
            while True:
                if is_candidate:
                    noise = random.choice(self.candidate[rel])
                else:
                    noise = random.choice(list(self.ent2id_noinv.values()))
                if (noise not in self.sr2o[(sub, rel)]) and noise != obj:
                    break
            false_pairs.append([sub, noise, ts])

        return support_pairs, query_pairs, false_pairs

    def convert_list2id(self, data: list, file2id):
        id = []
        for i, value in enumerate(data):
            id.append(file2id[value])
        return id

        return data

    def convert_q2id(self, data, ent2id, rel2id, ts2id):
        id = defaultdict(list)
        for i, key in enumerate(data.keys()):
            if key not in rel2id.keys():
                continue
            else:
                relid = rel2id[key]
                for q in data[key]:
                    if (q[0] not in ent2id.keys() or q[2] not in ent2id.keys()):
                        continue
                    else:
                        q[0] = ent2id[q[0]]
                        q[1] = rel2id[q[1]]
                        q[2] = ent2id[q[2]]
                        q[3] = ts2id[q[3]]
                        if relid not in id.keys():
                            id.update({relid: [q]})
                        else:
                            id[relid].append(q)
        return id

    def add_reverse(self, data):
        if type(data) == list:
            new_data = [f(r) for r in data for f in (lambda x: x, lambda y: y + self.num_rel)]
            return new_data
        else:
            r2q = defaultdict(list)
            num_rel = len(self.rel2quadruple.keys())
            for rel in self.rel2quadruple.keys():
                new_rel = rel + num_rel
                for q in self.rel2quadruple[rel]:
                    reverse_q = [q[2], q[1] + num_rel, q[0], q[3]]
                    if rel not in r2q.keys():
                        r2q.update({rel: [q]})
                    else:
                        r2q[rel].append(q)
                    if new_rel not in r2q.keys():
                        r2q.update({new_rel: [reverse_q]})
                    else:
                        r2q[new_rel].append(reverse_q)

            return r2q

    def save_data(self, path):
        with open(path + 'o2srt.pkl', 'wb') as f:
            pickle.dump(self.o2srt, f)
        with open(path + 'so2t2r.pkl', 'wb') as f:
            pickle.dump(self.so2t2r, f)
        with open(path + 'srt2o.pkl', 'wb') as f:
            pickle.dump(self.srt2o, f)
        with open(path + 'sr2o.pkl', 'wb') as f:
            pickle.dump(self.sr2o, f)

    def candidate_filter(self, candidate):
        id_candidate = defaultdict(list)
        for rel in self.candidate.keys():
            if rel not in self.rel2id_noinv.keys():
                continue
            for e in self.candidate[rel]:
                if e not in self.ent2id_noinv.keys():
                    continue
                else:
                    rel_id = self.rel2id_noinv[rel]
                    e_id = self.ent2id_noinv[e]
                    if rel_id in id_candidate.keys():
                        id_candidate[rel_id].append(e_id)
                    else:
                        id_candidate.update({rel_id: [e_id]})
        return id_candidate

    def get_id2ts(self, ts2id):
        id2ts = {}
        for (key, value) in ts2id.items():
            id2ts.update({int(value): key})
        return id2ts

    def _construct_srt2o(self, rel2quadruple):
        srt2o = defaultdict(list)
        sr2o = defaultdict(list)
        for rel in rel2quadruple.keys():
            for q in rel2quadruple[rel]:
                # srt2o
                if (q[0], q[1], q[3]) not in srt2o.keys():
                    srt2o.update({(q[0], q[1], q[3]): [q[2]]})
                else:
                    srt2o[(q[0], q[1], q[3])].append(q[2])
                # sr2o
                if (q[0], q[1]) not in sr2o.keys():
                    sr2o.update({(q[0], q[1]): [q[2]]})
                else:
                    sr2o[(q[0], q[1])].append(q[2])
        return sr2o, srt2o

    def _construct_o2srt(self, rel2quadruple):
        o2srt = defaultdict(list)

        for rel in self.background_rel:
            for q in rel2quadruple[rel]:
                if q[2] not in o2srt.keys():
                    o2srt.update({q[2]: [[q[0], q[1], q[3]]]})
                else:
                    o2srt[q[2]].append([q[0], q[1], q[3]])

        return o2srt

    def _construct_so2t2r(self, rel2quadruple):
        so2t2r = defaultdict(dict)
        for rel in self.background_rel:
            for q in rel2quadruple[rel]:
                if q[3] in so2t2r[(q[0], q[2])].keys():
                    so2t2r[(q[0], q[2])][q[3]].append(q[1])
                else:
                    so2t2r[(q[0], q[2])].update({q[3]: [q[1]]})
        return so2t2r
