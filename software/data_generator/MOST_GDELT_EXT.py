import sys
import pathlib
import random
import pickle
import pandas as pd
import numpy as np
import datetime
from collections import Counter, defaultdict
import os
import json
import pandas as pd

os.chdir(pathlib.Path(os.path.abspath(__file__)).parents[0])
sys.path.append(os.getcwd())

random.seed(2022)
class PATHS():
    DATA_DIR = './'
    DATASET_DIR = DATA_DIR + '%s_%s/'

    EMB_PATH = DATASET_DIR + 'embeddings/'
    BASELINES_DIR = DATASET_DIR + 'baselines/'
    HIST_DIR = DATASET_DIR + 'hist_%d_%d/'
    FEWSHOT_QUADS = 'fewshot.txt'


class DataProcessor():
    def __init__(self, dataset_name, mode=''):
        self.dataset = dataset_name
        self.mode = mode
        self._path = PATHS.DATASET_DIR % (dataset_name, mode)
        self._oripath = PATHS.DATA_DIR
        self.event_data = self.read_data()

        self.SOURCE_CLMN = 'Sub'
        self.TARGET_CLMN = 'Obj'

    def make_data(self):
        print("MAKING REL2QUADRUPLE")
        gdelt_data = []
        all_ent = set()
        all_rel = set()
        all_ts = set()

        with open(self._path + 'train.txt', 'r') as f:
            for q in f.readlines():
                quadruple = q.replace(' ', '').rstrip().split('\t')[:-1]
                gdelt_data.append(quadruple)
                s,r,o,t = quadruple
                all_ent.add(s)
                all_ent.add(o)
                all_rel.add(r)
                all_ts.add(int(t))

        with open(self._path + 'valid.txt', 'r') as f:
            for q in f.readlines():
                quadruple = q.replace(' ', '').rstrip().split('\t')[:-1]
                gdelt_data.append(quadruple)
                s, r, o, t = quadruple
                all_ent.add(s)
                all_ent.add(o)
                all_rel.add(r)
                all_ts.add(int(t))

        with open(self._path + 'test.txt', 'r') as f:
            for q in f.readlines():
                quadruple = q.replace(' ', '').rstrip().split('\t')[:-1]
                gdelt_data.append(quadruple)
                s, r, o, t = quadruple
                all_ent.add(s)
                all_ent.add(o)
                all_rel.add(r)
                all_ts.add(int(t))

        strid2ent_ori = {}
        with open(self._path + 'entity2id.txt','r') as f:
            for ent2id in f.readlines():
                ent2id_list = ent2id.rstrip().split('\t')
                str_id = ent2id_list[1]
                str_ent = ent2id_list[0]
                strid2ent_ori.update({str_id:str_ent})

        strid2rel = {}
        with open(self._path + 'relation2id.txt', 'r') as f:
            for rel2id in f.readlines():
                rel2id_list = rel2id.rstrip().split('\t')
                str_id = int(rel2id_list[1])
                str_ent = rel2id_list[0]
                strid2rel.update({str_id: str_ent})

        all_ts = sorted(all_ts)
        id2ts = {id:ts for id,ts in enumerate(all_ts)}

        gdelt_rel2quadruple = defaultdict(list)
        gdelt_data_str = []
        all_ent_str = set()
        all_rel_str = set()

        for qd in gdelt_data:
            s_or,r_or,o_or,t_or = qd
            str_s = strid2ent_ori[s_or]
            str_o = strid2ent_ori[o_or]
            str_t = t_or
            str_r = strid2rel[int(r_or)]
            all_ent_str.add(str_s)
            all_ent_str.add(str_o)
            all_rel_str.add(str_r)
            gdelt_data_str.append([str_s, str_r, str_o, str_t])
            gdelt_rel2quadruple[str_r].append([str_s,str_r,str_o,str_t])

        gdelt_ent2id_noinv = {key: idx for idx, key in enumerate(sorted(all_ent_str))}
        gdelt_rel2id_noinv = {key: idx for idx, key in enumerate(sorted(all_rel_str))}
        gdelt_ts2id_noinv = {key: idx for idx, key in enumerate(sorted(all_ts))}

        with open(self._path + 'rel2quadruple.pkl', 'wb+') as r2qf:
            pickle.dump(gdelt_rel2quadruple, r2qf)
        with open(self._path + 'gdelt_data_str.json', 'w+') as gdsf:
            json.dump(gdelt_data_str, gdsf)

        with open(self._path+'ent2id_noinv.json', 'w+') as ef:
            json.dump(gdelt_ent2id_noinv, ef)
        with open(self._path +'rel2id_noinv.json', 'w+') as rf:
            json.dump(gdelt_rel2id_noinv, rf)
        with open(self._path +'ts2id_noinv.json', 'w+') as tf:
            json.dump(gdelt_ts2id_noinv, tf)

    def read_data(self):
        try:
            with open(self._path + 'rel2quadruple.pkl', 'rb') as f:
                self.rel2quadruple = pickle.load(f)
            with open(self._path + 'rel2id_noinv.json', 'rb') as f:
                self.rel2id_noinv = json.load(f)
            with open(self._path + 'ent2id_noinv.json', 'rb') as f:
                self.ent2id_noinv = json.load(f)
            with open(self._path + 'ts2id_noinv.json', 'rb') as f:
                self.ts2id_noinv = json.load(f)
            print("LOADING REL2QUADRUPLE")
        except FileNotFoundError :
            print("REL2QUADRUPLE NOT FOUND")
            # self.make_data()
            with open(self._path + 'rel2quadruple.pkl', 'rb') as f:
                self.rel2quadruple = pickle.load(f)
            with open(self._path + 'rel2id_noinv.json', 'rb') as f:
                self.rel2id_noinv = json.load(f)
            with open(self._path + 'ent2id_noinv.json', 'rb') as f:
                self.ent2id_noinv = json.load(f)
            with open(self._path + 'ts2id_noinv.json', 'rb') as f:
                self.ts2id_noinv = json.load(f)

        event_data_list = []
        for rel in self.rel2quadruple.keys():
            for q in self.rel2quadruple[rel]:
                event_data_list.append(q)


        event_data = pd.DataFrame(event_data_list)

        event_data = event_data.rename(columns={event_data.columns[0]: 'Sub',
                                                event_data.columns[1]: 'Cameo Code',
                                                event_data.columns[2]: 'Obj',
                                                event_data.columns[3]: 'Event Date'})
        subfilter = event_data['Sub'].isin(self.ent2id_noinv.keys())
        objfilter = event_data['Obj'].isin(self.ent2id_noinv.keys())
        relfilter = event_data['Cameo Code'].isin(self.rel2id_noinv.keys())
        tsfilter = event_data['Event Date'].isin(self.ts2id_noinv.keys())
        event_data = event_data[subfilter & objfilter & relfilter & tsfilter]

        event_data = event_data.astype({'Event Date':"int64"})
        event_data.sort_values(by=['Event Date'], inplace=True)

        return event_data

    def select_relations(self, low_thresh, high_thresh):
        relations = self.event_data['Cameo Code'].values
        counted = Counter(relations)
        self.meta_rels = [x for x, y in counted.items() if low_thresh <= y <= high_thresh]
        self.background_rels = [x for x, y in counted.items() if y > high_thresh]
        print(f"NUMBER OF SPARSE RELATION: {len(self.meta_rels)}")
        print(f"NUMBER OF BACKGROUND RELATION: {len(self.background_rels)}")
        # select train, test, val relations
        self.val_rels = random.sample(self.meta_rels, int(len(self.meta_rels)*(6/(30+6+11))))
        self.test_rels = random.sample([x for x in self.meta_rels if x not in self.val_rels], int(len(self.meta_rels)*(11/(30+6+11))))
        self.train_rels = [x for x in self.meta_rels if x not in (self.val_rels + self.test_rels)]
        print(f"NUMBER OF SPARSE TRAIN: {len(self.train_rels)}")
        print(f"NUMBER OF SPARSE VALID: {len(self.val_rels)}")
        print(f"NUMBER OF SPARSE TEST: {len(self.test_rels)}")

    @staticmethod
    def get_dict(keys, dct1, dct2):
        output1 = {}
        output2 = {}
        for key in keys:
            try:
                output1[key] = dct1[key]
                output2[key] = dct2[key]
            except KeyError:
                print(key)
        return {'tasks': output1, 'nexts': output2}

    def create_write_symbol_to_id(self, df):
        self.id2dt = {}
        self.id2ent = {}
        self.id2rel_inv = {}
        self.rel2id_inv = {}
        self.num_rel = len(self.rel2id_noinv)
        self.ts2id = {}
        for ts, id in self.ts2id_noinv.items():
            self.ts2id.update({ts: id})
            self.id2dt.update({id: ts})
        for ent, id in self.ent2id_noinv.items():
            self.id2ent.update({id: ent})
        for rel, id in self.rel2id_noinv.items():
            self.id2rel_inv.update({id: rel})
            self.id2rel_inv.update({id + self.num_rel: rel + '_inverse'})

            self.rel2id_inv.update({rel: id})
            self.rel2id_inv.update({rel + '_inverse': id + self.num_rel})

        self.symbol2id = {
            'dt2id': self.ts2id,
            'ent2id': self.ent2id_noinv,
            'rel2id': self.rel2id_inv}

        self.id2symbol = {
            'id2dt': self.id2dt,
            'id2ent': self.id2ent,
            'id2rel': self.id2rel_inv
        }


    def prediction_create_write_task_pools(self, split_test_date, split_val_date):
        task_pools = defaultdict(list)
        time_pools = defaultdict(list)

        meta_train = set()
        meta_test = set()
        meta_val = set()

        quads = []
        relation_count = defaultdict(int)
        meta_quads_str_no_inv = []
        with open(self._path + PATHS.FEWSHOT_QUADS, 'w') as fp:
            i = 0
            for idx, row in self.meta_icews.iterrows():
                s = row[self.SOURCE_CLMN]
                o = row[self.TARGET_CLMN]
                r = row['Cameo Code']
                t = row['Event Date']


                s_id = self.symbol2id['ent2id'][s]
                o_id = self.symbol2id['ent2id'][o]
                r_id = self.symbol2id['rel2id'][r]
                t_id = self.symbol2id['dt2id'][str(t)]

                if r in self.train_rels:
                    if t < split_val_date:
                        quads.append((s_id, r_id, o_id, t_id))
                        quads.append((o_id, r_id + self.num_rel, s_id, t_id))
                        meta_quads_str_no_inv.append([s,r,o,t])
                        fp.write("%d\t%d\t%d\t%d\n" % (s_id, r_id, o_id, t_id))
                        fp.write("%d\t%d\t%d\t%d\n" % (o_id, r_id + self.num_rel, s_id, t_id))
                        task_pools[r_id].append(i)
                        task_pools[r_id + self.num_rel].append(i + 1)
                        meta_train.add(r_id)
                        meta_train.add(r_id + self.num_rel)
                        i += 2
                        relation_count[r] +=1

                elif r in self.val_rels:
                    # print(t, val_date <= t < test_date)
                    if split_val_date <= t < split_test_date:
                        meta_quads_str_no_inv.append([s, r, o, t])
                        quads.append((s_id, r_id, o_id, t_id))
                        quads.append((o_id, r_id + self.num_rel, s_id, t_id))
                        fp.write("%d\t%d\t%d\t%d\n" % (s_id, r_id, o_id, t_id))
                        fp.write("%d\t%d\t%d\t%d\n" % (o_id, r_id + self.num_rel, s_id, t_id))

                        task_pools[r_id].append(i)
                        task_pools[r_id + self.num_rel].append(i + 1)
                        meta_val.add(r_id)
                        meta_val.add(self.num_rel + r_id)
                        i += 2
                        # print('valid')
                        relation_count[r] += 1
                elif r in self.test_rels:
                    if split_test_date <= t:
                        meta_quads_str_no_inv.append([s, r, o, t])
                        quads.append((s_id, r_id, o_id, t_id))
                        quads.append((o_id, r_id + self.num_rel, s_id, t_id))
                        fp.write("%d\t%d\t%d\t%d\n" % (s_id, r_id, o_id, t_id))
                        fp.write("%d\t%d\t%d\t%d\n" % (o_id, r_id + self.num_rel, s_id, t_id))
                        task_pools[r_id].append(i)
                        task_pools[r_id + self.num_rel].append(i + 1)
                        meta_test.add(r_id)
                        meta_test.add(r_id + self.num_rel)
                        i += 2
                        relation_count[r] += 1

                else:
                    raise ValueError


        ent_set = set()
        filtered_relation_count = {}
        filtered_relation = {}
        for key,count in relation_count.items():
            if count >= 100:
                filtered_relation_count[key] = count
            else:
                filtered_relation[key] = count

        sparse_rel_train = [rel for rel in self.train_rels if rel in filtered_relation_count.keys()]
        sparse_rel_valid = [rel for rel in self.val_rels if rel in filtered_relation_count.keys()]
        sparse_rel_test = [rel for rel in self.test_rels if rel in filtered_relation_count.keys()]
        assert (len(sparse_rel_test) + len(sparse_rel_train) + len(sparse_rel_valid)) == len(
            filtered_relation_count.keys())

        print(f"Number of all sparse relations {len(self.meta_rels)}")
        print(f"Number of relation before filter: {len(relation_count)}")
        print(f"Number of relation after filter: {len(filtered_relation_count)}")
        print(f"Filtered relations are: {list(filtered_relation.keys())}")
        print(f"Number of background relations: {len(self.background_rels)}")
        print(f'valid set:{len(self.val_rels)}, {len(self.val_rels) - len(sparse_rel_valid)} valid relations are filtered, {len(sparse_rel_valid)} remains')
        print(f'test set:{len(self.test_rels)}, {len(self.test_rels) - len(sparse_rel_test)} test relations are filtered, {len(sparse_rel_test)} remains')
        print(f'train set:{len(self.train_rels)}, {len(self.train_rels) - len(sparse_rel_train)} train relations are filtered, {len(sparse_rel_train)} remains')


        rel2quadruple = defaultdict(list)
        for q in meta_quads_str_no_inv:
            s,r,o,t = q
            if r in filtered_relation_count.keys():
                ent_set.add(s)
                ent_set.add(o)
                rel2quadruple[r].append([s,r,o,str(t)])

        for idex, row in self.background_icews.iterrows():
            s = row['Sub']
            o = row['Obj']
            r = row['Cameo Code']
            t = row['Event Date']
            ent_set.update(s)
            ent_set.update(o)
            rel2quadruple[r].append([s, r, o, str(t)])

        sparse_rel = list(filtered_relation_count.keys())

        data_path = './GDELT_EXTRAPOLATION/'
        pathlib.Path(data_path).mkdir(parents=True,exist_ok=True)
        with open(f'{data_path}sparse_rel.pkl', 'wb+') as f:
            pickle.dump(sparse_rel,f)
        with open(f'{data_path}background_rel.pkl', 'wb+') as f:
            pickle.dump(self.background_rels, f)
        with open(f'{data_path}rel2quadruple.pkl', 'wb+') as f:
            pickle.dump(rel2quadruple, f)
        with open(f'{data_path}sparse_rel_train.pkl', 'wb+') as f:
            pickle.dump(sparse_rel_train, f)
        with open(f'{data_path}sparse_rel_valid.pkl', 'wb+') as f:
            pickle.dump(sparse_rel_valid, f)
        with open(f'{data_path}sparse_rel_test.pkl', 'wb+') as f:
            pickle.dump(sparse_rel_test, f)


        ent_set = set()
        rel_set = set()
        ts_set = set()
        ts_set_int = set()
        for rel in rel2quadruple.keys():
            for q in rel2quadruple[rel]:
                s,r,o,t = q
                ent_set.add(s)
                ent_set.add(o)
                rel_set.add(r)
                ts_set.add(str(t))
                ts_set_int.add(int(t))
        ts_set_int = sorted(ts_set_int)

        filtered_ent2id_noinv = {}
        for id, ent in enumerate(ent_set):
            filtered_ent2id_noinv.update({ent: id})

        filtered_rel2id_noinv = {}
        for id, rel in enumerate(rel_set):
            filtered_rel2id_noinv.update({rel: id})
        filtered_ts2id_noinv = {}
        for id, ts in enumerate(ts_set_int):
            filtered_ts2id_noinv.update({str(ts): id})

        with open(f'{data_path}rel2id_noinv.pkl', 'wb+') as f:
            pickle.dump(filtered_rel2id_noinv, f)
        with open(f'{data_path}ent2id_noinv.pkl', 'wb+') as f:
            pickle.dump(filtered_ent2id_noinv, f)
        with open(f'{data_path}ts2id_noinv.pkl', 'wb+') as f:
            pickle.dump(filtered_ts2id_noinv, f)
        with open(f'{data_path}stat.txt', 'w+') as f:
            f.write(f'{len(filtered_ent2id_noinv)}\t{len(filtered_rel2id_noinv)}')


        filtered_ent2id_noinv = {}
        i = 0
        for ent in sorted(ent_set):
            filtered_ent2id_noinv.update({ent: i})
            i += 1
        with open(f'{data_path}ent2id_noinv.json', 'w+') as ef:
            json.dump(filtered_ent2id_noinv, ef)

        filtered_rel2id_noinv = {}
        i = 0
        for rel in sorted(rel_set):
            filtered_rel2id_noinv.update({rel: i})
            i += 1
        with open(f'{data_path}rel2id_noinv.json', 'w+') as rf:
            json.dump(filtered_rel2id_noinv, rf)

        filtered_ts2id_noinv = {}
        i = 0
        for t in sorted(ts_set_int):
            filtered_ts2id_noinv.update({str(t): i})
            i += 1
        with open(f'{data_path}ts2id_noinv.json', 'w+') as tf:
            json.dump(filtered_ts2id_noinv, tf)


    def split_train_test_val(self, split_test_date, split_val_date):
        filtered_icews = self.event_data

        time_filter = filtered_icews['Event Date'] < split_val_date
        background_rel_filter = filtered_icews['Cameo Code'].isin(self.background_rels)
        self.background_icews_before_validts = filtered_icews[time_filter & background_rel_filter]

        ent_set = list(self.background_icews_before_validts['Sub'].values) + list(
            self.background_icews_before_validts['Obj'].values)
        print(f'Number of ent in background before {split_val_date} is: {len(set(ent_set))}')
        entity_filter = filtered_icews['Sub'].isin(ent_set) & filtered_icews['Obj'].isin(ent_set)

        self.background_icews = filtered_icews[background_rel_filter & entity_filter]
        self.background_icews = self.background_icews[
            [self.SOURCE_CLMN, self.TARGET_CLMN, 'Event Date', 'Cameo Code']].drop_duplicates()  # pretrain.csv

        # Make background and meta dataset
        # entity_filter = filtered_icews['Sub'].isin(ent_set) & filtered_icews['Obj'].isin(ent_set)
        sparse_rel_filter = filtered_icews['Cameo Code'].isin(self.meta_rels)
        self.ori_meta_icews = filtered_icews[sparse_rel_filter]
        self.meta_icews = filtered_icews[sparse_rel_filter & entity_filter]
        self.meta_icews = self.meta_icews[
            [self.SOURCE_CLMN, self.TARGET_CLMN, 'Event Date', 'Cameo Code']].drop_duplicates()
        print(f"Number of quads in meta: {len(self.ori_meta_icews)}, after filter: {len(self.meta_icews)}")

        # Make symb2id and save the filtered dataset with ids
        self.create_write_symbol_to_id(filtered_icews)  # already saved data2id.csv(splite by 2011,2013)
        self.prediction_create_write_task_pools(split_test_date=split_test_date, split_val_date=split_val_date)
if __name__ == "__main__":
    PATHS = PATHS()
    lth = 200
    hth = 2000

    split_test = 35700
    split_val = 26775
    dp = DataProcessor(dataset_name='GDELT', mode='INTERPOLATION')
    dp.select_relations(lth, hth)
    dp.split_train_test_val(split_test_date=split_test, split_val_date=split_val)
