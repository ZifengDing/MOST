import pickle
import random
import copy
import shutil
from collections import defaultdict as ddict
import json
import pathlib
import shutil

FOLDER_NAME = 'GDELT_INTERPOLATION'
pathlib.Path(f'./{FOLDER_NAME}').mkdir(parents=True,exist_ok=True)

entities, relations, ts, relation_sparse_final = set(), set(), set(), set()
entities_background = set()
ent2id_noinv, rel2id_noinv, ts2id_noinv = {}, {}, {}
rel_counter = ddict(int)
rel2quadruple = ddict(list)
rel2quadruple_final = ddict(list)
sparse_rel_time_span_dict = {}
sparse_rel = []
background_rel = []
upper = 1000 # max number of occurrence of sparse relation
lower = 100 # min number of occurrence of sparse relation
srt2o = ddict(set)
so2t2r = ddict(dict)
sot2r_back = ddict(set)
sot2r_sparse = ddict(set)
r2sot = ddict(list)
t2so2r_back = ddict(dict)
r2t2so = ddict(dict)
#max_time = 0

def get_total_number(path):
    with open(path, 'r') as file:
        for line in file:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])

def rel_count():
    quadrupleList = []
    max_time = 0
    data = json.load(open('quadruples_gdelt_noinv_name', ))
    for line in data:
        head = line[0]
        tail = line[2]
        rel = line[1]
        time = line[3]
        entities.add(head)
        entities.add(tail)
        ts.add(int(time))
        if max_time == 0:
            max_time = int(time)
        rel2quadruple[rel].append([head, rel, tail, time]) # no inverse relation!!!
        #TODO
        # rel2quadruple[rel+num_ent]([tail,rel+num_ent,head,time])
        #quadrupleList.append([head, rel, tail, time])
        rel_counter[rel] += 1
        # rel_counter[rel+num_ent] +=1
        if int(time) > max_time:
            max_time = int(time)
    return max_time

def classify_rel():
    for (rel, count) in rel_counter.items():
        if count > upper:
            background_rel.append(rel)
        elif count >= lower:
            sparse_rel.append(rel)

def split_one_shot_dataset():
    with open('background_graph_name.txt', 'w') as fp: # all background relations associated quadruples, without inverse relation!!!!!!
        for rel_b in background_rel:
            eol = len(rel2quadruple[rel_b])
            for i, q in enumerate(rel2quadruple[rel_b]):
                entities_background.add(q[0])
                entities_background.add(q[2])
                relations.add(q[1])
                fp.write(q[0] + '\t' + q[1] + '\t' + q[2] + '\t' + q[3] + '\n')
    # print(len(entities_background))
    # print(len(relations))
    # assert 0
    with open('fewshot.txt', 'w') as fs: # all fewshot relations associated quadruples, without inverse relation!!!!!! ALL entities appeared in background graph!!!!
        for rel_s in sparse_rel:
            eol = len(rel2quadruple[rel_s])
            for i, q in enumerate(rel2quadruple[rel_s]):
                if (q[0] not in entities_background) or (q[2] not in entities_background):
                    continue
                fs.write(q[0] + '\t' + q[1] + '\t' + q[2] + '\t' + q[3] + '\n')
                relations.add(q[1])
                relation_sparse_final.add(q[1])

def remove_entity_in_rel2quadruple():
    data = json.load(open('quadruples_gdelt_noinv_name', ))
    for line in data:
        head = line[0]
        tail = line[2]
        rel = line[1]
        time = line[3]

        if (head not in entities_background) or (tail not in entities_background):
            continue
        else:
            rel2quadruple_final[rel].append([head, rel, tail, time])  # no inverse relation!!! removed additional entities

def get_ent2id_noinv():
    i = 0
    for ent in sorted(entities_background):
        ent2id_noinv.update({ent: i})
        i += 1
    with open(f'{FOLDER_NAME}/ent2id_noinv.json', 'w') as ef:
        json.dump(ent2id_noinv, ef)

def get_rel2id_noinv():
    i = 0
    for rel in sorted(relations):
        rel2id_noinv.update({rel: i})
        i += 1
    with open(f'{FOLDER_NAME}/rel2id_noinv.json', 'w') as rf:
        json.dump(rel2id_noinv, rf)

def get_ts2id_noinv():
    i = 0
    for t in sorted(ts):
        ts2id_noinv.update({str(t): i})
        i += 1
    with open(f'{FOLDER_NAME}/ts2id_noinv.json', 'w') as tf:
        json.dump(ts2id_noinv, tf)

def sparse_rel_time_span():
    for rel in sparse_rel:
        min_t, max_t = 100000000, 0
        for q in rel2quadruple_final[rel]:
            if int(q[3]) < min_t:
                min_t = int(q[3])
            if int(q[3]) > max_t:
                max_t = int(q[3])
        sparse_rel_time_span_dict.update({rel:[min_t, max_t]})


max_time= rel_count()
total_distribution = []
# for (rel, cnt) in rel_counter.items():

classify_rel() # pre classify

print(max_time)
# print(rel_counter)
print(len(rel_counter))
# print(background_rel)
print(len(background_rel))
# print(sparse_rel)
print(len(sparse_rel))


split_one_shot_dataset() # remove new entities associated quadruples. NOTE: this will remove sparse relation
remove_entity_in_rel2quadruple()
e_all = set()
for r in rel2quadruple_final.keys():
    for q in rel2quadruple_final[r]:
        e_all.add(q[0])
        e_all.add(q[2])
print(len(e_all))
get_ent2id_noinv()
get_rel2id_noinv()
get_ts2id_noinv()
# sparse_rel_time_span()
# print(sparse_rel_time_span_dict)

# split train, valid, test set
print(len(relations))
print(len(relation_sparse_final))
sparse_rel_rd = list(copy.deepcopy(relation_sparse_final))
random.shuffle(sparse_rel_rd)
# print(sparse_rel)
# print(sparse_rel_rd)
train_l = int(len(sparse_rel_rd) * 0.8)
valid_l = int(len(sparse_rel_rd) * 0.1)
sparse_rel_train = sparse_rel_rd[:train_l]
sparse_rel_valid = sparse_rel_rd[train_l:train_l+valid_l]
sparse_rel_test = sparse_rel_rd[train_l+valid_l:]

print(sparse_rel_train)
print(sparse_rel_valid)
print(sparse_rel_test)
print(len(sparse_rel_train), len(sparse_rel_valid), len(sparse_rel_test))

shutil.copyfile('./stat.txt',f'{FOLDER_NAME}/stat.txt')


with open(f'{FOLDER_NAME}/sparse_rel.pkl', 'wb') as f:
    pickle.dump(sparse_rel, f)

with open(f'{FOLDER_NAME}/background_rel.pkl', 'wb') as f:
    pickle.dump(background_rel, f)

with open(f'{FOLDER_NAME}/rel2quadruple.pkl', 'wb') as f:
    pickle.dump(rel2quadruple_final, f)

with open(f'{FOLDER_NAME}/sparse_rel_train.pkl', 'wb') as f:
    pickle.dump(sparse_rel_train, f)

with open(f'{FOLDER_NAME}/sparse_rel_valid.pkl', 'wb') as f:
    pickle.dump(sparse_rel_valid, f)

with open(f'{FOLDER_NAME}/sparse_rel_test.pkl', 'wb') as f:
    pickle.dump(sparse_rel_test, f)