from tqdm import tqdm
from utils import *
import torch
import numpy as np


def get_label(label, num_e):
    y = np.zeros([num_e], dtype=np.float32)
    for e2 in label: y[e2] = 1.0
    # print(y)
    return torch.FloatTensor(y)


def predict(loader, model, sr2o, srt2o, o2srt, ts2id, logger, args=None, mode=None):
    if 'icews' in args.dataset.lower():
        print('DATASET IS ICEWS0515')
        split_test = '2013-01-01 00:00:00'
        split_val = '2011-01-01 00:00:00'
    elif 'gdelt' in args.dataset.lower():
        print('DATASET IS GDELT')
        split_test = '35700'
        split_val = '26775'
    model.eval()

    print(f'START EVALUATION, MODE:{mode}')

    with torch.no_grad():
        results = {}
        t1 = time.time()
        for sample in tqdm(loader):
            support_pairs = sample.support_pairs
            query_pairs = sample.query_pairs
            str_rel = sample.str_rel
            rel_results = {}
            num_quadruple = sample.num_quadruple

            # all_entity no need
            rel = sample.rel

            # each query_pair
            sub_l = [q[0] for q in query_pairs]
            obj_l = [q[1] for q in query_pairs]
            ts_l = [q[2] for q in query_pairs]
            r_l = [rel] * len(query_pairs)

            assert len(sub_l) == len(r_l)


            all_score = []

            if mode == 'ext':
                if args.is_rawts:
                    model.o2srt_filter(pairs=support_pairs[0], filter_ts=args.last_support_ts,o2srt = o2srt)

                if args.filter_until_test and args.test == 'test':
                    last_valid_time = ts2id[split_test]
                    model.o2srt_filter_until_test(pairs=support_pairs[0], last_valid_time=last_valid_time, o2srt=o2srt)

                if args.filter_until_test and (not args.test) == 'valid':
                    last_valid_time = ts2id[split_val]
                    model.o2srt_filter_until_test(pairs=support_pairs[0], last_valid_time=last_valid_time, o2srt=o2srt)

                for one_query_pair in query_pairs:
                    support_pairs = torch.tensor(support_pairs, device=model.device)
                    support_emb = model(support_pairs, is_random=args.is_random)

                    one_query_pair = torch.tensor(one_query_pair, device=model.device).unsqueeze(dim=0)
                    one_query_scores = model.matcher(support_emb, support_pairs, one_query_pair, model.emb_e)
                    all_score.append(one_query_scores)

            elif mode =='int':
                support_pairs = torch.tensor(support_pairs, device=model.device)
                support_emb = model(support_pairs, is_random=args.is_random)

                for one_query_pair in query_pairs:
                    one_query_pair = torch.tensor(one_query_pair, device=model.device).unsqueeze(dim=0)
                    one_query_scores = model.matcher(support_emb, support_pairs, one_query_pair, model.emb_e)
                    all_score.append(one_query_scores)

            else:
                raise NotImplementedError

            all_score = torch.vstack(all_score)

            b_range = torch.arange(all_score.size()[0], device=model.device)

            # raw ranking
            ranks = 1 + torch.argsort(torch.argsort(all_score, dim=1, descending=True), dim=1, descending=False)[
                b_range, obj_l]
            ranks = ranks.float()
            results['count_raw'] = torch.numel(ranks) + results.get('count_raw', 0.0)
            results['mar_raw'] = torch.sum(ranks).item() + results.get('mar_raw', 0.0)
            results['mrr_raw'] = torch.sum(1.0 / ranks).item() + results.get('mrr_raw', 0.0)
            for k in range(10):
                results['hits@{}_raw'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get(
                    'hits@{}_raw'.format(k + 1), 0.0)

            # need to clone otherwise the target_score has the same memory with score
            target_score = all_score[b_range, obj_l].clone()
            save_score = all_score.clone()
            # time independent filtering
            # print([sr2o[(int(s),int(r))] for (s, r) in zip(sub_l, r_l)])
            filter_label_ind = torch.stack(
                [get_label(sr2o[(int(s), int(r))], model.num_ent) for (s, r) in zip(sub_l, r_l)],
                dim=0).to(model.device)
            # self.byte() is equivalent to self.to(torch.uint8).
            all_score_ind = torch.where(filter_label_ind > 0,
                                        - torch.ones_like(all_score, device=model.device) * 10000000,
                                        save_score)

            all_score_ind[b_range, obj_l] = target_score

            # time independent filtered ranking
            ranks_ind = 1 + \
                        torch.argsort(torch.argsort(all_score_ind, dim=1, descending=True), dim=1, descending=False)[
                            b_range, obj_l]
            ranks_ind = ranks_ind.float()
            results['count_ind'] = torch.numel(ranks_ind) + results.get('count_ind', 0.0)
            results['mar_ind'] = torch.sum(ranks_ind).item() + results.get('mar_ind', 0.0)
            results['mrr_ind'] = torch.sum(1.0 / ranks_ind).item() + results.get('mrr_ind', 0.0)
            for k in range(10):
                results['hits@{}_ind'.format(k + 1)] = torch.numel(ranks_ind[ranks_ind <= (k + 1)]) + results.get(
                    'hits@{}_ind'.format(k + 1), 0.0)

            # time dependent filtering
            filter_label_dep = torch.stack(
                [get_label(srt2o[(int(s), int(r), int(t))], model.num_ent) for (s, r, t) in zip(sub_l, r_l, ts_l)],
                dim=0).to(model.device)

            # self.byte() is equivalent to self.to(torch.uint8).
            all_score_dep = torch.where(filter_label_dep > 0,
                                        - torch.ones_like(all_score, device=model.device) * 10000000,
                                        save_score)
            all_score_dep[b_range, obj_l] = target_score

            # time dependent filtered ranking
            ranks_dep = 1 + \
                        torch.argsort(torch.argsort(all_score_dep, dim=1, descending=True), dim=1, descending=False)[
                            b_range, obj_l]
            ranks_dep = ranks_dep.float()

            results['count_dep'] = torch.numel(ranks_dep) + results.get('count_dep', 0.0)
            results['mar_dep'] = torch.sum(ranks_dep).item() + results.get('mar_dep', 0.0)
            results['mrr_dep'] = torch.sum(1.0 / ranks_dep).item() + results.get('mrr_dep', 0.0)
            for k in range(10):
                results['hits@{}_dep'.format(k + 1)] = torch.numel(ranks_dep[ranks_dep <= (k + 1)]) + results.get(
                    'hits@{}_dep'.format(k + 1), 0.0)


            ind_mrr = torch.sum(1.0 / ranks_ind).item() / (torch.numel(ranks_ind))
            inf_hit10 = torch.numel(ranks_ind[ranks_ind <= (9 + 1)]) / (torch.numel(ranks_ind))
            rel_results[str_rel] = {'NUM_QUAD': num_quadruple,
                                    'IND_MRR': ind_mrr,
                                    'IND_HIT10': inf_hit10}

            logging.info(f"{str_rel}: NUM_QUAD:{num_quadruple},IND_MRR:{ind_mrr},IND_HIT10:{inf_hit10}")


        results['mar_raw'] = round(results['mar_raw'] / results['count_raw'], 5)
        results['mrr_raw'] = round(results['mrr_raw'] / results['count_raw'], 5)
        results['mar_ind'] = round(results['mar_ind'] / results['count_ind'], 5)
        results['mrr_ind'] = round(results['mrr_ind'] / results['count_ind'], 5)
        results['mar_dep'] = round(results['mar_dep'] / results['count_dep'], 5)
        results['mrr_dep'] = round(results['mrr_dep'] / results['count_dep'], 5)
        for k in range(10):
            results['hits@{}_raw'.format(k + 1)] = round(results['hits@{}_raw'.format(k + 1)] / results['count_raw'], 5)
            results['hits@{}_ind'.format(k + 1)] = round(results['hits@{}_ind'.format(k + 1)] / results['count_ind'], 5)
            results['hits@{}_dep'.format(k + 1)] = round(results['hits@{}_dep'.format(k + 1)] / results['count_dep'], 5)

        t2 = time.time()
        print("evaluation time: ", t2 - t1)
        logger.info("evaluation time: {}".format(t2 - t1))

    return results
