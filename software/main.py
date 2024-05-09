import wandb
from most import *
from eval import *
from utils import *
import argparse
from torch.utils.data import DataLoader


class SimpleCustomBatch:
    def __init__(self, data):
        self.rel = data[0]
        self.str_rel = contents.id2rel_inv[self.rel]
        train_and_test = contents.rel2quadruple[self.rel]
        self.num_quadruple = len(train_and_test)
        # do not shuffle when testing
        # random.shuffle(train_and_test)
        support_quadruples = train_and_test[:args.few]
        self.support_pairs = [[q[0], q[2], q[3]] for q in support_quadruples]
        query_quadruples = train_and_test[args.few:]
        self.query_pairs = [[q[0], q[2], q[3]] for q in query_quadruples]
        # use all the rest quadruples as query in training
        if len(query_quadruples) == 0:
            print('COULD NOT FIND QUERY')
            self.query_pairs = None
        else:
            self.query_pairs = [[q[0], q[2], q[3]] for q in query_quadruples]

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.support_pairs = self.support_pairs.pin_memory()
        self.query_pairs = self.query_pairs.pin_memory()
        self.rel = self.rel.pin_memory()
        self.str_rel = self.str_rel.pin_memory()
        self.num_quadruple = self.num_quadruple.pin_memory()
        pass


def collate_wrapper(batch):
    return SimpleCustomBatch(batch)

# help Module for custom Dataloader
def reset_time_cost():
    return {
        'model': defaultdict(float),
        'graph': defaultdict(float),
        'grad': defaultdict(float),
        'data': defaultdict(float)}

def save_model(model, args, best_val, best_batch, optimizer, save_path):
    state = {
        'state_dict': model.state_dict(),
        'best_val': best_val,
        'best_batch': best_batch,
        'optimizer': optimizer.state_dict(),
        'args': vars(args)
    }
    torch.save(state, save_path)
#
def load_model(load_path, optimizer, model):
    # state = torch.load(load_path,map_location={'cuda:0':'cuda:1'})
    # state = torch.load(load_path)
    state = torch.load(load_path, map_location=torch.device('cpu'))
    state_dict = state['state_dict']
    best_val = state['best_val']
    best_val_mrr = best_val['mrr_ind']

    model.load_state_dict(state_dict)
    optimizer.load_state_dict(state['optimizer'])

    return best_val_mrr
#

def load_emb(load_path, model):
    state = torch.load(load_path)
    state_dict = state['state_dict']
    ent_embed = state_dict['emb_e']
    rel_embed = state_dict['emb_r']
    assert ent_embed.shape[0] == len(contents.ent2id_noinv)
    assert rel_embed.shape[0] == len(contents.rel2id_noinv.keys()) * 2
    model_state_dict = model.state_dict()
    model_state_dict['emb_e'] = ent_embed
    model_state_dict['emb_r'] = rel_embed
    model.load_state_dict(model_state_dict)
    print("LOADING COMPLETE")
    return


if __name__ == '__main__':
    # curdir = os.getcwd()
    modelpth = './checkpoints/'
    parser = argparse.ArgumentParser(description='MOST')
    # args for rel_model
    parser.add_argument('--timer', action='store_true', default=True,
                        help='set to profile time consumption for some func')
    parser.add_argument('--few', type=int, default=1)
    parser.add_argument('--add_reverse', action='store_true', default=True, help='add reverse relation into data set')
    parser.add_argument('--num_batch', type=int, default=1000000, help='number of maximum epoch')
    parser.add_argument('--num_sample', type=int, default=512, help='number of samples for the neighbor')
    parser.add_argument('--core_layer', type=int, default=1, help='number of core function layers')
    parser.add_argument('--batch_size', type=int, default=64, help='number of examples in a batch')
    parser.add_argument('--embsize', type=int, default=100, help='size of output embeddings')
    parser.add_argument('--sim_inner', type=int, default=100, help='size of hidden embeddings')
    parser.add_argument("--process_steps", default=1, type=int)
    parser.add_argument('--test_step', type=int, default=100, help='test every test_step iteration')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument("--grad_clip", default=1, type=float)
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight of regularizer')
    parser.add_argument('--device', type=int, default=1, help='-1: cpu, >=0, cuda device')
    parser.add_argument('--dataset', type=str, default='ICEWS0515_INTERPOLATION', help='dataset name:ICEWS0515_INTERPOLATION,ICEWS0515_EXTRAPOLATION, GDELT_INTERPOLATION, GDELT_EXTRAPOLATION')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--bias', action='store_true', help='whether to use bias in relation specific transformation')
    parser.add_argument('--shuffle', action='store_true', help='shuffle in dataloader')
    parser.add_argument('--resume', type=bool, default=False, help='continue training')
    parser.add_argument('--name', type=str, help='name of checkpoint')
    parser.add_argument('--is_random', type=bool, default=False, help='random sample neighbors')
    parser.add_argument('--is_rawts', default=True, action='store_true',help='T:absolute time for node emb,F:time diff')
    parser.add_argument('--filter_until_test', type=bool, default=True, help='T:use last valid ts')
    parser.add_argument('--loss_fun', type=str, default='bce',
                        help='Loss Function')
    parser.add_argument('--activation', type=str, default='relu', help='activation function')
    parser.add_argument('--res', action='store_false')
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--rawportion', type=float, default=1)
    parser.add_argument('--rel_regularization', type=float, default=1, help='relation scale regularization')
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--test', default=True, action='store_true', help='test mode')
    parser.add_argument('--eval', default=False, action='store_true', help='evaluation')
    args = parser.parse_args()

    # Reproducibility
    # setup random seeds
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    # CUDA convolution determinism
    torch.backends.cudnn.deterministic = True
    # The dynamically changing network structure needs to be set to False
    torch.backends.cudnn.benchmark = False
    save_name = 'YOUR NAME HERE'+ time.strftime( "%Y_%m_%d_%H_%M_%S", time.localtime())

    LOG = get_logger(save_name)
    LOG.info(args)

    if not os.path.exists(modelpth):
        os.mkdir(modelpth)
    loadpth = modelpth + save_name

    # check cuda
    if torch.cuda.is_available():
        device = 'cuda:{}'.format(args.device) if args.device >= 0 else 'cpu'
        args.device = device
    else:
        device = 'cpu'
        args.device = device
    print(f'DEVICE: {device}')

    # profile time consumption
    time_cost = None
    if args.timer:
        time_cost = reset_time_cost()

    # init model and checkpoint folder
    start_time = time.time()
    struct_time = time.gmtime(start_time)

    # load data
    contents = MOSTData(dataset=args.dataset, shuffle_first=True)

    num_entities = len(contents.ent2id_noinv)
    num_relations = contents.num_rel
    max_time = max(contents.ts2id_noinv.values())

    LOG.info('NUMBER OF ENTITIES:{0}'.format(num_entities))
    LOG.info('NUMBER OF RELATIONS:{0}, ADD_REVERSE'.format(num_relations, args.add_reverse))

    # initialize time encoder
    time_encoder = TimeEncoder(args.embsize, device=device)
    # time_encoder.to(device)
    # initiallize Meta-relational Decoder
    matcher = Decoder(args.embsize, args.sim_inner, args.dropout, loss_fun=args.loss_fun,
                             time_encoder=time_encoder, is_rawts=args.is_rawts, rel_reg=args.rel_regularization)

    # initial loss function
    if args.loss_fun == 'bce':
        LOG.info('USING BCELOSS')
    else:
        raise NotImplementedError('SPECIFY THE LOSS FUNCTION')

    # initiallize Model
    model = MOST(matcher, num_entities, num_relations, contents.id2ts, contents.o2srt, args)
    model.to(device)

    # inspect model parameters for name
    for name, param in model.named_parameters():
        print(name, '   ', param.size())

    # optimizer
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[100000], gamma=0.1)

    # training loop
    best_batch = 0
    best_val = 0
    best_val_mrr = 0

    batch_loss = 0
    running_loss = 0
    t_train_start = time.time()
    rel_pool = contents.sparse_rel_train
    loss = 0
    model.zero_grad()
    model.train()


    if contents.mode == 'ext':
        if args.is_rawts:
            print('*' * 100)
            print('SAVING LAST TS OF SUPPORT')
            print('*' * 100)
        else:
            print('*' * 100)
            print('SAVING THE MAXIMUM TS GAP FOR TEST')
            print('*' * 100)
        # for absolute timestamp using the last ts of support as filter ts
        args.last_support_ts = 0
        args.max_ts_diff = 0


    if args.eval:
        batch_num = 'EVALUATION'
        if args.dataset == 'ICEWS0515_INTERPOLATION':
            if args.is_rawts == True:
                LOG.info('LOADING ICEWS0515 MOST_INT_TA')
                best_val_mrr = load_model(load_path='',
                                          optimizer=optim,model=model)
            elif args.is_rawts == False:
                LOG.info('LOADING ICEWS0515 MOST_INT_TD')
                best_val_mrr = load_model(load_path='',
                                          optimizer=optim,model=model)
        #
        if args.dataset == 'ICEWS0515_EXTRAPOLATION':
            if args.is_rawts == True:
                LOG.info('LOADING ICEWS0515 MOST_EXT_TA')
                best_val_mrr = load_model(load_path='',
                                          optimizer=optim,model=model)
            elif args.is_rawts == False:
                LOG.info('LOADING ICEWS0515 MOST_EXT_TD')
                best_val_mrr = load_model(load_path='./saved_model/ICEWS0515_EXTRAPOLATION_TD',
                                          optimizer=optim,model=model)
        if args.dataset == 'GDELT_INTERPOLATION':
            if args.is_rawts == True:
                LOG.info('LOADING GDELT MOST_INT_TA')
                best_val_mrr = load_model(load_path='./saved_model/GDELT_INTERPOLATION_TA',
                                          optimizer=optim,model=model)
            elif args.is_rawts == False:
                LOG.info('LOADING ICEWS0515 MOST_INT_TD')
                best_val_mrr = load_model(load_path='./saved_model/GDELT_INTERPOLATION_TD',
                                          optimizer=optim,model=model)

        if args.dataset == 'GDELT_EXTRAPOLATION':
            if args.is_rawts == True:
                LOG.info('LOADING GDELT MOST_EXT_TA')
                best_val_mrr = load_model(load_path='./saved_model/GDELT_EXTRAPOLATION_TA',
                                          optimizer=optim,model=model)
            elif args.is_rawts == False:
                LOG.info('LOADING GDELT MOST_EXT_TD')
                best_val_mrr = load_model(load_path='./saved_model/GDELT_EXTRAPOLATION_TD',
                                          optimizer=optim,model=model)
        #
        print(f'RELOAD MODEL, BEST_VAL_MRR IS {best_val_mrr}')

        # loader for test
        if args.test:
            sparse_rel = contents.sparse_rel_test
        else:
            sparse_rel = contents.sparse_rel_valid

        data_loader = DataLoader(sparse_rel, batch_size=1, collate_fn=collate_wrapper, pin_memory=False, shuffle=True)
        results = predict(data_loader, model, contents.sr2o, contents.srt2o, contents.o2srt, contents.ts2id_noinv, LOG, args=args, mode = contents.mode)

        LOG.info("===========RAW===========")
        LOG.info("Batch {}, HITS10 {}".format(batch_num, results['hits@10_raw']))
        LOG.info("Batch {}, HITS5 {}".format(batch_num, results['hits@5_raw']))
        LOG.info("Batch {}, HITS3 {}".format(batch_num, results['hits@3_raw']))
        LOG.info("Batch {}, HITS1 {}".format(batch_num, results['hits@1_raw']))
        LOG.info("Batch {}, MRR {}".format(batch_num, results['mrr_raw']))
        LOG.info("Batch {}, MAR {}".format(batch_num, results['mar_raw']))

        LOG.info("====TIME INDEP FILTER====")
        LOG.info("Batch {}, HITS10 {}".format(batch_num, results['hits@10_ind']))
        LOG.info("Batch {}, HITS5 {}".format(batch_num, results['hits@5_ind']))
        LOG.info("Batch {}, HITS3 {}".format(batch_num, results['hits@3_ind']))
        LOG.info("Batch {}, HITS1 {}".format(batch_num, results['hits@1_ind']))
        LOG.info("Batch {}, MRR {}".format(batch_num, results['mrr_ind']))
        LOG.info("Batch {}, MAR {}".format(batch_num, results['mar_ind']))

        LOG.info("=====TIME DEP FILTER=====")
        LOG.info("Batch {}, HITS10 {}".format(batch_num, results['hits@10_dep']))
        LOG.info("Batch {}, HITS5 {}".format(batch_num, results['hits@5_dep']))
        LOG.info("Batch {}, HITS3 {}".format(batch_num, results['hits@3_dep']))
        LOG.info("Batch {}, HITS1 {}".format(batch_num, results['hits@1_dep']))
        LOG.info("Batch {}, MRR {}".format(batch_num, results['mrr_dep']))
        LOG.info("Batch {}, MAR {}".format(batch_num, results['mar_dep']))

    else:
        print(args.resume)
        if args.resume:
            LOG.info('CONTINUE TRANING')
            _ = load_model(load_path=args.name, optimizer=optim, model=model)

        batch_true_query_emb = []
        batch_false_query_emb = []

        for batch_num in tqdm(range(1, args.num_batch + 1)):
            if batch_num % len(rel_pool) == 0:
                random.shuffle(rel_pool)
            rel_train = rel_pool[batch_num % len(rel_pool)]

            # get training data
            support_pairs, query_pairs, false_pairs = contents.get_data(rel_train, args.few, args.batch_size)
            assert len(query_pairs) == len(false_pairs)
            if len(query_pairs) < 1:
                print(f'NO CANDIDATES IN RELATION: {rel_train}')
                continue

            if contents.mode == 'ext':
                if args.is_rawts:
                    if args.last_support_ts < support_pairs[0][-1]:
                        args.last_support_ts = support_pairs[0][-1]
                else:
                    support_t = support_pairs[0][-1]
                    query_t = [q[-1] for q in query_pairs]
                    if args.max_ts_diff < (max(query_t) - support_t):
                        args.max_ts_diff = max(query_t) - support_t

            all_score = []
            if contents.mode == 'ext':
                for one_query_pair in query_pairs:
                    model.o2srt_filter(pairs=support_pairs[0], filter_ts=one_query_pair[2], o2srt=contents.o2srt)
                    support_pairs = torch.tensor(support_pairs, device=device)
                    support_emb = model(support_pairs, is_random=args.is_random)
                    one_query_pair = torch.tensor(one_query_pair, device=device).unsqueeze(dim=0)
                    one_query_scores = model.matcher(support_emb, support_pairs, one_query_pair, model.emb_e)
                    all_score.append(one_query_scores)
                query_scores = torch.vstack(all_score)

            elif contents.mode =='int':
                support_pairs = torch.tensor(support_pairs, device=device)
                support_emb = model(support_pairs, is_random=args.is_random)
                for one_query_pair in query_pairs:
                    one_query_pair = torch.tensor(one_query_pair, device=device).unsqueeze(dim=0)
                    one_query_scores = model.matcher(support_emb, support_pairs,one_query_pair, model.emb_e)
                    all_score.append(one_query_scores)
                query_scores = torch.vstack(all_score)
            else:
                raise NotImplementedError

            true_index = torch.tensor(query_pairs, device=device)[:, 1].to(device)
            loss_fun = torch.nn.BCELoss()
            trp_label = torch.zeros_like(query_scores, device=device)
            for id, e2 in enumerate(true_index): trp_label[id, e2] = 1.0
            loss = loss_fun(query_scores, trp_label)

            # update output
            running_loss += loss.item()
            # wandb.log({"loss": loss.item()})
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()
            scheduler.step()

            if (batch_num) % args.test_step == 0:
                t_train_end = time.time()
                batch_loss = running_loss / args.test_step
                running_loss = 0
                # loader for test
                if args.test:
                    sparse_rel = contents.sparse_rel_test
                else:
                    sparse_rel = contents.sparse_rel_valid

                data_loader = DataLoader(sparse_rel, batch_size=1, collate_fn=collate_wrapper,
                                         pin_memory=False, shuffle=True)
                results = predict(data_loader, model, contents.sr2o, contents.srt2o, contents.o2srt,
                                  contents.ts2id_noinv, LOG, args=args,mode=contents.mode)

                # wandb.log({"ind_val_mrr": results['mrr_ind'], "dep_val_mrr": results['mrr_dep']})
                
                # report loss information
                LOG.info("Batch " + str(batch_num) + ": " + str(batch_loss) + " Time: " + str(t_train_end - t_train_start))
                t_train_start = time.time()

                LOG.info("===========RAW===========")
                LOG.info("Batch {}, HITS10 {}".format(batch_num, results['hits@10_raw']))
                LOG.info("Batch {}, HITS5 {}".format(batch_num, results['hits@5_raw']))
                LOG.info("Batch {}, HITS3 {}".format(batch_num, results['hits@3_raw']))
                LOG.info("Batch {}, HITS1 {}".format(batch_num, results['hits@1_raw']))
                LOG.info("Batch {}, MRR {}".format(batch_num, results['mrr_raw']))
                LOG.info("Batch {}, MAR {}".format(batch_num, results['mar_raw']))

                LOG.info("====TIME INDEP FILTER====")
                LOG.info("Batch {}, HITS10 {}".format(batch_num, results['hits@10_ind']))
                LOG.info("Batch {}, HITS5 {}".format(batch_num, results['hits@5_ind']))
                LOG.info("Batch {}, HITS3 {}".format(batch_num, results['hits@3_ind']))
                LOG.info("Batch {}, HITS1 {}".format(batch_num, results['hits@1_ind']))
                LOG.info("Batch {}, MRR {}".format(batch_num, results['mrr_ind']))
                LOG.info("Batch {}, MAR {}".format(batch_num, results['mar_ind']))

                LOG.info("=====TIME DEP FILTER=====")
                LOG.info("Batch {}, HITS10 {}".format(batch_num, results['hits@10_dep']))
                LOG.info("Batch {}, HITS5 {}".format(batch_num, results['hits@5_dep']))
                LOG.info("Batch {}, HITS3 {}".format(batch_num, results['hits@3_dep']))
                LOG.info("Batch {}, HITS1 {}".format(batch_num, results['hits@1_dep']))
                LOG.info("Batch {}, MRR {}".format(batch_num, results['mrr_dep']))
                LOG.info("Batch {}, MAR {}".format(batch_num, results['mar_dep']))

                if results['mrr_ind'] > best_val_mrr:
                    best_val = results
                    best_val_mrr = results['mrr_ind']
                    best_batch = batch_num
                    save_model(model, args, best_val, best_batch, optim, loadpth)

                LOG.info("========BEST INDEP MRR=========")
                LOG.info("Batch {}, MRR {}".format(best_batch, best_val_mrr))
