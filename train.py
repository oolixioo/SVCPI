import pickle
import model as model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timeit, random
from tqdm import tqdm
import numpy as np
import utils, os, json, sys
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, precision_score, recall_score, precision_recall_curve, auc
import shutil
from sklearn.model_selection import KFold

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

DEVICE = torch.cuda.is_available() and torch.device('cuda') or torch.device('cpu')

print(DEVICE)

loss_func = nn.CrossEntropyLoss()

def make_next_record_dir(basedir):
    path = '%s/001/' %basedir
    n = 2
    while os.path.exists(path):
        path = '%s/%.3d/' %(basedir, n)
        n += 1
    os.makedirs(path)
    return path

def save_model(net, file_path):
    torch.save(net, file_path)

def load_model_state(path, net):
    print('Load model state from %s' %path)
    d = torch.load(path)
    net.load_state_dict(d.state_dict())

header = 'Epoch\tTime(sec)\tLoss_train\tAUC_train\tAUC_dev\tPrec_train\tPrec_dev\tRecall_train\tRecall_dev\tPR_AUC_dev\tTest_eval\n'
def fmt_aucs(rst, pheader=False):
    fmt_func = lambda x:str(round(x,5)).rjust(9,' ')
    txt = str(rst[0]) + '\t' + '\t'.join(map(fmt_func, rst[1:])) + '\n'

    if pheader:
        txt = header + txt
    return txt

def save_aucs(rst, fn):
    txt = fmt_aucs(rst)

    if not os.path.exists(fn):
        with open(fn, 'w') as f:
            f.write(header)
            f.write(txt)
    else:
        with open(fn, 'a') as f:
            f.write(txt)

def test(net, data, batch_size, isopt=False):
    num_correct = 0
    len_dataset = len(data)
    net.eval()
    T, Y, S = [], [], []
    running_loss = 0.0

    rounds = len_dataset/batch_size
    if rounds > int(rounds):
        rounds = int(rounds) + 1
    rounds = int(rounds)

    with torch.no_grad():
        for n, i in enumerate(range(0, len_dataset, batch_size)):
            i, j = i, i+batch_size
            data_batch = utils.create_mol_protein_batch(data[i:j], pad=True, device=DEVICE, pr=False)
            outputs, labels, pred_labels, pred_scores = net(data_batch)

            T.extend(labels.to('cpu').data.numpy())
            Y.extend(pred_labels.to('cpu').data.numpy())
            S.extend(pred_scores.to('cpu').data.numpy())

            if not isopt:
                print(' Testing... [%s/%s] %.2f%%              ' %(n+1, rounds, (n+1)/rounds*100), end='\r')

    rocauc      = roc_auc_score(T, S)
    prec        = precision_score(T, Y, zero_division=0)
    recall      = recall_score(T, Y, zero_division=0)
    tpr, fpr, _ = precision_recall_curve(T, S)
    prcauc      = auc(fpr, tpr)

    return rocauc, prec, recall, prcauc

def run_train(cfg, data, test_data=None, isopt=False):
    global header
    print(cfg)

    batch_size = cfg['batch_size']
    n_filters_list = cfg['n_filters_list']
    mlp_layers = cfg['mlp_layers']
    n_head = cfg['n_head']
    readout_layers = cfg['readout_layers']
    dim = cfg['dim']
    bias = cfg['bias']
    split_index_path = cfg['split_index_path']
    nfold = cfg['nfold']
    fp_linear_layers = cfg['fp_linear_layers']
    model_path = cfg['model_path']

    save_path               = make_next_record_dir(cfg['save_dir']) + os.sep
    file_net                = save_path + 'net.txt'
    file_cfg                = save_path + 'cfg.json'
    file_maes               = save_path + 'train_log.txt'
    file_split_idx          = save_path + 'split_index.pkl'

    net                     = None
    optimizer               = None
    scheduler               = None

    C     = data[0]['V'].shape[-1]
    L     = data[0]['A'].shape[-2]
    fpdim = data[0]['fp'].shape[-1]

    def init_model():
        try:
            seed = cfg['seed']
            setup_seed(seed)
        except:
            setup_seed(1776)

        net = model.Net(C, L, fpdim, n_filters_list, mlp_layers, n_head, readout_layers, dim=dim, bias=bias, fp_linear_layers=fp_linear_layers).to(DEVICE)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-03)
        scheduler = ReduceLROnPlateau(optimizer, patience=4, mode='min', verbose=True)
        #print(net, next(net.parameters()).is_cuda)
        
        if model_path and os.path.exists(model_path):
            load_model_state(model_path, net, optimizer)

        return net, optimizer, scheduler

    # nfold validation
    uskf = False
    if os.path.exists(split_index_path):
        try:
            itero = pickle.load(open(split_index_path, 'rb'))
            print('Using split index from file: %s' %split_index_path)
        except:
            uskf = True
    else:
        uskf = True

    if uskf:
        print('Using split index from KFold.split')
        kf = KFold(n_splits=nfold, shuffle=True, random_state=1000)
        itero = list(kf.split(data))

    pickle.dump(itero, open(file_split_idx, 'wb'))

    #copy_code_files(save_path)

    json.dump(cfg, open(file_cfg, 'w'), indent=2)

    net, optimizer, scheduler = init_model()

    open(file_net, 'w').write(str(net))

    start_t = timeit.default_timer()

    # nfold validation
    uskf = False
    if os.path.exists(split_index_path):
        try:
            itero = pickle.load(open(split_index_path, 'rb'))
            print('Using split index from file: %s' %split_index_path)
        except:
            uskf = True
    else:
        uskf = True

    if uskf:
        print('Using split index from KFold.split')
        kf = KFold(n_splits=nfold, shuffle=True, random_state=1000)
        itero = list(kf.split(data))

    pickle.dump(itero, open(file_split_idx, 'wb'))

    for fold, (train_idx, dev_idx) in enumerate(itero):
        print('Training fold %s of %s...' %(fold+1, len(itero)))

        save_aucs('Fold %s:\n' %fold, file_maes)

        if cfg['shuffle_idx']:
            print('Shuffle split index...')
            pickle.dump(train_idx, open('/tmp/tmp.pkl', 'wb'))
            os.system('python shuffle.py /tmp/tmp.pkl')
            train_idx = pickle.load(open('/tmp/tmp.pkl', 'rb'))

            pickle.dump(dev_idx, open('/tmp/tmp.pkl', 'wb'))
            os.system('python shuffle.py /tmp/tmp.pkl')
            dev_idx = pickle.load(open('/tmp/tmp.pkl', 'rb'))

            if test_data:
                print('Shuffle test_data...')
                pickle.dump(test_data, open('/tmp/tmp.pkl', 'wb'))
                os.system('python shuffle.py /tmp/tmp.pkl')
                test_data = pickle.load(open('/tmp/tmp.pkl', 'rb'))

        best_auc = 0
        best_eval_val = 0

        if fold > 0:
            print('New net and optimizer for new fold')
            net, optimizer, scheduler = init_model()

        train_data = [data[i] for i in train_idx]
        dev_data   = [data[i] for i in dev_idx]
        len_dataset = len(train_data)

        for epoch in range(cfg['max_epochs']):  # loop over the dataset multiple times
            running_loss = 0.0
            net.train(True)
            rounds = len_dataset/batch_size
            if rounds > int(rounds):
                rounds = int(rounds) + 1
            rounds = int(rounds)

            for n, i in enumerate(range(0, len_dataset, batch_size)):
                i, j = i, i+batch_size
                data_batch = utils.create_mol_protein_batch(train_data[i:j], pad=True, device=DEVICE, pr=False)
                optimizer.zero_grad()
                outputs, labels, pred_labels, pred_scores = net(data_batch)
                #print(outputs.shape, lbs.shape)
                loss = loss_func(outputs, labels)
                running_loss += loss.item()
                loss.backward()
                optimizer.step()

                if not isopt:
                    print(' Training... [%s/%s] %.2f%%               ' %(n+1, rounds, (n+1)/rounds*100), end='\r')

            # eval
            train_auc, train_prec, train_recall = 0, 0, 0
            if not isopt and epoch % 1 == 0:
                train_auc, train_prec, train_recall, train_pr_auc = test(net, train_data, batch_size, isopt)
            dev_auc, dev_prec, dev_recall, dev_pr_auc = test(net, dev_data, batch_size, isopt)

            scheduler.step(dev_auc)

            if test_data:
                header = 'Epoch\tTime(sec)\tLoss_train\tAUC_train\tAUC_dev \tAUC_test\tPrec_train\tPrec_dev \tPrec_test\tRecall_train\tRecall_dev \tRecall_test\tPR_AUC_test\tTest_eval\n'
                test_auc, test_prec, test_recall, test_pr_auc = test(net, test_data, batch_size, isopt)
                test_eval_val = sum([test_auc, test_prec, test_recall])/3.

                if test_auc > best_auc:
                    best_auc = test_auc

                rst = [epoch, 0, running_loss, train_auc, dev_auc, test_auc, train_prec, dev_prec, test_prec, train_recall, dev_recall, test_recall, test_pr_auc, 0]
            else:
                header = 'Epoch\tTime(sec)\tLoss_train\tAUC_train\tAUC_dev \tPrec_train\tPrec_dev \tRecall_train\tRecall_dev \tPR_AUC_dev\tTest_eval\n'
                test_eval_val = sum([dev_auc, dev_prec, dev_recall])/3.
                if dev_auc > best_auc:
                    best_auc = dev_auc

                rst = [epoch, 0, running_loss, train_auc, dev_auc, train_prec, dev_prec, train_recall, dev_recall, dev_pr_auc, 0]

            if test_eval_val > best_eval_val:
                best_eval_val = test_eval_val
                save_model(net, save_path+os.sep+'model-fold%s.h5' %fold)
                
            elapsed_t = timeit.default_timer() - start_t

            rst[1] = elapsed_t
            rst[-1] = test_eval_val

            if epoch == 0:
                ph = True
            else:
                ph = False

            if not isopt:
                print(' '*50, end='\r')
            print(fmt_aucs(rst, ph).strip())

            save_aucs(rst, file_maes)

    return save_path, best_auc, best_eval_val

if __name__ == '__main__':

    cfg = {
            'max_epochs': 21,
            'batch_size': 16,
            'n_filters_list': [16, 256, 32, 128],
            'mlp_layers': 3,
            'readout_layers': 1,
            'n_head': 2,
            'dim': 128,
            'bias': True,
            'shuffle_idx': False,
            'datadir': 'data/human',
            'nfold': 5,
            'fp_linear_layers': 2,
            'model_path': 'None',
            'seed': 1776
          }

    try:
        cfg = json.load(open(sys.argv[1]))
        print('Using config from file %s' %sys.argv[1])
    except:
        pass

    data = pickle.load(open(cfg['datadir'] + os.sep + 'data.pkl', 'rb'))

    if not os.path.exists(cfg['split_index_path']):
        cfg['split_index_path'] = cfg['datadir'] + os.sep + 'split_index.pkl'

    if not os.path.exists(cfg['save_dir']):
        cfg['save_dir'] = cfg['datadir'] + os.sep + 'result'

    testdata = None
    if os.path.exists(cfg['datadir'] + os.sep + 'test.pkl'):
        testdata = pickle.load(open(cfg['datadir'] + os.sep + 'test.pkl', 'rb'))

    print('len_dataset:', len(data), 'fold:', cfg['nfold'])

    run_train(cfg, data, testdata)