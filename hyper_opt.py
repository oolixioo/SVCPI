from hyperopt import hp, fmin, tpe
from copy import deepcopy
from train import run_train
import pickle, os

def gen_hyperopt():
    cfg = {}
    cfg['seed']                = 1776
    cfg['max_epochs']          = 15
    cfg['batch_size']          = hp.choice('batch_train', [8, 16, 32, 64, 128])

    b1 = hp.choice('blcok_1_size', (16,32,64,128,256))
    b2 = hp.choice('blcok_2_size', (16,32,64,128,256,None))
    b3 = hp.choice('blcok_3_size', (16,32,64,128,256,None))
    b4 = hp.choice('blcok_4_size', (16,32,64,128,256,None))

    cfg['n_filters_list']      = [b1, b2, b3, b4]

    cfg['mlp_layers']           = hp.choice('mlp_layers', (1,2,3))
    cfg['readout_layers']       = hp.choice('readout_layers', (1,2,3))
    cfg['cnn_layers']           = 1
    cfg['n_head']               = hp.choice('n_head', (2,4,6,8,10))
    cfg['dim']                  = hp.choice('dim', (10,16,32,64,128))
    cfg['fp_linear_layers']     = hp.choice('fp_linear_layers', (1,2,3))

    return cfg

def bayesian_opt(max_evals=500, save_dir='hyperopt', log_fn=''):
    ''' Run grid searching for best hyperopts '''
    results = []
    candidate_hypers = gen_hyperopt()

    d = {
    	'save_dir': save_dir,
    	'bias': True,
    	'shuffle': False,
    	'shuffle_idx': False,
    	'nfold': 5,
    	'datadir': 'data/human',
        'model_path': None,
    	'split_index_path': 'data/human/split_index.pkl',
    	}

    candidate_hypers.update(d)

    data = pickle.load(open(d['datadir'] + os.sep + 'data.pkl', 'rb'))

    testdata = None
    if os.path.exists(d['datadir'] + os.sep + 'test.pkl'):
        testdata = pickle.load(open(d['datadir'] + os.sep + 'test.pkl', 'rb'))

    if log_fn:
        open(log_fn, 'w').write('AUCs\tEval_val\tPath\n')

    def objective(cfg):
        train_path, best_auc, best_eval_val = run_train(cfg, data, testdata, isopt=True)
        results.append((train_path, best_auc, best_eval_val))

        if log_fn:
            open(log_fn, 'a').write('%s\t%s\t%s\n' %(round(best_auc, 4), round(best_eval_val, 4), train_path))

        return best_eval_val

    fmin(lambda x: -objective(x), candidate_hypers, algo=tpe.suggest, max_evals=max_evals)

    bev = 0
    bidx = 0
    for idx, (_, _, e) in enumerate(results):
    	if e > bev:
    		bev = e
    		bidx = idx

    best_trial_path, best_auc, best_eval_val = results[bidx]

    open(log_fn, 'a').write('Best hypers in %s, best auc is %s, best best_eval_val is %s.' %(best_trial_path, best_auc, best_eval_val))

    return best_trial_path, best_auc, best_eval_val

if __name__ == '__main__':
    max_evals = 500
    save_dir = 'data/human/hyperopt'
    log_fn = '%s/log.txt' %save_dir
    bayesian_opt(max_evals, save_dir, log_fn)
