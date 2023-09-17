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
import shutil
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, precision_recall_curve, auc
from create_dataset import create_for_predict
import argparse

DEVICE = torch.cuda.is_available() and torch.device('cuda') or torch.device('cpu')

print(DEVICE)

def predict(net, data, batch_size, protein_named={}):
    num_correct = 0
    len_dataset = len(data)
    net.eval()
    T, Y, S = [], [], []
    datad = {'index': [], 'smiles': [], 'protein': []}

    rounds = len_dataset/batch_size
    if rounds > int(rounds):
        rounds = int(rounds) + 1
    rounds = int(rounds)

    with torch.no_grad():
        for n, i in enumerate(range(0, len_dataset, batch_size)):
            i, j = i, i+batch_size
            data_batch = utils.create_mol_protein_batch(data[i:j], pad=True, device=DEVICE, pr=False)
            outputs, labels, pred_labels, pred_scores = net(data_batch)

            datad['index'].extend(data_batch['index'])
            datad['smiles'].extend(data_batch['smiles'])

            if protein_named:
                for seq in data_batch['protein']:
                    if seq in protein_named:
                        datad['protein'].append(protein_named[seq])
                    else:
                        datad['protein'].append(seq)
            else:
                datad['protein'].extend(data_batch['protein'])

            T.extend(labels.to('cpu').data.numpy())
            Y.extend(pred_labels.to('cpu').data.numpy())
            S.extend(pred_scores.to('cpu').data.numpy())

            print(' Predicting... [%s/%s] %.2f%%              ' %(n+1, rounds, (n+1)/rounds*100), end='\r')

    df = pd.DataFrame({
                        'Index': datad['index'], 
                        'Smiles': datad['smiles'],
                        'Protein': datad['protein'],
                        'Label': T,
                        'Predict': Y,
                        'Score': S})

    rocauc      = roc_auc_score(T, S)
    prec        = precision_score(T, Y, zero_division=0)
    recall      = recall_score(T, Y, zero_division=0)
    tpr, fpr, _ = precision_recall_curve(T, S)
    prcauc      = auc(fpr, tpr)

    fmt_func = lambda x:str(round(x,5)).rjust(9,' ')
    rst = [rocauc, prec, recall, prcauc]
    s = '\t'.join(map(fmt_func, rst))

    print('\n\nAUC_pred\tPrec_pred\tRecall_pred\tPRAUC_pred\n%s\n' %s)

    return df

def run_predict(cfg):
    print(cfg)

    batch_size = cfg['batch_size']
    data_file = cfg['data_file']
    out_file = cfg['out_file']
    model_path = cfg['model_path']
    protein_name_file = cfg['protein_name_file']

    if model_path and os.path.exists(model_path):
        net = torch.load(model_path)
        net.eval()
        print(net)
    else:
        return

    in_file = data_file
    if not data_file.endswith('.pkl'):
        in_file = data_file.split('.')[0] + '_for_pred.pkl'
        print('Processing input file...')
        create_for_predict(open(data_file).readlines(), in_file)

    protein_named = {}
    if os.path.exists(protein_name_file):
        for l in open(protein_name_file).readlines():
            ll = [s.strip() for s in l.split(',') if s.strip()]
            protein_named[ll[0]] = ll[1]

    print('Loading dataset...')
    data = pickle.load(open(in_file, 'rb'))

    start_t = timeit.default_timer()

    df = predict(net, data, batch_size, protein_named)

    df.to_csv(out_file, index=False)

    elapsed_t = timeit.default_timer() - start_t

    print('Time used: %s s\n' %elapsed_t)


if __name__ == '__main__':
    cfg = {}

    parser = argparse.ArgumentParser()
    parser.add_argument('-df', '--data_file', type=str, required=True, default='', help='The data path')
    parser.add_argument('-bs', '--batch_size', type=int, required=False, default=16, help='Batch size')
    parser.add_argument('-of', '--out_file', type=str, required=True, default='', help='The out file path')
    parser.add_argument('-mp', '--model_path', type=str, required=True, default='', help='The model path')
    parser.add_argument('-pnf', '--protein_name_file', type=str, required=False, default='', help='The protein name path')
    args = parser.parse_args()

    cfg['data_file']        = args.data_file
    cfg['batch_size']       = args.batch_size
    cfg['out_file']         = args.out_file
    cfg['model_path']       = args.model_path
    cfg['protein_name_file']= args.protein_name_file

    run_predict(cfg)