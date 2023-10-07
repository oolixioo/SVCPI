import torch, json, sys
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import pickle, os
from rdkit import Chem
from tqdm import tqdm
import argparse
import featurizer as ft
import utils
import protein as prot
from joblib import Parallel, delayed, cpu_count, parallel_backend
from sklearn.model_selection import KFold
import subgraphfp as subfp

def calc_single(df,
                idx,
                smiles_column='smiles',
                protein_column='protein',
                label_column='activity',
                featd = {},
                include_seq=False):

    smi   = df.loc[idx][smiles_column]
    seq   = df.loc[idx][protein_column]
    label = df.loc[idx][label_column]

    # mol_graph = {V, A, global_state, mol_size, subgraph_size}
    mol_graph = ft.calc_data_from_smile(smi,
                                        addh=featd.setdefault('addh', True),
                                        calc_desc=featd.setdefault('calc_desc', False),
                                        with_ring_conj=featd.setdefault('with_ring_conj', True),
                                        with_atom_feats=featd.setdefault('with_atom_feats', True),
                                        with_submol_fp=featd.setdefault('with_submol_fp', False),
                                        with_subgraph_fp=featd.setdefault('with_subgraph_fp', False))
    mol = Chem.MolFromSmiles(smi)
 
    fp = subfp.gen_fps_from_mol(mol,
                                nbits=featd.setdefault('nbits', 512),
                                use_morgan=featd.setdefault('use_morgan', True),
                                use_macc=featd.setdefault('use_macc', True),
                                use_rdkit=featd.setdefault('use_rdkit', False))

    protein_seq = prot.split_prot_seq(seq, featd.setdefault('prot_n', 3), cover=featd.setdefault('cover', 1))

    if not include_seq:
        seq = ''

    mol_graph.update({'label': torch.LongTensor([label]), 'protein_seq': protein_seq, 'fp': torch.FloatTensor(fp), 'index': idx, 'smiles': smi, 'protein': seq})

    return mol_graph

def create_for_predict(smi_seq_lines, output_path, featd={}):
    smi = []
    seq = []
    label = []
    for line in smi_seq_lines:
        l = [s.strip() for s in line.split(' ') if s.strip()]
        smi.append(l[0])
        seq.append(l[1])
        try:
            label.append(int(l[2]))
        except:
            label.append(-1)

    lensmi = len(smi)

    pred_df = pd.DataFrame({'smiles': smi, 'protein': seq, 'activity': label})

    pred_df.index.name = 'index'

    def calc(idx, df):
        try:
            return calc_single(df, idx, featd=featd, include_seq=True)
        except Exception as e:
            print('\nError processing idx %s: %s' %(idx, e))
            print(df.loc[idx])
            return None

    dataset = []

    with parallel_backend('threading', n_jobs=cpu_count()-1):
        out = Parallel()(delayed(calc)(idx, df=pred_df) for idx in tqdm(pred_df.index))
        for ds in out:
            if ds:
                dataset.append(ds)

    print('Len dataset:', len(dataset))
    pickle.dump(dataset, open(output_path, 'wb'), protocol=-1)

def create( data_dir,
            smiles_column='smiles',
            protein_column='protein',
            label_column='activity',
            is_shuffle=True,
            max_mol_size=150,
            nfold=10,
            featd = {}
            ):

    dataset = []

    data_path        = data_dir + os.sep + 'data.txt'
    output_path      = data_dir + os.sep + 'data.pkl'
    test_path        = data_dir + os.sep + 'test.txt'
    test_data_path   = data_dir + os.sep + 'test.pkl'
    split_index_path = data_dir + os.sep + 'split_index.pkl'

    sep = ' '
    if data_path.endswith('csv'):
        sep = ','

    def calc(idx, df):
        try:
            return calc_single(df, idx, smiles_column, protein_column, label_column, featd=featd)
        except Exception as e:
            print('\nError processing idx %s: %s' %(idx, e))
            print(df.loc[idx])
            return None

    # train and dev datasets
    if os.path.exists(data_path):
        if os.path.isdir(data_path):
            print('Processing %s dir' %data_path)
            df1 = pd.read_csv(data_path + os.sep + 'train.txt', sep=sep)
            df2 = pd.read_csv(data_path + os.sep + 'dev.txt', sep=sep)

            df1idx = list(df1.index)
            df2idx = list(df2.index + df1.shape[0])

            if is_shuffle:
                print('Shuffle train and dev data index')
                np.random.shuffle(df1idx)
                np.random.shuffle(df2idx)

            pickle.dump([[df1idx, df2idx]], open(split_index_path, 'wb'))

            train_df = pd.concat([df1, df2], ignore_index=True)
        else:
            print('Processing %s file' %data_path)
            train_df = pd.read_csv(data_path, sep=sep)

        with parallel_backend('threading', n_jobs=cpu_count()-1):
            out = Parallel()(delayed(calc)(idx, df=train_df) for idx in tqdm(train_df.index))
            for ds in out:
                if ds:
                    dataset.append(ds)

        print('Len dataset:', len(dataset))
        print('Processing dataset finished.\nSpliting to %s fold and saving split index to file\n' %nfold)

        if not os.path.exists(split_index_path):
            print('Spliting index using KFold.split...')
            kf = KFold(n_splits=nfold, shuffle=is_shuffle, random_state=1000)
            itero = list(kf.split(dataset))
            pickle.dump(itero, open(split_index_path, 'wb'))
        else:
            print('Split file %s already exists.' %split_index_path)

        print('Saving train and dev datasets...')
        pickle.dump(dataset, open(output_path, 'wb'), protocol=-1)

    # test dataset
    test_fn = ''
    for fn in [test_path, data_path + os.sep + 'test.txt']:
        if os.path.exists(fn):
            test_fn = fn
            break

    if test_fn:
        test_dataset = []
        df3 = pd.read_csv(test_fn, sep=sep)
        print('Processing test dataset...')
        with parallel_backend('threading', n_jobs=cpu_count()-1):
            out = Parallel()(delayed(calc)(idx, df=df3) for idx in tqdm(df3.index))
            for ds in out:
                if ds:
                    test_dataset.append(ds)

        if is_shuffle:
            print('Shuffle test data')
            np.random.shuffle(test_dataset)

        print('Processing test dataset finished.\nLen test dataset:', len(test_dataset))
        print('Saving test dataset...')
        pickle.dump(test_dataset, open(test_data_path, 'wb'), protocol=-1)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-dd', '--data_dir', type=str, required=False, default='data/celegans', help='The data path')
    parser.add_argument('-sc', '--smiles_column', type=str, required=False, default='smiles', help='The smile column name')
    parser.add_argument('-pc', '--protein_column', type=str, required=False, default='protein', help='The protein column name')
    parser.add_argument('-lc', '--label_column', type=str, required=False, default='activity', help='The label column name')
    parser.add_argument('-sf', '--shuffle', type=bool, required=False, default=True, help='Should we shuffle the data')
    parser.add_argument('-mms', '--max_mol_size', type=int, required=False, default=150, help='The max molecular size')
    parser.add_argument('-nf', '--nfold', type=int, required=False, default=5, help='The split fold')
    parser.add_argument('-fc', '--feat_conf', type=str, required=False, default='', help='The featurizer config file')
    args = parser.parse_args()

    prot.load_word_dict()

    featd = {
            'addh': True,
            'calc_desc': False,
            'with_ring_conj': True,
            'with_atom_feats': True,
            'with_submol_fp': False,
            'with_subgraph_fp': False,
            'nbits': 512,
            'use_morgan': True,
            'use_macc': True,
            'use_rdkit': False,
            'prot_cover': 1,
            'prot_n': 3
    }

    try:
        d = json.load(open(args.feat_conf))
        featd.update(d)
        print('Use feat conf from file %s.' %args.feat_conf)
    except:
        pass

    create( args.data_dir,
            args.smiles_column,
            args.protein_column,
            args.label_column,
            args.shuffle,
            args.max_mol_size,
            args.nfold,
            featd
            )

    prot.dump_word_dict()