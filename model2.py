import enum
import math, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from sklearn.metrics import roc_auc_score
from layers import *
import protein as prot

DEVICE = torch.cuda.is_available() and torch.device('cuda') or torch.device('cpu')

class Net(nn.Module):
    def __init__(self,
        n_feats,
        adj_chans,
        fpdim = 512+167,
        n_filters_list = [256, 256, 256],
        mlp_layers = 3,
        n_head = 8,
        readout_layers = 1,
        mols = 1,
        dim = 10,
        bias = True,
        fp_linear_layers = 2,
        dropout_p=0.2
        ):
        super(Net, self).__init__()

        n_filters_list    = [i for i in n_filters_list if i is not None]

        n_word = 8600 # 8587

        l = []
        for i, nf in enumerate(n_filters_list):
            if i == 0:
                n_feats1 = n_feats
            else:
                n_feats1 = prevnf
            
            prevnf = nf

            ly = GConvBlock1(n_feats1, nf, mols, adj_chans, bias).to(DEVICE)
            l.append(ly)

        self.block_layers = nn.ModuleList(l)

        l = [nn.Linear(fpdim, dim, bias=bias)]
        l += [nn.Linear(dim, dim, bias=bias) for _ in range(fp_linear_layers - 1)]
        self.fp_linear_layers = nn.ModuleList(l)

        self.mol_attention_layer = MultiHeadGlobalAttention(nf, n_head=n_head, concat=True, bias=bias).to(DEVICE)
        self.readout_layers = nn.ModuleList([nn.Linear(nf*n_head, dim, bias=bias).to(DEVICE)] + [nn.Linear(dim, dim).to(DEVICE) for _ in range(readout_layers-1)])

        self.merge_feats_layer = nn.Linear(dim+dim, dim, bias=bias)
        self.prot_emb_layer  = nn.Embedding(n_word, dim)

        self.mol_feat_linear_fp_layer = nn.Linear(dim, dim, bias=bias)
        self.prot_emb_linear_fp_layer = nn.Linear(dim, dim, bias=bias)

        self.mol_feat_linear_v_layer = nn.Linear(dim, dim, bias=bias)
        self.prot_emb_linear_v_layer = nn.Linear(dim, dim, bias=bias)

        self.mlp_layers1 = nn.ModuleList([nn.Linear(dim*2, dim*2, bias=bias) for _ in range(mlp_layers)])
        self.mlp_layers2 = nn.ModuleList([nn.Linear(dim*2, dim*2, bias=bias) for _ in range(mlp_layers)])
        self.out_layer1  = nn.Linear(dim*2, 2, bias=bias)
        self.out_layer2  = nn.Linear(dim*2, 2, bias=bias)

        self.dropout     = nn.Dropout(p=dropout_p)
    
    def attention_cnn_fp(self, fp, prot_emb):
        # V: [b, dim], prot_emb: [b, N, dim]
        b, N, dim = prot_emb.shape

        # [b, dim] -> [b, dim]
        h = torch.relu(self.mol_feat_linear_fp_layer(fp))

        # [b, N, dim] -> [b, N, dim]
        hs = torch.relu(self.prot_emb_linear_fp_layer(prot_emb))

        ysl = []
        for i in range(h.shape[0]):
            # [b, dim], [b, N, dim]
            hi  = h[i].view(-1, dim)      # [1, dim]
            hsi = hs[i]                   # [N, dim]
            # matmul([1, dim], [dim, N]) -> [1, N]
            wi = torch.tanh(torch.matmul(hi, hsi.t()))
            # [1, N] -> [N, 1] * [N, dim] -> [N, dim]
            ysl.append(wi.t() * hsi)

        # [b, N, dim]
        ys = torch.stack(ysl, dim=0)

        # [b, dim]
        #return torch.mean(ys, dim=1)
        return torch.max(ys, dim=1)[0]

    def attention_cnn_v(self, V, prot_emb):
        # V: [b, dim], prot_emb: [b, N, dim]
        b, N, dim = prot_emb.shape

        # [b, dim] -> [b, dim]
        h = self.dropout(F.relu(self.mol_feat_linear_v_layer(V)))

        # [b, N, dim] -> [b, N, dim]
        hs = self.dropout(F.relu(self.prot_emb_linear_v_layer(prot_emb)))

        ysl = []
        for i in range(h.shape[0]):
            # [b, dim], [b, N, dim]
            hi  = h[i].view(-1, dim)      # [1, dim]
            hsi = hs[i]                   # [N, dim]
            # matmul([1, dim], [dim, N]) -> [1, N]
            wi = torch.tanh(torch.matmul(hi, hsi.t()))
            # [1, N] -> [N, 1] * [N, dim] -> [N, dim]
            ysl.append(wi.t() * hsi)

        # [b, N, dim]
        ys = torch.stack(ysl, dim=0)

        # [b, dim]
        #return torch.mean(ys, dim=1)
        return torch.max(ys, dim=1)[0]

    def forward(self, inputs):
        V               = inputs['V']
        A               = inputs['A']
        G               = inputs['G']
        fp              = inputs['fp']
        mol_size        = inputs['mol_size']
        subgraph_size   = inputs['subgraph_size']
        protein_seq     = inputs['protein_seq']
        labels          = inputs['label']

        for ly in self.fp_linear_layers:
            fp = self.dropout(F.relu(ly(fp)))

        for n, ly in enumerate(self.block_layers):
            V = self.dropout(ly(V, A))

        V = self.mol_attention_layer(V, mol_size)

        for ly in self.readout_layers:
            V = F.relu(ly(V))

        prot_emb  = self.prot_emb_layer(protein_seq)    # [b, N, dim] protein embedding

        itact1     = self.attention_cnn_fp(fp, prot_emb)   # [b, dim], fp and protein interaction
        itact2     = self.attention_cnn_v(V, prot_emb)    # [b, dim], V and protein interaction

        cat_mat1 = torch.cat([fp, itact1], dim=1)       # [b, dim*2]

        # [b, dim*2]
        for ly in self.mlp_layers1:
            cat_mat1 = self.dropout(F.relu(ly(cat_mat1)))
            
        cat_mat2 = torch.cat([V, itact2], dim=1)        # [b, dim*2]

        for ly in self.mlp_layers2:
            cat_mat2 = self.dropout(F.relu(ly(cat_mat2)))

        pred_itact1 = self.out_layer1(cat_mat1)         # [b, 2]
        pred_itact2 = self.out_layer2(cat_mat2)         # [b, 2]

        pred_itact = pred_itact1 + pred_itact2         # [b, 2]

        pred_labels = pred_itact.argmax(dim=1)
        pred_scores = F.softmax(pred_itact, 1)[:,1]     # scores of active prediction

        return pred_itact, labels, pred_labels, pred_scores