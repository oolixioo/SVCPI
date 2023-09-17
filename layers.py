import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils
import pickle

DEVICE = torch.cuda.is_available() and torch.device('cuda') or torch.device('cpu')

class GraphCNNLayer(nn.Module):
    def __init__(self, n_feats, adj_chans=4, n_filters=64, bias=True):
        super(GraphCNNLayer, self).__init__()
        self.n_feats = n_feats
        self.adj_chans = adj_chans
        self.n_filters = n_filters
        self.has_bias = bias

        # [C*L, F], C = n_feats, L = adj_chans, F = n_filters; this is for the edge feats 
        self.weight_list = nn.ParameterList([nn.Parameter(torch.FloatTensor(n_feats, n_filters)) for _ in range(adj_chans)])
        # [C, F], this is for 𝐈𝐕in𝐖0
        self.weight_i = nn.Parameter(torch.FloatTensor(n_feats, self.n_filters))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(n_filters))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        for w in self.weight_list:
            nn.init.xavier_uniform_(w)

        nn.init.xavier_uniform_(self.weight_i)

        if self.bias is not None:
            self.bias.data.fill_(0.01)

    def forward(self, V, A):
        '''V node features: [b, N, C], A adjs: [b, N, L, N], L = adj_chans'''
        b, N, C = V.shape
        b, N, L, _ = A.shape

        # formula: 𝐕out = 𝐈𝐕in𝐖0 + GConv(𝐕in, 𝐹) + 𝐛; 𝐈𝐕in = 𝐕in, so 𝐈𝐕in𝐖0 = 𝐕in𝐖0
        
        # [b, N, C] * [C, F] -> [b, N, F]
        output = torch.matmul(V, self.weight_i)

        for i in range(self.adj_chans):
            a = A[:, :, i, :]
            a = a.view(-1, N, N)
            # [b, N, N] * [b, N, C] -> [b, N, C]
            n = torch.bmm(a, V)
            # [b, N, C] * [C, F] -> [b, N, F]
            output += torch.matmul(n, self.weight_list[i])

        if self.has_bias:
            output += self.bias

        # output: [b, N, F]
        return output

    def __repr__(self):
        return f'{self.__class__.__name__}(n_feats={self.n_feats},adj_chans={self.adj_chans},n_filters={self.n_filters},bias={self.has_bias}) -> [b, N, {self.n_filters}]'

class GraphResidualCNNLayer(nn.Module):
    def __init__(self, n_feats, adj_chans=4, bias=True):
        super(GraphResidualCNNLayer, self).__init__()
        self.n_feats = n_feats
        self.adj_chans = adj_chans
        self.has_bias = bias

        # [C*L, F], C = n_feats, L = adj_chans 
        self.weight_layers = nn.ModuleList([nn.Linear(n_feats, n_feats) for _ in range(adj_chans)])

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(n_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            self.bias.data.fill_(0.01)

    def forward(self, V, A):
        '''V node features: [b, N, C], A adjs: [b, N, L, N], L = adj_chans'''
        b, N, C = V.shape
        b, N, L, _ = A.shape

        for i in range(self.adj_chans):
            # [b, N, C] -> [b, N, C]
            hs = torch.relu(self.weight_layers[i](V))
            # [b, N, N]
            a = A[:, :, i, :]
            a = a.view(-1, N, N)
            # [b, N, N] * [b, N, C] -> [b, N, C]
            V = V + torch.bmm(a, hs)

        if self.has_bias:
            V += self.bias

        # output: [b, N, C]
        return V

    def __repr__(self):
        return f'{self.__class__.__name__}(n_feats={self.n_feats},adj_chans={self.adj_chans},bias={self.has_bias}) -> [b, N, {self.n_feats}]'

class GraphAttentionLayer(nn.Module):
    def __init__(self, n_feats, adj_chans=4, n_filters=64, bias=True, dropout=0., alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.n_feats = n_feats
        self.adj_chans = adj_chans
        self.n_filters = n_filters
        self.has_bias = bias
        self.dropout = dropout
        self.alpha = alpha

        # [C*L, F], C = n_feats, L = adj_chans, F = n_filters; this is for the edge feats 
        self.weight_list = nn.ParameterList([nn.Parameter(torch.FloatTensor(n_feats, n_filters)) for _ in range(adj_chans)])
        self.a1_list     = nn.ParameterList([nn.Parameter(torch.FloatTensor(n_filters, 1)) for _ in range(adj_chans)])
        self.a2_list     = nn.ParameterList([nn.Parameter(torch.FloatTensor(n_filters, 1)) for _ in range(adj_chans)])

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(n_filters))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        for w in self.weight_list:
            nn.init.xavier_uniform_(w)
        for w in self.a1_list:
            nn.init.xavier_uniform_(w)
        for w in self.a2_list:
            nn.init.xavier_uniform_(w)
        if self.bias is not None:
            self.bias.data.fill_(0.01)

    def forward(self, V, A):
        '''V node features: [b, N, C], A adjs: [b, N, L, N], L = adj_chans'''
        b, N, C = V.shape
        b, N, L, _ = A.shape

        output = None

        # formula: 𝐕out = 𝐈𝐕in𝐖0 + GConv(𝐕in, 𝐹) + 𝐛; 𝐈𝐕in = 𝐕in, so 𝐈𝐕in𝐖0 = 𝐕in𝐖0
        for i in range(self.adj_chans):
            # [b, N, 1, N] -> [b, N, N]
            adj = A[:, :, i, :].view(-1, N, N)

            # [b, N, C] * [C, F] -> [b, N, F]
            h = torch.matmul(V, self.weight_list[i])
            # [b, N, F] * [F, 1] -> [b, N, 1]
            f_1 = torch.matmul(h, self.a1_list[i])
            # [b, N, F] * [F, 1] -> [b, N, 1]
            f_2 = torch.matmul(h, self.a2_list[i])

            # leaky_relu([b, N, 1] + [b, 1, N]) -> [b, N, N]
            e = F.leaky_relu(f_1 + f_2.transpose(1, 2), self.alpha)

            zero_vec = -9e15 * torch.ones_like(e)
            # [b, N, N]
            att = torch.where(adj > 0, e, zero_vec)
            att = F.softmax(att, dim=1)
            att = F.dropout(att, self.dropout, training=self.training)
            # [b, N, N] * [b, N, F] -> [b, N, F]
            if output is None:
                output = torch.matmul(att, h)
            else:
                output += torch.matmul(att, h)

        if self.has_bias:
            output += self.bias

        # output: [b, N, F]
        return output

    def __repr__(self):
        return f'{self.__class__.__name__}(n_feats={self.n_feats},adj_chans={self.adj_chans},n_filters={self.n_filters},bias={self.has_bias},dropout={self.dropout},alpha={self.alpha}) -> [b, N, {self.n_filters}]'

class GraphNodeCatGlobalFeatures(nn.Module):
    def __init__(self, global_feats, out_feats, mols=1, bias=True):
        super(GraphNodeCatGlobalFeatures, self).__init__()
        self.global_feats = global_feats
        self.out_feats = out_feats
        self.mols = mols
        self.has_bias = bias

        self.weights = nn.ParameterList([nn.Parameter(torch.FloatTensor(int(global_feats/mols), out_feats)) for _ in range(mols)])

        self.biass = []
        if bias:
            self.biass = nn.ParameterList([nn.Parameter(torch.FloatTensor(out_feats)) for _ in range(mols)])
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.weights:
            nn.init.xavier_uniform_(weight)
        for bias in self.biass:
            bias.data.fill_(0.01)

    def forward(self, V, global_state, graph_size, subgraph_size=None):
        # V: [b, N, Ov], global_state: [b, F], subgraph_size: [b, mols]
        b, N, Ov = V.shape
        O = self.out_feats
        if self.mols == 1:
            subgraph_size = graph_size.view(-1, 1)
            global_state = torch.mm(global_state, self.weights[0])
        else:
            # global_state: [b, F] view -> [b*mols, F/mols]
            global_state_view = global_state.view(b*self.mols, -1)

            # split global_state into that of individual mols
            idxmols = []
            for i in range(self.mols):
                idxmols.append(torch.IntTensor(list(range(i, b*self.mols, self.mols))).to(self.weights[0].device))

            global_states = []
            for i, idx in enumerate(idxmols):
                # selected global_state of mols from global_state_view [b*mols, F/mols]. Out shape is [b, F/mols]
                gs = global_state_view.index_select(dim=0, index=idx)
                # gs: [b, F/mols] * weight: [F/mols, O] -> [b, O]; F = global_feats, O = out_feats
                gs = torch.mm(gs, self.weights[i])

                if self.has_bias:
                    gs += self.biass[i]

                global_states.append(F.relu(gs))

            # convert global_states back to global_state
            # [[b, O] ... ] -> [b, mols*O]
            global_state = torch.cat(global_states, dim=1)

        # [b, mols*O] || [b, O] -> [b, (mols+1)*O]
        global_state_new = torch.cat([global_state, torch.zeros(b, O).to(self.weights[0].device)], dim=-1)
        # [b*(mols+1), O]
        global_state_new = global_state_new.view(-1, O)

        repeats = []
        for sz in subgraph_size:
            repeats.extend(sz.tolist() + [N-sz.sum()])
        repeats = torch.tensor(repeats).to(self.weights[0].device)

        # repeat form [b*(mols+1), O] -> [b*N, O], the content like [m1_feats, m2_feats, ... mn_feats, pads, ...]
        global_state_new = global_state_new.repeat_interleave(repeats, dim=0)

        # V view: [b*N, Ov], global_state_new: [b*N, O]
        output = torch.cat([V.contiguous().view(-1, Ov), global_state_new], dim=1)

        # output: [b, N, Ov+O]
        return output.view(-1, N, Ov+O), global_state

    def __repr__(self):
        return f'{self.__class__.__name__}(global_feats={self.global_feats},out_feats={self.out_feats},bias={self.has_bias}) -> [b, N, {self.global_feats+self.out_feats}], [b, out_feats]'

class MultiHeadGlobalAttention(nn.Module):
    '''Input [b, N, C] -> output [b, n_head*C] if concat or else [b, n_head]'''
    def __init__(self, n_feats, n_head=5, alpha=0.2, concat=True, bias=True):
        super(MultiHeadGlobalAttention, self).__init__()

        self.n_feats = n_feats
        self.n_head = n_head
        self.alpha = alpha
        self.concat = concat
        self.has_bias = bias

        self.weight = nn.Parameter(torch.FloatTensor(n_feats, n_head*n_feats))
        self.tune_weight = nn.Parameter(torch.FloatTensor(1, n_head, n_feats))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(n_head*n_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.tune_weight)
        if self.bias is not None:
            self.bias.data.fill_(0.01)

    def forward(self, V, graph_size):
        # Gather V of mols in a batch, after this, the pad was removed.
        #print(248, V.shape, graph_size)
        if V.shape[0] == 1:
            Vg = torch.squeeze(V)
            graph_size = [graph_size]
        else:
            Vg = torch.cat([torch.split(v.view(-1, v.shape[-1]), graph_size[i])[0] for i,v in enumerate(torch.split(V, 1))], dim=0)

        Vg = torch.matmul(Vg, self.weight)
        if self.has_bias:
            Vg += self.bias
        Vg = Vg.view(-1, self.n_head, self.n_feats)

        alpha = torch.mul(self.tune_weight, Vg)
        alpha = torch.sum(alpha, dim=-1)
        alpha = F.leaky_relu(alpha, self.alpha) # original code is "alpha = tf.nn.leaky_relu(alpha, alpha=0.2)"
        alpha = utils.segment_softmax(alpha, graph_size)

        #alpha_collect = torch.mean(alpha, dim=-1) # origin code like this. alpha_collect not used?
        alpha = alpha.view(-1, self.n_head, 1)
        V = torch.mul(Vg, alpha)

        if self.concat:
            V = utils.segment_sum(V, graph_size)
            V = V.view(-1, self.n_head*self.n_feats)
        else:
            V = torch.mean(V, dim=1)
            V = utils.segment_sum(V, graph_size)

        return V

    def __repr__(self):
        if self.concat:
            outc = self.n_head*self.n_feats
        else:
            outc = self.n_head
        return f'{self.__class__.__name__}(n_feats={self.n_feats},n_head={self.n_head},alpha={self.alpha},concat={self.concat},bias={self.has_bias}) -> [b, {outc}]'

class GraphEmbedPoolingLayer(nn.Module):
    def __init__(self, n_feats, n_filters=1, mask=None, bias=True):
        super(GraphEmbedPoolingLayer, self).__init__()
        self.n_feats = n_feats
        self.n_filters = n_filters
        self.mask = mask
        self.has_bias = bias

        self.emb = nn.Linear(n_feats, n_filters, bias=bias)

    def forward(self, V, A):
        # [b, N, F]
        factors = self.emb(V)

        if self.mask is not None:
            factors = torch.mul(factors, self.mask)

        factors = F.softmax(factors, dim=1)
        # [b, N, F] trans -> [b, F, N] * [b, N, C] -> [b, F, C]
        result = torch.matmul(factors.transpose(1, 2).contiguous(), V)

        if self.n_filters == 1:
            return result.view(-1, self.n_feats), A

        result_A = A.view(A.shape[0], -1, A.shape[-1])
        result_A = torch.matmul(result_A, factors)
        result_A = result_A.view(A.shape[0], A.shape[-1], -1)
        result_A = torch.matmul(factors.transpose(1, 2).contiguous(), result_A)
        result_A = result_A.view(A.shape[0], self.n_filters, A.shape[2], self.n_filters)

        return result, result_A

    def __repr__(self):
        return f'{self.__class__.__name__}(n_feats={self.n_feats},n_filters={self.n_filters},mask={self.mask},bias={self.has_bias}) -> [b, {self.n_filters}, {self.n_feats}], [b, {self.n_filters}, L, {self.n_filters}]'

class GConvBlock(nn.Module):
    def __init__(   self,
                    n_feats,
                    n_filters,
                    global_feats,
                    global_out_feats,
                    mols=1,
                    adj_chans=4,
                    bias=True,
                    usegat=False):

        super(GConvBlock, self).__init__()

        self.n_feats = n_feats
        self.n_filters = n_filters
        self.global_out_feats = global_out_feats
        self.global_feats = global_feats
        self.mols = mols
        self.adj_chans = adj_chans
        self.has_bias = bias
        self.usegat = usegat

        self.broadcast_global_state = GraphNodeCatGlobalFeatures(global_feats, global_out_feats, mols, bias)
        if usegat:
            self.graph_conv = GraphAttentionLayer(n_feats+global_out_feats, adj_chans, n_filters)
        else:
            self.graph_conv = GraphCNNLayer(n_feats+global_out_feats, adj_chans, n_filters, bias)

        self.bn_global = nn.BatchNorm1d(global_out_feats*mols)
        self.bn_graph  = nn.BatchNorm1d(n_filters)

    def forward(self, V, A, global_state, graph_size, subgraph_size):
        ######## transfer global_state #########
        # V shape from [b, N, C] to [b, N, C+F], F is n_filters
        V, global_state = self.broadcast_global_state(V, global_state, graph_size, subgraph_size)

        ######## Graph Convolution #########
        # V shape from [b, N, C+F] to [b, N, F1], F1 is n_filters
        V = self.graph_conv(V, A)
        V = self.bn_graph(V.transpose(1, 2).contiguous())
        V = F.relu(V.transpose(1, 2))

        global_state = F.relu(self.bn_global(global_state))

        return V, global_state

    def __repr__(self):
        return f'{self.__class__.__name__}(n_feats={self.n_feats},n_filters={self.n_filters},global_feats={self.global_feats},global_out_feats={self.global_out_feats},mols={self.mols},adj_chans={self.adj_chans},bias={self.has_bias},usegat={self.usegat}) -> [b, N, {self.n_filters}], [b, {self.global_out_feats*self.mols}]'

class GConvBlock1(nn.Module):
    def __init__(   self,
                    n_feats,
                    n_filters,
                    mols=1,
                    adj_chans=4,
                    bias=True):

        super(GConvBlock1, self).__init__()

        self.n_feats = n_feats
        self.n_filters = n_filters
        self.mols = mols
        self.adj_chans = adj_chans
        self.has_bias = bias

        #self.graph_conv = GraphCNNLayer(n_feats+n_filters, adj_chans, n_filters, bias)
        self.graph_conv = GraphCNNLayer(n_feats, adj_chans, n_filters, bias)

        #self.bn_global = nn.BatchNorm1d(n_filters*mols)
        self.bn_graph  = nn.BatchNorm1d(n_filters)

    def forward(self, V, A):
        ######## Graph Convolution #########
        # V shape from [b, N, C+F] to [b, N, F1], F1 is n_filters
        V = self.graph_conv(V, A)
        V = self.bn_graph(V.transpose(1, 2).contiguous())
        V = F.relu(V.transpose(1, 2))

        return V

    def __repr__(self):
        return f'{self.__class__.__name__}(n_feats={self.n_feats},n_filters={self.n_filters},mols={self.mols},adj_chans={self.adj_chans},bias={self.has_bias}) -> [b, N, {self.n_filters}]' 