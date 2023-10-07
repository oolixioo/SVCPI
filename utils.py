import torch
import protein

protein.load_word_dict()

EE_DICT = {}

def gather(x, indices):
    indices = indices.view(-1, indices.shape[-1]).tolist()
    out = torch.cat([x[i] for i in indices])

    return out

def gather_nd(x, indices):
    newshape = indices.shape[:-1] + x.shape[indices.shape[-1]:]
    indices = indices.view(-1, indices.shape[-1]).tolist()
    out = torch.cat([x[tuple(i)] for i in indices])

    return out.reshape(newshape)

def gen_node_indices(size_list):
    '''generate node index for extraction of nodes of each graph from batched data'''
    node_num = []
    node_range = []
    size_list = [int(i) for i in size_list]
    for i, n in enumerate(size_list):
        node_num.extend([i]*n)
        node_range.extend(list(range(n)))

    node_num = torch.tensor(node_num)
    node_range = torch.tensor(node_range)
    indices = torch.stack([node_num, node_range], axis=1)
    return indices, node_num, node_range

def segment_max(x, size_list):
    size_list = [int(i) for i in size_list]
    return torch.stack([torch.max(v, 0).values for v in torch.split(x, size_list)])

def segment_sum(x, size_list):
    size_list = [int(i) for i in size_list]
    return torch.stack([torch.sum(v, 0) for v in torch.split(x, size_list)])

def segment_softmax(gate, size_list):
    segmax = segment_max(gate, size_list)
    # expand segmax shape to alpha shape
    segmax_expand = torch.cat([segmax[i].repeat(n,1) for i,n in enumerate(size_list)], dim=0)
    subtract = gate - segmax_expand
    exp = torch.exp(subtract)
    segsum = segment_sum(exp, size_list)
    # expand segmax shape to alpha shape
    segsum_expand = torch.cat([segsum[i].repeat(n,1) for i,n in enumerate(size_list)], dim=0)
    attention = exp / (segsum_expand + 1e-16)

    return attention

def pad_V(V, max_n):
    N, C = V.shape
    if max_n > N:
        zeros = torch.zeros(max_n-N, C)
        V = torch.cat([V, zeros], dim=0)
    return V

def pad_A(A, max_n):
    N, L, _ = A.shape
    if max_n > N:
        zeros = torch.zeros(N, L, max_n-N)
        A = torch.cat([A, zeros], dim=-1)
        zeros = torch.zeros(max_n-N, L, max_n)
        A = torch.cat([A, zeros], dim=0)

    return A

def pad_prot(P, max_n):
    N, = P.shape
    if max_n > N:
        zeros = torch.zeros(max_n-N)
        P = torch.cat([P, zeros], dim=0)

    return P.type(torch.IntTensor)

def create_batch(input, pad=False, device=torch.device('cpu')):
    vl = []
    al = []
    gsl = []
    msl = []
    ssl = []
    lbl = []
    idxs = []
    smis = []

    for d in input:
        vl.append(d['V'])
        al.append(d['A'])
        gsl.append(d['G'])
        msl.append(d['mol_size'])
        ssl.append(d['subgraph_size'])
        lbl.append(d['label'])
        idxs.append(d['index'])
        smis.append(d['smiles'])

    if gsl[0] is not None:
        gsl = torch.stack(gsl, dim=0).to(device)

    if pad:
        max_n = max(map(lambda x:x.shape[0], vl))
        vl1 = []
        for v in vl:
            vl1.append(pad_V(v, max_n))
        al1 = []
        for a in al:
            al1.append(pad_A(a, max_n))

        return {'V': torch.stack(vl1, dim=0).to(device),
                'A': torch.stack(al1, dim=0).to(device),
                'G': gsl,
                'mol_size': torch.cat(msl, dim=0).to(device), 
                'subgraph_size': torch.stack(ssl, dim=0).to(device),
                'label': torch.stack(lbl, dim=0).to(device),
                'index': idxs,
                'smiles': smis}
    
    return {'V': torch.stack(vl, dim=0).to(device),
            'A': torch.stack(al, dim=0).to(device),
            'G': gsl,
            'mol_size': torch.cat(msl, dim=0).to(device),
            'subgraph_size': torch.stack(ssl, dim=0).to(device),
            'label': torch.stack(lbl, dim=0).to(device),
            'index': idxs,
            'smiles': smis}

def create_mol_protein_batch(input, pad=False, device=torch.device('cpu'), pr=True, ee=0):
    vl = []
    al = []
    gsl = []
    msl = []
    ssl = []
    prot = []
    seq = []
    lbl = []
    idxs = []
    smis = []
    fpl = []

    for n, d in enumerate(input):
        vl.append(d['V'])
        al.append(d['A'])
        gsl.append(d['G'])
        msl.append(d['mol_size'])
        ssl.append(d['subgraph_size'])
        prot.append(d['protein_seq'])
        if 'protein' in d:
            seq.append(d['protein'])
        else:
            seq.append('')
        lbl.append(d['label'])
        idxs.append(d['index'])
        smis.append(d['smiles'])
        if 'fp' in d:
            fpl.append(d['fp'])

        if ee > 0 and n % ee == 0:
            for l in [vl, al, gsl, msl, ssl, seq, idxs, smis, fpl]:
                try:
                    l.append(l[-1])
                except:
                    pass
                    
            if d['index'] in EE_DICT:
                t = EE_DICT[d['index']]
                prot.append(t)
                lbl.append(torch.LongTensor([0]))
            else:
                idx = torch.randperm(d['protein_seq'].nelement()-2) + 1
                t = torch.cat([torch.IntTensor([protein.WORD_DICT['^^^']]), d['protein_seq'][idx], torch.IntTensor([protein.WORD_DICT['$$$']])])
                prot.append(t)
                lbl.append(torch.LongTensor([0]))
                EE_DICT[d['index']] = t

    if gsl[0] is not None:
        if pad:
            gsl = torch.stack(gsl, dim=0).to(device)
        else:
            gsl = [torch.unsqueeze(g, 0) for g in gsl]

    if pad:
        max_n = max(map(lambda x:x.shape[0], vl))
        vl1 = []
        if pr:
            print('\tPadding V to max_n:', max_n)
        for v in vl:
            vl1.append(pad_V(v, max_n))

        al1 = []
        if pr:
            print('\tPadding A to max_n:', max_n)
        for a in al:
            al1.append(pad_A(a, max_n))

        max_prot = max(map(lambda x:x.shape[0], prot))
        prot1 = []
        if pr:
            print('\tPadding protein_seq to max_n:', max_prot)
        for p in prot:
            prot1.append(pad_prot(p, max_prot))

        fpt = None
        if fpl:
            fpt = torch.stack(fpl, dim=0).to(device)

        return {'V': torch.stack(vl1, dim=0).to(device),
                'A': torch.stack(al1, dim=0).to(device),
                'G': gsl,
                'fp': fpt,
                'mol_size': torch.cat(msl, dim=0).to(device), 
                'subgraph_size': torch.stack(ssl, dim=0).to(device),
                'protein_seq': torch.stack(prot1, dim=0).to(device),
                'label': torch.stack(lbl, dim=0).view(-1).to(device),
                'index': idxs,
                'smiles': smis,
                'protein': seq}
    
    return {'V': [torch.unsqueeze(v, 0) for v in vl],
            'A': [torch.unsqueeze(a, 0) for a in al],
            'G': gsl,
            'fp': fpt,
            'mol_size': torch.cat(msl, dim=0).to(device),
            'subgraph_size': [torch.unsqueeze(s, 0) for s in ssl],
            'protein_seq': [torch.unsqueeze(p, 0) for p in prot],
            'label': torch.stack(lbl, dim=0).view(-1).to(device),
            'index': idxs,
            'smiles': smis,
            'protein': seq}

def create_mol_protein_fp_batch(input, pad=False, device=torch.device('cpu'), pr=True):
    fp = []
    prot = []
    lbl = []
    idxs = []
    smis = []

    for d in input:
        fp.append(d['fp'])
        prot.append(d['protein_seq'])
        lbl.append(d['label'])
        idxs.append(d['index'])
        smis.append(d['smiles'])

    if pad:
        max_prot = max(map(lambda x:x.shape[0], prot))
        prot1 = []
        if pr:
            print('\tPadding protein_seq to max_n:', max_prot)
        for p in prot:
            prot1.append(pad_prot(p, max_prot))

        return {'fp': torch.stack(fp, dim=0).to(device),
                'protein_seq': torch.stack(prot1, dim=0).to(device),
                'label': torch.stack(lbl, dim=0).view(-1).to(device),
                'index': idxs,
                'smiles': smis}
    
    return {'fp':          [torch.unsqueeze(f, 0) for f in fp],
            'protein_seq': [torch.unsqueeze(p, 0) for p in prot],
            'label': torch.stack(lbl, dim=0).view(-1).to(device),
            'index': idxs,
            'smiles': smis}