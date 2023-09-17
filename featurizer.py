import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors as rdDesc
from utils import *
import torch
import copy
import descripters as desc
import subgraphfp as subfp

PERIODIC_TABLE  = Chem.GetPeriodicTable()
POSSIBLE_ATOMS  = ['H', 'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br','I', 'B']
HYBRIDS         = [ Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2]
CHIRALS         = [ Chem.rdchem.ChiralType.CHI_UNSPECIFIED, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW, Chem.rdchem.ChiralType.CHI_OTHER]
BOND_TYPES      = [ Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC ]

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]

    return list(map(lambda s: x == s, allowable_set))

def calc_atom_features_onehot(atom, feature):
    '''
    Method that computes atom level features from rdkit atom object
    '''
    atom_features  = one_of_k_encoding_unk(atom.GetSymbol(), POSSIBLE_ATOMS)
    atom_features += one_of_k_encoding_unk(atom.GetExplicitValence(), list(range(7)))
    atom_features += one_of_k_encoding_unk(atom.GetImplicitValence(), list(range(7)))
    atom_features += one_of_k_encoding_unk(atom.GetTotalNumHs(), list(range(5)))
    atom_features += one_of_k_encoding_unk(atom.GetNumRadicalElectrons(), list(range(5)))
    atom_features += one_of_k_encoding_unk(atom.GetTotalDegree(), list(range(7)))
    atom_features += one_of_k_encoding_unk(atom.GetFormalCharge(), list(range(-2, 3)))
    atom_features += one_of_k_encoding_unk(atom.GetHybridization(), HYBRIDS)
    atom_features += one_of_k_encoding_unk(atom.GetIsAromatic(), [False, True])
    atom_features += one_of_k_encoding_unk(atom.IsInRing(), [False, True])
    atom_features += one_of_k_encoding_unk(atom.GetChiralTag(), CHIRALS)
    atom_features += one_of_k_encoding_unk(atom.HasProp('_CIPCode'), ['R', 'S'])
    atom_features += [PERIODIC_TABLE.GetRvdw(atom.GetSymbol())]
    atom_features += [atom.HasProp('_ChiralityPossible')]
    atom_features += [atom.GetAtomicNum()]
    atom_features += [atom.GetMass() * 0.01]
    atom_features += [atom.GetDegree()]
    atom_features += [int(i) for i in list('{0:06b}'.format(feature))]

    return atom_features

def calc_adjacent_tensor(bonds, atom_num, with_ring_conj=False):
    '''
    Method that constructs a AdjecentTensor with many AdjecentMatrics
    :param bonds: bonds of a rdkit mol
    :param atom_num: the atom number of the rdkit mol
    :param with_ring_conj: should the AdjecentTensor contains bond in ring and
        is conjugated info
    :return: AdjecentTensor A shaped [N, F, N], where N is atom number and F is bond types
    '''
    bond_types = len(BOND_TYPES)
    if with_ring_conj:
        bond_types += 2

    A = np.zeros([atom_num, bond_types, atom_num])

    for bond in bonds:
        b, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        try:
            bond_type = BOND_TYPES.index(bond.GetBondType())
            A[b, bond_type, e] = 1
            A[e, bond_type, b] = 1
            if with_ring_conj:
                if bond.IsInRing():
                    A[b, bond_types-2, e] = 1
                    A[e, bond_types-2, b] = 1
                if bond.GetIsConjugated():
                    A[b, bond_types-1, e] = 1
                    A[e, bond_types-1, b] = 1
        except:
            pass
    return A

def calc_data_from_smile(smiles, addh=False, calc_desc=False, with_ring_conj=False, with_atom_feats=True, with_submol_fp=False, with_subgraph_fp=False, wordsdict={}, radius=2):
    '''
    Method that constructs the data of a molecular.
    :param smiles: SMILES representation of a molecule
    :param addh: should we add all the Hs of the mol
    :param with_ring_conj: should the AdjecentTensor contains bond in ring and
        is conjugated info
    :return: V, A, global_state, mol_size, subgraph_size
    '''
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    #mol.UpdatePropertyCache(strict=False)

    if addh:
        mol = Chem.AddHs(mol)
    else:
        mol = Chem.RemoveHs(mol, sanitize=False)

    mol_size = torch.IntTensor([mol.GetNumAtoms()])
    subgraph_size = torch.IntTensor([mol.GetNumAtoms()])

    V = []

    if with_atom_feats:
        features = rdDesc.GetFeatureInvariants(mol)

    submoldict = {}
    if with_submol_fp or with_subgraph_fp:
        atoms, submols = subfp.get_atom_submol_radn(mol, radius, sanitize=True)
        submoldict = dict(zip([a.GetIdx() for a in atoms], submols))
        submoldict2 = None
        if radius >= 2:
            atoms, submols = subfp.get_atom_submol_radn(mol, radius-1, sanitize=True)
            submoldict2 = dict(zip([a.GetIdx() for a in atoms], submols))

    for i in range(mol.GetNumAtoms()):
        atom_i = mol.GetAtomWithIdx(i)
        if with_atom_feats:
            atom_i_features = calc_atom_features_onehot(atom_i, features[i])
        else:
            atom_i_features = []

        if with_submol_fp:
            submol = submoldict[i]
            #print(Chem.MolToSmiles(submol))
            submolfp = subfp.gen_fps_from_mol(submol)
            atom_i_features.extend(submolfp)

        if with_subgraph_fp:
            submol = submoldict[i]
            subgraphfp = subfp.gen_subgraph_fps_from_mol(submol, wordsdict)

            if subgraphfp[0] == len(wordsdict):
                submol = submoldict2[i]
                subgraphfp = subfp.gen_subgraph_fps_from_mol(submol, wordsdict)

                if subgraphfp[0] == len(wordsdict):
                    subgraphfp = subfp.gen_subgraph_fps_from_str(atom_i.GetSymbol(), wordsdict)

            atom_i_features.extend(subgraphfp)

        V.append(atom_i_features)

    V = torch.FloatTensor(V)

    A = calc_adjacent_tensor(mol.GetBonds(), mol.GetNumAtoms(), with_ring_conj)
    A = torch.FloatTensor(A)

    if calc_desc:
        global_state = torch.FloatTensor(desc.CalcRdkitDescriptors(mol))
    else:
        global_state = None

    
    return {'V': V, 'A': A, 'G': global_state, 'mol_size': mol_size, 'subgraph_size': subgraph_size}