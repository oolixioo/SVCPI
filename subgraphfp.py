from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors as rdDesc
from collections import defaultdict
import numpy as np
import os, pickle, hashlib

AllChem.SetPreferCoordGen(True)

FINGERPRINT_DICT = defaultdict(lambda : len(FINGERPRINT_DICT))

ELEMENTS = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al',
            'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn',
            'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb',
            'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In',
            'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm',
            'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta',
            'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At',
            'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
            'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt',
            'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

for e in ELEMENTS:
    FINGERPRINT_DICT[e]

if os.path.exists('rdkit_fingerprint_list_r1.pkl'):
    l = pickle.load(open('rdkit_fingerprint_list_r1.pkl', 'rb'))

    for smi in l:
        FINGERPRINT_DICT[smi]

    print('Len fingerprint_list: %s' %len(FINGERPRINT_DICT)) + len(ELEMENTS)

def mol_with_atom_index(mol):
    atoms = mol.GetNumAtoms()
    for idx in range(atoms):
        mol.GetAtomWithIdx(idx).SetProp('molAtomMapNumber', str(mol.GetAtomWithIdx(idx).GetIdx()))
    return mol

def prepare_mol_for_drawing(mol):
    try:
        mol_draw = Draw.rdMolDraw2D.PrepareMolForDrawing(mol)
    except Chem.KekulizeException:
        mol_draw = Draw.rdMolDraw2D.PrepareMolForDrawing(mol, kekulize=False)
        Chem.SanitizeMol(mol_draw, Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
    return mol_draw

def get_atom_submol_radn(mol, radius, sanitize=True):
    atoms = []
    submols = []
    #smis = []
    for atom in mol.GetAtoms():
        atoms.append(atom)
        r = radius
        while r > 0:
            try:
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, r, atom.GetIdx())
                amap={}
                submol = Chem.PathToSubmol(mol, env, atomMap=amap)
                if sanitize:
                    Chem.SanitizeMol(submol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                #smis.append(Chem.MolToSmiles(submol))
                submols.append(submol)
                break
            except Exception as e:
                print(64, e)
                r -= 1

    return atoms, submols #, smis

def gen_fps_from_mol(mol, nbits=128, use_morgan=True, use_macc=True, use_rdkit=False):
    # morgan
    fp = []
    if use_morgan:
        fp_vec = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nbits)
        fp1 = np.frombuffer(fp_vec.ToBitString().encode(), 'u1') - ord('0')
        fp = fp1.tolist()
    if use_macc:
        # MACCSkeys
        fp_vec = MACCSkeys.GenMACCSKeys(mol)
        fp1 = np.frombuffer(fp_vec.ToBitString().encode(), 'u1') - ord('0')
        fp.extend(fp1.tolist())
    if use_rdkit:
        fp_vec = Chem.RDKFingerprint(mol)
        fp1 = np.frombuffer(fp_vec.ToBitString().encode(), 'u1') - ord('0')
        fp.extend(fp1.tolist())

    return fp

def gen_subgraph_fps_from_str(s, wordsdict={}):
    if s in wordsdict:
        return [wordsdict[s]]
    else:
        return [len(wordsdict)]

def gen_subgraph_fps_from_mol(mol, wordsdict={}):
    try:
        k = Chem.MolToSmiles(mol)
        return gen_subgraph_fps_from_str(k, wordsdict)
    except Exception as e:
        print(e)
        return [len(wordsdict)]

def calc_subgraph_fps_from_mol(mol, radius=2, nbits=128, use_macc=True, fptype=1, wordsdict={}):
    #atoms, submols, smis = get_atom_submol_radn(mol, radius, True)
    atoms, submols = get_atom_submol_radn(mol, radius, True)
    feats = []
    for idx, submol in enumerate(submols):
        if fptype == 1:
            feat = gen_fps_from_mol(submol, nbits, use_macc)
            feats.append(feat)
        elif fptype == 2:
            feat = gen_subgraph_fps_from_mol(submol, wordsdict)
            feats.append(feat)

    return np.array(feats)

if __name__ == '__main__':
    smi = 'C=C(S)C(N)(O)C'
    smi = 'CC1CCN(CC1N(C)C2=NC=NC3=C2C=CN3)C(=O)CC#N'

    mol = Chem.MolFromSmiles(smi, sanitize=False)

    print(calc_subgraph_fps_from_mol(mol, 3))

    mol = mol_with_atom_index(mol)
    submols = get_atom_submol_radn(mol, 3)
    submols = [prepare_mol_for_drawing(m) for m in submols]
    hl = []
    for idx, m in enumerate(submols):
        for a in m.GetAtoms():
            if int(a.GetProp('molAtomMapNumber')) == idx:
                hl.append([a.GetIdx()])
                break

    draw = Draw.MolsToGridImage([mol] + submols, highlightAtomLists=[[]] + hl, molsPerRow=5)
    draw.show()

