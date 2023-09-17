# -*- coding: utf-8 -*-
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors, rdFreeSASA, rdMolDescriptors
from openbabel import openbabel as ob
from openbabel import pybel
import numpy as np
import math


Rvdw = {'H': 1.2,'He': 1.4,'Li': 1.82,'Be': 1.7,'B': 2.08,'C': 1.95,'N': 1.85,
        'O': 1.7,'F': 1.73,'Ne': 1.54,'Na': 2.27,'Mg': 1.73,'Al': 2.05,'Si': 2.1,
        'P': 2.08,'S': 2.0,'Cl': 1.97,'Ar': 1.88,'Br': 2.1,'I': 2.15}


def coordinate_adjusting(mol):
    '''
    The algorithm refers from：http://sobereva.com/426
    '''
    mat_coor = [[0, 1], [0, 2], [1, 2]]
    diag_coor = [[1, 2], [0, 2], [0, 1]]
    atoms = mol.GetAtoms()
    AllChem.EmbedMolecule(mol)
    AllChem.UFFOptimizeMolecule(mol)
    atom_coors = np.array(
        [mol.GetConformer().GetAtomPosition(i.GetIdx()) for i in atoms])
    wts = np.array([i.GetMass() for i in atoms])
    wts = np.expand_dims(wts, axis=1)
    diag_val = [np.sum(wts*atom_coors[:, i]**2) for i in diag_coor]
    mat_val = [np.sum(wts*np.prod(atom_coors[:, i], axis=1))
               * -1 for i in mat_coor]
    imt = np.zeros([3, 3])
    for i in range(3):
        imt[i, i] = diag_val[i]
        pos = mat_coor[i]
        imt[pos[0], pos[1]] = mat_val[i]
        imt[pos[1], pos[0]] = mat_val[i]
    eig_v, eig_m = np.linalg.eig(imt)
    am = atom_coors.dot(eig_m)
    return am


def MaxMinValue(array):
    max_idx, max_val = np.argmax(array), np.max(array)
    min_idx, min_val = np.argmin(array), np.min(array)
    return int(max_idx), max_val, int(min_idx), min_val

def CalcuAxisLenth(mol):
    coors = coordinate_adjusting(mol)
    axis = []
    for i in range(3):
        max_idx, max_val, min_idx, min_val = MaxMinValue(coors[:,i])
        max_ = max_val + Rvdw[mol.GetAtomWithIdx(max_idx).GetSymbol()]
        min_ = min_val - Rvdw[mol.GetAtomWithIdx(min_idx).GetSymbol()]
        axis.append(max_-min_)
    S, M, L = sorted(axis)
    return S, M, L

def Ratio_S_M_L(mol):
    S, M, L = CalcuAxisLenth(mol)
    S_L = S/L
    M_L = M/L
    S_M = S/M
    return S_L, M_L, S_M, S

def GlobularityAndFrTPSA(mol, includeSandP=1):
    # Calculate globularity: surface of a sphere with the same volume as the molecule / Area
    # FrTPSA = TPSA / SASA
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    mol_vol = AllChem.ComputeMolVolume(mol,confId=0)
    r_sphere = math.pow(mol_vol*0.75/math.pi, 1.0/3)
    area_sphere = 4*math.pi*r_sphere*r_sphere
    radii = rdFreeSASA.classifyAtoms(mol)
    sasa = rdFreeSASA.CalcSASA(mol, radii)
    globularity = area_sphere / sasa
    FrTPSA = Descriptors.TPSA(mol, includeSandP=includeSandP) / sasa
    return globularity, FrTPSA

def FractionNO(mol):
    return Descriptors.NOCount(mol) / float(mol.GetNumHeavyAtoms())

def FractionAromaticAtoms(mol):
    return len(mol.GetAromaticAtoms()) / float(mol.GetNumHeavyAtoms())

def NumHAcceptorsAndDonors(mol):
    return Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol)

def RotatableBondNumber(mol):
    mol = Chem.RemoveHs(mol)  
    return Descriptors.NumRotatableBonds(mol) #/ float(mol.GetNumBonds())

# Using openbabel for calculating dipole moment
def DipoleMoment(mol, charge_model='eem2015bm'):
    mol_block = Chem.MolToMolBlock(mol)
    ob_mol = pybel.readstring('mol', mol_block)
    # We choose 'eem2015bm' to calculate dipole
    # Using 'obabel -L charges' can get a list of charge models
    dipole = ob.OBChargeModel_FindType(charge_model).GetDipoleMoment(ob_mol.OBMol)
    dipole_moment = math.sqrt(dipole.GetX()**2+dipole.GetY()**2+dipole.GetZ()**2)
    return dipole_moment

def CalcuDescriptors(mol, includeSandP=1, charge_model='eem2015bm'):
    S_L, M_L, S_M, S = Ratio_S_M_L(mol)
    #globularity, FrTPSA = GlobularityAndFrTPSA(mol, includeSandP=includeSandP)
    Fr_NO = FractionNO(mol)
    Fr_AromaticAtoms = FractionAromaticAtoms(mol)
    HBA,HBD = NumHAcceptorsAndDonors(mol)
    RBN = RotatableBondNumber(mol)
    dipole_moment = DipoleMoment(mol, charge_model=charge_model)
    #return np.array([S_L, M_L, S_M, S, globularity, FrTPSA, Fr_NO, Fr_AromaticAtoms, HBA, HBD, RBN, dipole_moment])
    return np.array([S_L, M_L, S_M, S, Fr_NO, Fr_AromaticAtoms, HBA, HBD, RBN, dipole_moment])

def GetMolDescriptors(mol, missingVal=None):
    ''' calculate the full list of descriptors for a molecule
    
        missingVal is used if the descriptor cannot be calculated
    '''
    desc = []
    for nm, fn in Descriptors._descList:
        # some of the descriptor fucntions can throw errors if they fail, catch those here:
        try:
            val = fn(mol)
        except:
            # print the error message:
            import traceback
            traceback.print_exc()
            # and set the descriptor value to whatever missingVal is
            val = missingVal
        desc.append(val)

    return np.array(desc)

def CalcRdkitDescriptors(mol):
    Chem.SanitizeMol(mol)
    AllChem.EmbedMolecule(mol)
    funcs = [
        Descriptors.HeavyAtomCount,
        Descriptors.NumAromaticRings,
        Descriptors.NumAliphaticRings,
        Descriptors.NumSaturatedRings,
        Descriptors.HallKierAlpha,
        Descriptors.Kappa1,
        Descriptors.Kappa2,
        Descriptors.Kappa3,
        Descriptors.Chi0v,
        Descriptors.Chi1v,
        Descriptors.Chi2v,
        Descriptors.Chi3v,
        Descriptors.Chi4v,
        Descriptors.Chi0n,
        Descriptors.Chi1n,
        Descriptors.Chi2n,
        Descriptors.Chi3n,
        Descriptors.Chi4n,
        rdMolDescriptors.CalcNumRings,
        rdMolDescriptors.CalcNumHeterocycles,
        rdMolDescriptors.CalcNumBridgeheadAtoms,
        rdMolDescriptors.CalcNumAtomStereoCenters,
        rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters,
        rdMolDescriptors.CalcNumSpiroAtoms,
        rdMolDescriptors.CalcNumBridgeheadAtoms,
        rdMolDescriptors.CalcNumAtomStereoCenters,
        rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters,
        rdMolDescriptors.CalcFractionCSP3,
        rdMolDescriptors.CalcLabuteASA,
        rdMolDescriptors.CalcPhi,
        rdMolDescriptors.CalcTPSA,
        Chem.Crippen.MolLogP,
    ]

    desc = []
    for func in funcs:
        try:
            desc.append(func(mol))
        except:
            desc.append(0)

    return np.array(desc)