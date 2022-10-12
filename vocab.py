import datetime
import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import IPythonConsole
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF
from pysmiles import read_smiles
import matplotlib 
from torch_geometric.data import Data
from pysmiles import write_smiles, fill_valence
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
matplotlib.rcParams.update({'font.size': 22})
from rdkit import Chem,DataStructs
cwd = os.getcwd()


def GetRingSystems(mol, includeSpiro=False):
    ri = mol.GetRingInfo()
    systems = []
    for ring in ri.AtomRings():
        ringAts = set(ring)
        nSystems = []
        for system in systems:
            nInCommon = len(ringAts.intersection(system))
            if nInCommon and (includeSpiro or nInCommon>1):
                ringAts = ringAts.union(system)
            else:
                nSystems.append(system)
        nSystems.append(ringAts)
        systems = nSystems
    return systems


def get_unique_rings(Smiles,data_str):
    num =0
    freq = np.zeros(10000)
    for idx,i in enumerate(Smiles):
        print("On Molecule",idx)
        mol = Chem.MolFromSmiles(i)
        ring_system = GetRingSystems(mol)
        if num == 0:
            f = open(str(cwd) + "/data/" + "Rings_vocab_"+str(data_str)+".txt",'w')
            for ring in ring_system:
                submol = Chem.MolFragmentToSmiles(mol,atomsToUse=list(ring),kekuleSmiles=True)
                f.write(str(submol))
                f.write("\n")
            f.close()

        else:
            with open(str(cwd) + "/data/" + "Rings_vocab_"+str(data_str)+".txt",'r') as file:
                 lines = file.readlines()
            for ring in ring_system:
                submol = Chem.MolFragmentToSmiles(mol,atomsToUse=list(ring),kekuleSmiles=True)
                same = False
                for idx,old in enumerate(lines):
                    mol1 = Chem.MolFromSmiles(submol)
                    mol2 = Chem.MolFromSmiles(old)
                    fp1 = Chem.RDKFingerprint(mol1)
                    fp2 = Chem.RDKFingerprint(mol2)
                    if DataStructs.TanimotoSimilarity(fp1,fp2) == 1:
                        same = True
                        freq[idx] = freq[idx] + 1
                if same is False:
                    with open(str(cwd) + "/data/" + "Rings_vocab_"+str(data_str)+".txt", "a+") as file_object:
                        file_object.write(str(submol))
                        file_object.write("\n")
        num = num + 1
        
    return freq



