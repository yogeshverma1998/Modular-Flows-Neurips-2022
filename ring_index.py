import warnings
from graphflow_model_function import *
from vocab import *
from utils import *
import lib.layers.wrappers.cnf_regularization as reg_lib
import lib.spectral_norm as spectral_norm
import lib.layers as layers
from lib.layers.odefunc import divergence_bf, divergence_approx
from torch_geometric.loader import DataLoader
import torch.nn.functional as Fin
import pandas as pd
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF
from pysmiles import read_smiles
import matplotlib 
from torch_geometric.data import Data
import matplotlib
matplotlib.use('Agg')
import argparse
import os
import time
import torch
import torch.optim as optim

import lib.toy_data as toy_data
import lib.utils as utils
from lib.visualize_flow import visualize_transform
import lib.layers.odefunc as odefunc


parser = argparse.ArgumentParser('Tree Representation')
parser.add_argument('--nsamples', type=int, default=10000)
parser.add_argument('--data',type=str, default="QM9")
parser.add_argument('--nrings', type=int, default=30)
args = parser.parse_args()

cwd = os.getcwd()
data_path =  str(cwd) + '/data/' + str(args.data) + ".txt"
with open(data_path) as f:
    Smiles = f.readlines()

freq = get_unique_rings(Smiles[0:args.nsamples],args.data)
print(freq)
top_ring_index = list(freq.argsort()[-args.nrings:][::-1])
print(top_ring_index)
f = open(str(cwd) + "/data/ring_index_"+str(args.data)+".txt", "w")
for i in top_ring_index:
    f.write(str(i))
    f.write("\n")
f.close()
