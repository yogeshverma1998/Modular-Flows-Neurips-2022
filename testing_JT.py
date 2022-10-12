import warnings
from graphflow_model_function import *
from utils import *
import lib.layers.wrappers.cnf_regularization as reg_lib
import lib.spectral_norm as spectral_norm
import lib.layers as layers
from lib.layers.odefunc import divergence_bf, divergence_approx
from torch_geometric.loader import DataLoader
import torch.nn.functional as Fin

import pandas as pd
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
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
matplotlib.rcParams.update({'font.size': 22})
import argparse
import os
import time
import torch
import torch.optim as optim

import lib.toy_data as toy_data
import lib.utils as utils
from lib.visualize_flow import visualize_transform
import lib.layers.odefunc as odefunc
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))

from  molecule_metrics import *


SOLVERS = ["dopri8","dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams']
parser = argparse.ArgumentParser('ModFlow')
parser.add_argument(
    "--layer_type", type=str, default="concatsquash",
    choices=["ignore", "concat", "concat_v2", "squash", "concatsquash", "concatcoord", "hyper", "blend"]
)

saved_models = ['/data/pre_trained_model/ZINC_JT_3D_pretrained.pt','']
parser.add_argument('--dims', type=str, default='64-64-64')
parser.add_argument("--num_blocks", type=int, default=1, help='Number of stacked CNFs.')
parser.add_argument('--time_length', type=float, default=0.5)
parser.add_argument('--train_T', type=eval, default=True)
parser.add_argument("--divergence_fn", type=str, default="brute_force", choices=["brute_force", "approximate"])
parser.add_argument("--nonlinearity", type=str, default="tanh", choices=odefunc.NONLINEARITIES)
parser.add_argument("--data", type=str, default="QM9", choices=["ZINC","QM9"])

parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
parser.add_argument('--atol', type=float, default=1e-5)
parser.add_argument('--rtol', type=float, default=1e-5)
parser.add_argument("--step_size", type=float, default=None, help="Optional fixed step size.")

parser.add_argument('--test_solver', type=str, default=None, choices=SOLVERS + [None])
parser.add_argument('--test_atol', type=float, default=None)
parser.add_argument('--test_rtol', type=float, default=None)
parser.add_argument('--n_trials', type=int, default=1)

parser.add_argument('--residual', type=eval, default=False, choices=[True, False])
parser.add_argument('--rademacher', type=eval, default=False, choices=[True, False])
parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])
parser.add_argument('--batch_norm', type=eval, default=False, choices=[True, False])
parser.add_argument('--bn_lag', type=float, default=0)
parser.add_argument('--model_name', type=str, default="none")


parser.add_argument('--esamples', type=int, default=1000)
parser.add_argument('--seed', type=int, default=99)
parser.add_argument('--nrings', type=int, default=30)

# Track quantities
parser.add_argument('--l1int', type=float, default=None, help="int_t ||f||_1")
parser.add_argument('--l2int', type=float, default=None, help="int_t ||f||_2")
parser.add_argument('--dl2int', type=float, default=None, help="int_t ||f^T df/dt||_2")
parser.add_argument('--JFrobint', type=float, default=None, help="int_t ||df/dx||_F")
parser.add_argument('--JdiagFrobint', type=float, default=None, help="int_t ||df_i/dx_i||_F")
parser.add_argument('--JoffdiagFrobint', type=float, default=None, help="int_t ||df/dx - df_i/dx_i||_F")

parser.add_argument('--save', type=str, default='experiments/cnf')
parser.add_argument('--viz_freq', type=int, default=10)
parser.add_argument('--val_freq', type=int, default=10)
parser.add_argument('--log_freq', type=int, default=10)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()
warnings.filterwarnings("ignore")    
eval_data = []
mol = []
cwd = os.getcwd()
data_path =  str(cwd) + '/data/' + str(args.data) + ".txt"
with open(data_path) as f:
    Smiles = f.readlines()
Unique_elements  = get_unique(Smiles[0:5000])


samples_to_evaluate = args.esamples
n_rings = args.nrings
top_ring_index = []
with open(str(cwd) + "/data/ring_index_"+str(args.data)+".txt") as f:
    lines = f.readlines()
for num,i in enumerate(lines):
    if num < n_rings:
        top_ring_index.append(int(i[0:-1]))

#Unique_elements  = get_unique(Smiles[0:samples_to_consider])
print(top_ring_index)

with open(str(cwd) + "/data/Rings_vocab_"+str(args.data)+".txt",'r') as file:
        lines = file.readlines()
        
rings_vocab = []
for entry in top_ring_index:
        rings_vocab.append(lines[entry])

        
top_ring_attributes = ["R"+str(int) for int in range(len(rings_vocab))]



for i in Smiles[9450:9450+args.esamples]:
    mol.append(Chem.MolFromSmiles(i))
    #data,order = get_graph_data_with_polar_2D_eval(i,Unique_elements)
    edges,node,node_pos = get_decomposed_mol(i,top_ring_attributes,rings_vocab)
    data = tensorize_molecule_eval(edges,node,Unique_elements,top_ring_attributes,node_pos)
    eval_data.append(data)
    

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
eval_loader = DataLoader(eval_data, batch_size=1, shuffle=False,num_workers=0)
model = torch.load(str(cwd) + "/Models/" +str(args.model_name))
print(model)
#model.eval()
Gen = []
Gen_mol = []
novel_ratio = []
validity_ratio = []
diversity_score = []
SAS_score = []

for trial in range(args.n_trials):
    valid = 0
    num = 0
    print("########################### Trial ",trial," ###############################")
    for idx,k in enumerate(eval_loader):   
                data = k[0].to(device)
                print(idx)
                if idx in  [184,222,253,473]: continue
                #print("Molecule " ,idx," Generated")
                z = reverse_transform(model,data,ntimes=101, memory=0.01, device='cpu')
                #print("Final",z[100])
                #print("Org",data.y)
                if idx ==0: 
                    Z_total = z[100]
                    Z_start = z[0]
                else: 
                    Z_total = np.concatenate((Z_total,z[100]),axis=0)
                    Z_start = np.concatenate((Z_start,z[0]),axis=0)
                final_edges,correc_syn,decomposed_node_attributes = reconstruct_valid_JT(data,z[100],Unique_elements,top_ring_attributes,0,0,rings_vocab)
                if correc_syn: continue
                #print("Final edges after correction",final_edges,decomposed_node_attributes)
                gen_smiles,score = check_JT(final_edges,decomposed_node_attributes)
                if score == 1: 
                    Gen.append(gen_smiles)
                    valid = valid + score
                    Gen_mol.append(Chem.MolFromSmiles(gen_smiles))

                num = num +1
            #f.close()
    novel_ratio.append(check_novelty(Gen, Smiles))
    validity_ratio.append(valid/num)
    #SAS = MolecularMetrics.synthetic_accessibility_score_scores(Gen_mol)
    diversity = MolecularMetrics.diversity_scores(Gen_mol, mol)
    for i in diversity:
        diversity_score.append(i)
print("Validity ",np.mean(validity_ratio),"Novelty ",np.mean(novel_ratio), "Diversity", sum(diversity_score)/len(diversity_score))