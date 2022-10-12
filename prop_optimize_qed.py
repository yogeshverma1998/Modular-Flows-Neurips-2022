import warnings
import os
from graphflow_model_function import *
from utils import *
import lib.layers.wrappers.cnf_regularization as reg_lib
import lib.spectral_norm as spectral_norm
import lib.layers as layers
from lib.layers.odefunc import divergence_bf, divergence_approx
from torch_geometric.loader import DataLoader
import torch.nn.functional as Fin
import timeit
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
import argparse
import os
import time
import torch
import torch.optim as optim
import random
import lib.toy_data as toy_data
import lib.utils as utils
from lib.visualize_flow import visualize_transform
import lib.layers.odefunc as odefunc
from rdkit.Chem.QED import qed
from sklearn.linear_model import LinearRegression
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from sascorer import *


SOLVERS = ["dopri8","dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams']
parser = argparse.ArgumentParser('ModFlow')
parser.add_argument(
    "--layer_type", type=str, default="concatsquash",
    choices=["ignore", "concat", "concat_v2", "squash", "concatsquash", "concatcoord", "hyper", "blend"]
)

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

parser.add_argument('--residual', type=eval, default=False, choices=[True, False])
parser.add_argument('--rademacher', type=eval, default=False, choices=[True, False])
parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])
parser.add_argument('--batch_norm', type=eval, default=False, choices=[True, False])
parser.add_argument('--bn_lag', type=float, default=0)

parser.add_argument('--niters', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--test_batch_size', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--nsamples', type=int, default=1000)
parser.add_argument('--seed', type=int, default=99)
parser.add_argument('--model_name', type=str, default="none")

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


cwd = os.getcwd()
data_path =  str(cwd) + '/data/' + str(args.data) + ".txt"
with open(data_path) as f:
    Smiles = f.readlines()
final_data = []    
samples_to_consider = args.nsamples
Unique_elements  = get_unique(Smiles[0:5000])
for i in Smiles[0:samples_to_consider]:
    data = get_graph_data_with_polar_2D(i,Unique_elements)
    final_data.append(data)
    
    


def fit_linear_model(model,data,smiles):
    z_data = []
    qed_scores = []
    for idx,i in enumerate(data):
        print(idx)
        zero = torch.zeros(i[0].x.shape[0], 1).to(i[0].x)
        z, delta_logp = model(i[0], zero)
        print(z.shape)
        for node in z:
            qed_scores.append(qed(Chem.MolFromSmiles(smiles[idx])))
        z_data.append(z.detach().numpy())
    
    z_data = np.vstack(z_data)
    true_scores = np.vstack(qed_scores)
    print(z_data.shape,true_scores.shape)
    
    linreg = LinearRegression(fit_intercept=True, normalize=True).fit(z_data, true_scores)
    print("R2 score:{:.2f}".format(linreg.score(z_data, true_scores)))
    return linreg

def get_latent_rep(model,smiles,Unique_elements):
    data = get_graph_data_with_polar_2D(smiles,Unique_elements)
    zero = torch.zeros(data[0].x.shape[0], 1).to(data[0].x)
    z, delta_logp = model(data[0], zero)
    return z,data[0].edge_index,data[1],data[0].pos

def generate_mol_along_axis(model,z0,edges,order,pos,axis,n_mols,delta):
    z_list= []
    Gen = []
    for dx in range(n_mols):
        z_new_mol = []
        for node in z0:
            z_new = node + axis*delta*dx
            z_new_mol.append(z_new)
               
        final_mol = torch.from_numpy(np.array(z_new_mol).reshape(z0.shape[0],-1))
        data = Data(x=final_mol, edge_index=edges,y=final_mol,pos=pos)
        reconst_mol = reverse_transform(model,data,ntimes=101, memory=0.01, device='cpu')
        gen_mol,score = check_validity_QM9(data,reconst_mol[100],Unique_elements,0,0,order)
        if score==1:
            gen_smiles = Chem.MolToSmiles(gen_mol,canonical=True,kekuleSmiles=True)
            Gen.append(gen_smiles)
            
    return Gen

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
modflow_model = torch.load(str(cwd) + "/Models/" +str(args.model_name))


print("Fitting property based model")
property_model = fit_linear_model(modflow_model,final_data,Smiles[0:samples_to_consider])
axis = property_model.coef_/np.linalg.norm(property_model.coef_)

smiles_to_be_considered = random.choices(Smiles,k=50000)
final_smile = []
for smi in smiles_to_be_considered:
    qed_smi = qed(Chem.MolFromSmiles(smi))
    if qed_smi > 0.5:
        final_smile.append(smi)
        
for mol_smiles in final_smile:
    print("For Molecule starting from ",mol_smiles," has QED ",qed(Chem.MolFromSmiles(mol_smiles)))
    z0,edges,order,pos = get_latent_rep(modflow_model,mol_smiles,Unique_elements)
    gen_smiles = generate_mol_along_axis(modflow_model, z0.detach().numpy(), edges,order,pos,axis=axis,n_mols=200,delta=0.02)
    if len(gen_smiles) ==0: continue
    for smi in gen_smiles:
        print(smi," has QED score as ",qed(Chem.MolFromSmiles(smi)))









        





