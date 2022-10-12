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
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
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
from scipy.interpolate import BSpline, make_interp_spline
import lib.toy_data as toy_data
import lib.utils as utils
from lib.visualize_flow import visualize_transform
import lib.layers.odefunc as odefunc
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
from scipy.special import softmax
from  molecule_metrics import *


SOLVERS = ["dopri8","dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams']
parser = argparse.ArgumentParser('ModFlow')
parser.add_argument(
    "--layer_type", type=str, default="concatsquash",
    choices=["ignore", "concat", "concat_v2", "squash", "concatsquash", "concatcoord", "hyper", "blend"]
)
saved_models = ['/data/pre_trained_model/ZINC_2D_pretrained.pt','']
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
parser.add_argument('--pretrain', type=bool, default=False)
parser.add_argument('--niters', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--test_batch_size', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--esamples', type=int, default=1000)
parser.add_argument('--seed', type=int, default=99)

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
args = parser.parse_args()
warnings.filterwarnings("ignore")    
eval_data = []
mol = []
cwd = os.getcwd()
data_path =  str(cwd) + '/data/' + str(args.data) + ".txt"
with open(data_path) as f:
    Smiles = f.readlines()
    
Unique_elements  = get_unique(Smiles[0:5000])

smiles_passed = []
for i in Smiles[0:args.esamples]:
    smiles_passed.append(i)
    mol.append(Chem.MolFromSmiles(i))
    data,order = get_graph_data_with_polar_2D_eval(i,Unique_elements)
    #data,order = get_graph_data_with_polar_CNN_eval(i,Unique_elements)
    eval_data.append((data,order))
    

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
eval_loader = DataLoader(eval_data, batch_size=1, shuffle=False,num_workers=0)
model = torch.load(str(cwd) + "/Models/" +str(args.model_name))
print(model)
model.eval()
Gen = []
Gen_mol = []
novel_ratio = []
validity_ratio = []
diversity_score = []
for trial in range(args.n_trials):
    valid = 0
    num = 0
    print("########################### Trial ",trial," ###############################")
    for idx,k in enumerate(eval_loader):  
                start_point = []
                end_point = []
                data = k[0].to(device)
                z = reverse_transform(model,data,ntimes=101, memory=0.01, device='cpu')
                gen_smiles,score = check_validity_QM9(data,z[100],Unique_elements,num,0,k[1])
                if score == 1:
                    Gen.append(gen_smiles)
                    valid = valid + score    
                    Gen_mol.append(Chem.MolFromSmiles(gen_smiles))       
                num = num +1
                
    novel_ratio.append(check_novelty(Gen, smiles_passed))
    diversity = MolecularMetrics.diversity_scores(Gen_mol, mol)
    for i in diversity:
        diversity_score.append(i)

print("Validity ",valid/num, " Novelty ",np.mean(novel_ratio)," Diversity ", sum(diversity_score)/len(diversity_score))




































'''              
                for channel in range(4):
                    one_density.append(list(z[100][:,channel].flatten()))
                 
                pred  = softmax(z[100],axis=1)
                for channel in range(4):
                    one_density_soft.append(list(pred[:,channel].flatten()))
                #for idx_point,points in enumerate(zip(start_point,end_point)):
                #        plt.scatter(0,points[0],color=color[idx_point],label=Unique_elements[idx_point])
                #        plt.scatter(time_stamp[-1],points[1],color=color[idx_point])
                #plt.legend(frameon=False,loc=0)
                #plt.xticks(np.arange(0,time_stamp[-1],500))
                #plt.axis("off")
                #plt.savefig("/notebooks/ffjord/trajectory_visual/Node_wise_density"+str(idx)+".png")
                #plt.clf()
                #plt.clf()
                 
    X_plot = np.linspace(-10, 6, 1000)[:, np.newaxis]
    bins = np.linspace(-10, 6, 100)
    fig, ax = plt.subplots(figsize=(20,6))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    color_choice = ['black','orange','red','blue']
    #ßone_density[3] = one_density[3]/2 - one_density[0] + one_density[2]  
    for channel in range(4):
        fact = -1**channel
        new_density = [i+2*fact*channel for i in one_density[channel]]
        kde = KernelDensity(kernel="cosine", bandwidth=0.9).fit(np.array(new_density).reshape(-1,1))
        log_dens = kde.score_samples(X_plot)
        ax.fill(X_plot[:, 0], np.exp(log_dens), fc="#AAAAFF",fill=False,linewidth=5,color=color_choice[channel],linestyle='-')
        plt.axis("off")
    plt.savefig("/notebooks/ffjord/trajectory_visual/kde_density_without_soft.png")
        #plt.show()
    #color_choice2 = ['orange','red','blue','black']
    X_plot = np.linspace(-2, 1, 300)[:, np.newaxis]
    bins = np.linspace(-2, 1, 50)
    fig, ax = plt.subplots(figsize=(20,6))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    #ßone_density[3] = one_density[3]/2 - one_density[0] + one_density[2]  
    for channel in range(4):
        new_soft = [i - 0.2*channel for i in one_density_soft[channel]]
        kde = KernelDensity(kernel="cosine", bandwidth=0.15).fit(np.array(new_soft).reshape(-1,1))
        log_dens = kde.score_samples(X_plot)
        ax.fill(X_plot[:, 0], np.exp(log_dens), fc="#AAAAFF",fill=False,linewidth=5,color=color_choice[channel],linestyle='-')
        ax.invert_xaxis()
        plt.axis("off")
    plt.savefig("/notebooks/ffjord/trajectory_visual/kde_density_with_soft.png")    
    
    
    
        
'''    
