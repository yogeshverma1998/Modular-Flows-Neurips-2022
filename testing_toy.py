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
from matplotlib.patches import ConnectionPatch
from  molecule_metrics import *


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
parser.add_argument("--data", type=str, default="4x4_chess", choices=["4x4_chess","16x16_chess","stripes"])



parser.add_argument('--niters', type=int, default=1000)
parser.add_argument('--n_trials',type=int,default=10)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--test_batch_size', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--esamples', type=int, default=1000)
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
warnings.filterwarnings("ignore")    
eval_data = []
mol = []
for j in range(args.esamples):
    if args.data = "stripes":
        data = alternate_strip_data_grid_eval()
    elif args.data = "4x4_chess" :
        data = alternate_checker_data_grid_mod_eval()
    else:
        data =  alternate_checker_data_grid_eval()
        
    eval_data.append(data)
    
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
eval_loader = DataLoader(eval_data, batch_size=1, shuffle=False,num_workers=0)
cwd = os.getcwd()

model = torch.load(str(cwd) + "/Models/" +str(args.model_name))
print(model)
model.eval()
final_result = 0
for trial in range(args.n_trials):
    print("########################### Trial ",trial," ###############################")
    for idx,data in enumerate(eval_loader):
        z = reverse_transform(model,data,ntimes=80, memory=0.01, device='cpu')
        final_pred = z[79]
        final_result = final_result + final_pred
        

final_result = final_result/len(eval_loader)
if args.data = "stripes":
        get_mapping_strip(final_result)
elif args.data = "4x4_chess" :
        get_mapping_small_chess(final_result)
else:
        get_mapping_big_chess(final_result)
    
    

        
        
        
        
        
'''        
        
        
        if idx ==0:
            final_result = z[79]
        else:
            final_result = np.concatenate((final_result,z[79]),axis=0)
        
        
    plt.scatter(final_result[:,0],final_result[:,1])
    plt.axis("off")
    plt.savefig("/notebooks/ffjord/Model_saved_toy/final_checker_small"+str(trial)+".png")
    plt.clf()
    density = plt.hist2d(final_result[:,0],final_result[:,1],bins=(4,4))
    plt.axis("off")
    plt.savefig("/notebooks/ffjord/Model_saved_toy/final_checker_small"+str(trial)+".png")
    plt.clf()
    print(density[0])
    final_density  = density[0]>55
    fig, ax = plt.subplots(figsize=(15,15))
    x = np.linspace(0,3,4)
    y = np.linspace(0,3,4)
    x1,y1 = np.meshgrid(x,y)
    ax.scatter(x1,y1,c=final_density,s=800)
    plt.axis("off")
    
    plt.xlim([-1,5])
    plt.ylim([-1,5])
    x_points =[] 
    y_points = []
    for x_pos in x:
        for y_pos in y:
            x_points.append(x_pos)
            y_points.append(y_pos)
            
    ax.plot(x_points, y_points, linewidth=0, marker="o", color="black",
         markersize=np.sqrt(800), markerfacecolor='none', markeredgewidth=3)
    
    for i in range(len(x_points)-1):
        new_x_pos = x_points[i] + 1
        new_y_pos = y_points[i]
        new_x_pos2 = x_points[i]
        new_y_pos2 = y_points[i]+1
        cp = ConnectionPatch((x_points[i],y_points[i]), (new_x_pos, new_y_pos), 
                         coordsA='data', coordsB='data', axesA=ax, axesB=ax,
                         shrinkA=np.sqrt(800)/2, shrinkB=np.sqrt(800)/2,
                         linewidth=2)
        cp2 = ConnectionPatch((x_points[i],y_points[i]), (new_x_pos2, new_y_pos2), 
                         coordsA='data', coordsB='data', axesA=ax, axesB=ax,
                         shrinkA=np.sqrt(800)/2, shrinkB=np.sqrt(800)/2,
                         linewidth=2)
        ax.add_patch(cp)
        ax.add_patch(cp2)
    plt.savefig("/notebooks/ffjord/Model_saved_toy/grid_checker_small"+str(trial)+".png")
        
        
        
'''        
        
        #final_result = final_result + z[67]
        
    #plt.imshow(final_result.reshape(16,16))
    #plt.savefig("/notebooks/ffjord/Model_saved_toy/new_checker_density_"+str(trial)+".png")
    #plt.clf()
    #plt.imshow(data.y.reshape(16,16))
    #plt.savefig("/notebooks/ffjord/Model_saved_toy/org_checker_density"+str(idx)+".png")
    #plt.clf()
    
    
    
    
    '''
    
    for time_step,time_value in enumerate(z):
        if time_step%10==0:
            if time_step!=0: samples = 100*time_step
            else: samples = 500
            check_data = alternate_checker_data_eval(samples)
            time_value = np.concatenate((time_value,check_data.y),axis=0)
            density = plt.hist2d(time_value[:,0],time_value[:,1],bins=(32,32),range=[[-4,4],[-4,4]],density=True)
            x, y = np.mgrid[-4:4:0.25, -4:4:0.25]
            plt.axis("off")
            plt.savefig("/notebooks/ffjord/Model_saved_toy/generated_time_density_checker"+str(time_step)+".png")
            plt.clf()
            grad = np.gradient(density[0]) # calculating array of gaussian gradient
            n=-2
            U = grad[0] # x component of gaussian gradient
            V = grad[1] # y component of gaussian gradient
            color = np.sqrt(((U-n)/2)*2 + ((V-n)/2)*2)

            X1, Y1 = np.meshgrid(x, y)
 
            fig, ax = plt.subplots()
            ax.quiver(x, y, U, V, color, alpha = 1,cmap=plt.cm.jet)
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])    
            ax.axis([-4,4, -4, 4])
            plt.savefig("/notebooks/ffjord/Model_saved_toy/vfield_checker"+str(time_step)+".png")
      
    
    plt.scatter(z[49][:,0],z[49][:,1])
    plt.savefig("/notebooks/ffjord/Model_saved_toy/Generated_sample_checker"+str(trial)+".png")
    plt.clf()
    final_data = z[49]
    for i in range(4):
        eval_data = alternate_checker_data_eval(100000)
        final_data = np.concatenate((final_data,eval_data.y),axis=0)
    plt.scatter(final_data[:,0],final_data[:,1])
    plt.savefig("/notebooks/ffjord/Model_saved_toy/final_generated_sample_checker"+str(trial)+".png")
    plt.clf()
    plt.scatter(eval_data.y[:,0],eval_data.y[:,1])
    plt.savefig("/notebooks/ffjord/Model_saved_toy/Original_sample_checker"+str(trial)+".png")
    plt.clf()
    density = plt.hist2d(final_data[:,0],final_data[:,1],bins=(32,32),range=[[-4,4],[-4,4]])
    plt.axis("off")
    plt.savefig("/notebooks/ffjord/Model_saved_toy/final_generated_sample_density_checker"+str(trial)+".png")
    '''               