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

import lib.toy_data as toy_data
import lib.utils as utils
from lib.visualize_flow import visualize_transform
import lib.layers.odefunc as odefunc

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



def compute_loss(x,model):
    #if batch_size is None: batch_size = args.batch_size

    # load data
    zero = torch.zeros(x.x.shape[0], 1).to(x.x)
    
    # transform to z
    z, delta_logp = model(x, zero)
    
    # compute log q(z)
    logpz = standard_normal_logprob(z).sum(1, keepdim=True)

    logpx = logpz - delta_logp
    loss = -torch.mean(logpx)
    return loss


# logger
utils.makedirs(args.save)
logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))

if args.layer_type == "blend":
    logger.info("!! Setting time_length from None to 1.0 due to use of Blend layers.")
    args.time_length = 1.0

logger.info(args)

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')


final_data = []
cwd = os.getcwd()
data_path =  str(cwd) + '/data/' + str(args.data) + ".txt"
with open(data_path) as f:
    Smiles = f.readlines()


    
samples_to_consider = args.nsamples
Unique_elements  = get_unique(Smiles[0:samples_to_consider])
for i in Smiles[0:samples_to_consider]:
    rdkit_mol = Chem.MolFromSmiles(i)
    rdkit_mol = Chem.AddHs(rdkit_mol)
    y = AllChem.EmbedMolecule(rdkit_mol,useRandomCoords=True)
    rdkit_mol = Chem.AddHs(rdkit_mol)
    if y == -1: continue
    AllChem.UFFOptimizeMolecule(rdkit_mol)
    conf = rdkit_mol.GetConformer()
    data = get_graph_data_with_polar_3D(i,Unique_elements,conf,rdkit_mol)
    final_data.append(data)

    
print("########################################################################### Molecular Data is Loaded #######################################################################################")

regularization_fns, regularization_coeffs = create_regularization_fns(args)
hidden_dims = tuple(map(int, args.dims.split("-")))
diffeq = GraphFlow_EGNN_3D(len(Unique_elements),len(Unique_elements))
odefunc = layers.ODEfunc(
            diffeq=diffeq,
            divergence_fn=args.divergence_fn,
            residual=args.residual,
            rademacher=args.rademacher,
        )
cnf = layers.CNF(
            odefunc=odefunc,
            T=args.time_length,
            train_T=args.train_T,
            regularization_fns=regularization_fns,
            solver=args.solver,
        )

chain = [cnf for _ in range(args.num_blocks)]
if args.batch_norm:
        bn_layers = [layers.MovingBatchNorm1d(dims, bn_lag=args.bn_lag) for _ in range(args.num_blocks)]
        bn_chain = [layers.MovingBatchNorm1d(dims, bn_lag=args.bn_lag)]
        for a, b in zip(chain, bn_layers):
            bn_chain.append(a)
            bn_chain.append(b)
        chain = bn_chain
model = layers.SequentialFlow(chain).to(device)
set_cnf_options(args, model)

if args.spectral_norm: add_spectral_norm(model)
set_cnf_options(args, model)

logger.info(model)
logger.info("Number of trainable parameters: {}".format(count_parameters(model)))

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

time_meter = utils.RunningAverageMeter(0.93)
loss_meter = utils.RunningAverageMeter(0.93)
nfef_meter = utils.RunningAverageMeter(0.93)
nfeb_meter = utils.RunningAverageMeter(0.93)
tt_meter = utils.RunningAverageMeter(0.93)
end = time.time()
best_loss = float('inf')
model.train()


train_size = int(0.75*samples_to_consider)
Train_dataset = final_data
Test_dataset = final_data[train_size:]

train_loader = DataLoader(Train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=0)
test_loader = DataLoader(Test_dataset, batch_size=args.batch_size, shuffle=True,num_workers=0)

warnings.filterwarnings("ignore")
for itr in range(1, args.niters):
    total_loss = 0 
    total_test_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        data = batch[0].to(device)
        loss = compute_loss(data,model)
        loss_meter.update(loss.item())
        print("Loss for ",data,"sample is ",loss.item())

        total_loss += loss.item()
        total_time = count_total_time(model)
        nfe_forward = count_nfe(model)
        loss.backward()
        optimizer.step()

    nfe_total = count_nfe(model)
    nfe_backward = nfe_total - nfe_forward
    nfef_meter.update(nfe_forward)
    nfeb_meter.update(nfe_backward)
    time_meter.update(time.time() - end)
    tt_meter.update(total_time)
    
    log_message = (
            'Iter {:04d} | Time {:.4f}({:.4f}) | Loss {:.6f}({:.6f}) | NFE Forward {:.0f}({:.1f})'
            ' | NFE Backward {:.0f}({:.1f}) | CNF Time {:.4f}({:.4f})'.format(
                itr, time_meter.val, time_meter.avg, total_loss, loss_meter.avg, nfef_meter.val, nfef_meter.avg,
                nfeb_meter.val, nfeb_meter.avg, tt_meter.val, tt_meter.avg
            )
        )
    logger.info(log_message)
    if itr%1 == 0 or itr == args.niters:
        with torch.no_grad():
            model.eval()
            for test_batch in test_loader:
                data = test_batch[0].to(device)
                test_loss = compute_loss(data,model)
                total_test_loss += test_loss.item()
                test_nfe = count_nfe(model)
                print("Test  Loss for ",data,"sample is ",test_loss.item())
                
                
 
            log_message = '[TEST] Iter {:04d} | Test Loss {:.6f} | NFE {:.0f}'.format(itr, total_test_loss, test_nfe)
            logger.info(log_message)

            if total_test_loss < best_loss:
                best_loss = total_test_loss
                torch.save(model,str(cwd) + "/Models/" + "modflow_3d_"+str(args.data)+"_model_" + str(itr) + ".pt")
                
            model.train()


logger.info('Training and validation is finished and models are saved in the respective directories.')

