import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.metrics.pairwise import rbf_kernel
import warnings
from torch_geometric.nn import GAT
from torch_geometric.nn import GCNConv
from torch_geometric.nn import DimeNet
import matplotlib.pyplot as plt
from  torch.distributions import multivariate_normal
import egnn_clean as eg
from torch_scatter import scatter_add
from torch.nn.functional import normalize
from torch_geometric.utils import add_self_loops
from utils import *


    
class GraphFlow_GCN(nn.Module): ##Simple NN-masking approach
    
    def __init__(self,c_in,c_out,params=None):
        super().__init__()
        layers = []
        activation_fns = []
        layers.append(GCNConv(c_in+1, 64))
        layers.append(GCNConv(64, 32))
        layers.append(GCNConv(32, c_out))
        activation_fns.append(nn.Tanh())
        activation_fns.append(nn.Tanh())
        #activation_fns.append(nn.Sigmoid())
        #activation_fns.append(nn.Softmax(dim=1))
        self.layer = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns)

        
    def forward(self,t,data,edges,pos,edge_attr):
        #print("########## I am here working ###########") 
        #print(self.params)
        Node_feats, Edge_index,Pos = data, edges,pos
        dx = data.float()
        tt = torch.ones_like(dx[:, :1]) * t #Concatenating the time
        dx_final = torch.cat([tt, dx], 1)
        for l, Layer in enumerate(self.layer):
            dx_final = Layer(dx_final,edge_index = edges.long(),edge_weight=edge_attr.float())
            # if not last layer, use nonlinearity
            if l !=2: dx_final = self.activation_fns[l](dx_final)

        return dx_final
    
class GraphFlow_EGNN(nn.Module): ##Simple NN-masking approach
    
    def __init__(self,c_in,c_out,params=None):
        super().__init__()
        layers = []
        activation_fns = []
        layers.append(eg.EGNN(in_node_nf=c_in+1, hidden_nf=32, out_node_nf=c_out, in_edge_nf=1))
        #activation_fns.append(nn.Softmax(dim=1))
        #activation_fns.append(nn.Tanh())
        #activation_fns.append(nn.Tanh())
        #activation_fns.append(nn.Softmax(dim=1))
        self.layer = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns)

        
    def forward(self,t,data,edges,pos,edge_attr):
        Node_feats, Edge_index = data, edges
        edge_attr = edge_attr.float()
        edges_final = [edges[0].long(), edges[1].long()]
        pos = torch.cat((pos,torch.zeros((len(pos),1))),1).float()
        
        dx = data.float()
        tt = torch.ones_like(dx[:, :1]) * t #Concatenating the time
        dx_final = torch.cat([tt.float(), dx], 1)
        for l, Layer in enumerate(self.layer):
            dx_final,pos = Layer(dx_final,pos,edges_final,edge_attr)

        return dx_final 
    
class GraphFlow_EGNN_3D(nn.Module): ##Simple NN-masking approach
    
    def __init__(self,c_in,c_out,params=None):
        super().__init__()
        layers = []
        activation_fns = []
        layers.append(eg.EGNN(in_node_nf=c_in+1, hidden_nf=32, out_node_nf=c_out, in_edge_nf=1))
        #activation_fns.append(nn.Softmax(dim=1))
        self.layer = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns)

        
    def forward(self,t,data,edges,pos,edge_attr):
        Node_feats, Edge_index = data, edges
        edge_attr = edge_attr.float()
        edges_final = [edges[0].long(), edges[1].long()]
       
        dx = data.float()
        tt = torch.ones_like(dx[:, :1]) * t #Concatenating the time
        dx_final = torch.cat([tt, dx], 1)
        for l, Layer in enumerate(self.layer):
            dx_final,pos = Layer(dx_final,pos.float(),edges_final,edge_attr)

        return dx_final 
