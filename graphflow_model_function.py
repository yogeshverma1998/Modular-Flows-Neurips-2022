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

class GraphFlow_NN(nn.Module): ##Simple NN-masking approach
    
    def __init__(self,c_in,c_out,params=None):
        super().__init__()
        layers = []
        activation_fns = []
        layers.append(nn.Linear(5*(c_in)+1,15))
        layers.append(nn.Linear(15,c_out))
        activation_fns.append(nn.Tanh())
        #activation_fns.append(nn.Sigmoid())
        activation_fns.append(nn.Softmax(dim=1))
        #activation_fns.append(nn.Tanh())
        self.layer = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns)

    def forward(self,t,data,edges):
        #print("########## I am here working ###########") 
        #print(self.params)
        Node_feats, Edge_index = data, edges
        Adj = np.zeros((len(Node_feats)+1,len(Node_feats)+1))
        
        for edge_start,edge_end in zip(Edge_index[0],Edge_index[1]): #create Adjacency matrix
            Adj[int(edge_start)][int(edge_end)] = 1
        #print("Good till now") 
        edges = torch.nonzero(torch.from_numpy(Adj),as_tuple=False) # Returns indices where the adjacency matrix is not 0 => edges
        num_nodes = len(Node_feats)
        num_elements = Node_feats.shape[1]
        data = pd.DataFrame(edges.numpy(),columns=["node_number","neighbour"])
        Final_input = torch.zeros(1,5*num_elements)
        for j in range(num_nodes): #number_of_nodes
            a_input = torch.index_select(input=Node_feats, index=torch.tensor([j]), dim=0)
            neighs = torch.tensor(list(data[data['node_number'] == j]['neighbour'].to_numpy()))
            if len(neighs) == 0:
                    neigh_add_feats = torch.Tensor([[0 for num in range(4*num_elements)]])
                    a_input = torch.cat([a_input,neigh_add_feats],dim=-1)
    
                    
            if len(neighs) < 4 and len(neighs) > 0 :
                    #print(j,neighs,Node_feats.shape)
                    neigh_feats = torch.index_select(input=Node_feats, index=neighs, dim=0).view(1,len(neighs)*num_elements)
                    neigh_add_feats = torch.Tensor([[0 for num in range((4-len(neighs))*num_elements)]])
                    a_input = torch.cat([a_input,neigh_feats,neigh_add_feats],dim=-1)
                    
            if len(neighs) == 4:
                    neigh_feats = torch.index_select(input=Node_feats, index=neighs, dim=0).view(1,len(neighs)*num_elements)
                    a_input = torch.cat([a_input,neigh_feats],dim=-1)
     
            Final_input = torch.cat([Final_input,a_input],dim=0)
        

        Final_input = Final_input[1:len(Final_input)].view(num_nodes,num_elements*5)
        dx = Final_input.float()
        tt = torch.ones_like(dx[:, :1]) * t #Concatenating the time
        dx_final = torch.cat([tt, dx], 1)
        for l, Layer in enumerate(self.layer):
            dx_final = Layer(dx_final)
            # if not last layer, use nonlinearity
            if l==0: dx_final = self.activation_fns[l](dx_final)

        return dx_final



    
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
    
    
class GraphFlow_MPNN(nn.Module): ##Simple NN-masking approach
    
    def __init__(self,c_in,c_out,params=None):
        super().__init__()
        layers = []
        activation_fns = []
        layers.append(GCNConv(c_in+1, 64))
        layers.append(GCNConv(64, 64))
        layers.append(GCNConv(64, 32))
        layers.append(GCNConv(32, c_out))
        activation_fns.append(nn.Tanh())
        activation_fns.append(nn.Tanh())
        activation_fns.append(nn.Tanh())
        activation_fns.append(nn.Softmax(dim=1))
        self.layer = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns)

        
    def forward(self,t,data,edges):
        #print("########## I am here working ###########") 
        #print(self.params)
        Node_feats, Edge_index = data, edges
        dx = data.float()
        tt = torch.ones_like(dx[:, :1]) * t #Concatenating the time
        dx_final = torch.cat([tt, dx], 1)
        for l, Layer in enumerate(self.layer):
            dx_final = Layer(dx_final,edge_index = edges.long())
            # if not last layer, use nonlinearity
            dx_final = self.activation_fns[l](dx_final)

        return dx_final  

    

    
class GraphFlow_Polar_CNN(nn.Module): ##Simple NN-masking approach
    
    def __init__(self,c_in,c_out,params=None):
        super().__init__()
        layers = []
        activation_fns = []
        layers.append(nn.Conv2d(c_in+1, 15, kernel_size = 5))
        layers.append(nn.Conv2d(15, 5, kernel_size = 5))
        layers.append(nn.Conv2d(5, 1, kernel_size = 5))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(304,c_out))
        activation_fns.append(nn.Tanh())
        activation_fns.append(nn.Tanh())
        activation_fns.append(nn.Tanh())
        #self.layer_1 = 
        #self.max_pool = nn.MaxPool2d(2,4)
        #self.flatten_layer = nn.Flatten()
        #self.linear_layer = nn.Linear(490,c_out)
        #self.act1 = nn.Tanh()
        #activation_fns.append(nn.Sigmoid())
        #activation_fns.append(nn.Sigmoid())
        #activation_fns.append(nn.Softmax(dim=1))
        #self._time_dependent = nn.Linear(1, c_out,bias=False)
        self.layer = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns)
        self.Unique_elements = c_out

        
    def forward(self,t,data,edges,pos):
        
        Node_feats, Edge_index, density_map = data, edges,pos
        final_edge_index = add_self_loops(Edge_index)
        #print(final_edge_index[0])
        src_index = final_edge_index[0][0].long()
        target_index = final_edge_index[0][1].long()
        Rel_features = Node_feats[src_index] # num_edges x node_embedding_length (delta z_{ij})
        #print("Node_features",Node_feats)
        #print(pos[0])
        src_density_plane = torch.index_select(pos.view(len(pos),pos.shape[1]*pos.shape[2]), 0, src_index) #num_edges x pdf_flatten_vector
        #print("Src_density_plane",src_density_plane[0])
        #src_density_plane = src_density_plane.expand(-1,-1,Node_feats.shape[1]) # num_edges x pdf_flatten_vector x node_embedding_length
        src_density_plane = src_density_plane.view(len(src_index),1,src_density_plane.shape[1])
        
        
        Rel_features = Rel_features.view(len(src_index),Rel_features.shape[1],1)
        #print("Rel_features",Rel_features)
        Final_edge_features = torch.matmul(Rel_features,src_density_plane)   #num_edges x  Rel_features.shape[1] x src_density_plane.shape[1]
        Final_node_features = scatter_add(Final_edge_features, target_index, 0).view(len(Node_feats),Node_feats.shape[1],50,20)
        #Final_node_features = torch.nn.functional.normalize(Final_node_features,p=2,dim=2)
        #print("Final_node_feature_map",Final_node_features[0][3])
        
        dx = Final_node_features.float()
        tt = torch.ones_like(dx[:, :1, :, :]) * t
        dx = torch.cat([tt, dx], 1)
        for l, layer in enumerate(self.layer):
            dx = layer(dx)
            if l ==0: dx = self.activation_fns[l](dx)


        return dx 

    
class GraphFlow_Polar_CNN_3D(nn.Module): ##Simple NN-masking approach
    
    def __init__(self,c_in,c_out,params=None):
        super().__init__()
        layers = []
        activation_fns = []
        layers.append(nn.Conv3d(c_in+1, 10, kernel_size = 5))
        layers.append(nn.MaxPool3d(4,2,2))
        layers.append(nn.Conv3d(10, 5, kernel_size = 5))
        layers.append(nn.MaxPool3d(4,2,2))
        layers.append(nn.Conv3d(5, 1, kernel_size = 3))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(25,c_out))
        activation_fns.append(nn.Tanh())
        activation_fns.append(nn.Tanh())
        activation_fns.append(nn.Tanh())
        #self.layer_1 = 
        #self.max_pool = nn.MaxPool2d(2,4)
        #self.flatten_layer = nn.Flatten()
        #self.linear_layer = nn.Linear(490,c_out)
        #self.act1 = nn.Tanh()
        #activation_fns.append(nn.Sigmoid())
        #activation_fns.append(nn.Sigmoid())
        #activation_fns.append(nn.Softmax(dim=1))
        #self._time_dependent = nn.Linear(1, c_out,bias=False)
        self.layer = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns)
        self.Unique_elements = c_out

        
    def forward(self,t,data,edges,pos):
        
        Node_feats, Edge_index, density_map = data, edges,pos
        final_edge_index = add_self_loops(Edge_index)
        #print(final_edge_index[0])
        src_index = final_edge_index[0][0].long()
        target_index = final_edge_index[0][1].long()
        Rel_features = Node_feats[src_index] # num_edges x node_embedding_length (delta z_{ij})
        #print("Node_features",Node_feats)
        #print(pos[0])
        src_density_plane = torch.index_select(pos.view(len(pos),pos.shape[1]*pos.shape[2]*pos.shape[3]), 0, src_index) #num_edges x pdf_flatten_vector
        #print("Src_density_plane",src_density_plane[0])
        #src_density_plane = src_density_plane.expand(-1,-1,Node_feats.shape[1]) # num_edges x pdf_flatten_vector x node_embedding_length
        src_density_plane = src_density_plane.view(len(src_index),1,src_density_plane.shape[1])
        
        
        Rel_features = Rel_features.view(len(src_index),Rel_features.shape[1],1)
        #print("Rel_features",Rel_features)
        Final_edge_features = torch.matmul(Rel_features,src_density_plane)   #num_edges x  Rel_features.shape[1] x src_density_plane.shape[1]
        Final_node_features = scatter_add(Final_edge_features, target_index, 0).view(len(Node_feats),Node_feats.shape[1],pos.shape[1],pos.shape[2],pos.shape[3])
        #Final_node_features = torch.nn.functional.normalize(Final_node_features,p=2,dim=2)
        #print("Final_node_feature_map",Final_node_features[0][3])
        
        dx = Final_node_features.float()
        tt = torch.ones_like(dx[:, :1, :, :]) * t
        dx = torch.cat([tt, dx], 1)
        for l, layer in enumerate(self.layer):
            dx = layer(dx)
            #print(dx.shape)
            if l ==0: dx = self.activation_fns[l](dx)


        return dx
    
    
    
class GraphFlow_Polar_CNN_v2(nn.Module): ##Simple NN-masking approach
    
    def __init__(self,c_in,c_out,params=None):
        super().__init__()
        layers = []
        activation_fns = []
        layers.append(nn.Conv2d(c_in+1, 15, kernel_size = 5))
        layers.append(nn.Conv2d(15, 10, kernel_size = 5))
        layers.append(nn.Conv2d(10, 5, kernel_size = 5))
        layers.append(nn.Conv2d(5,1, kernel_size = 5))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(16,c_out))
        activation_fns.append(nn.Tanh())
        activation_fns.append(nn.Tanh())
        activation_fns.append(nn.Tanh())
        activation_fns.append(nn.Tanh())
        #self.layer_1 = 
        #self.max_pool = nn.MaxPool2d(2,4)
        #self.flatten_layer = nn.Flatten()
        #self.linear_layer = nn.Linear(490,c_out)
        #self.act1 = nn.Tanh()
        #activation_fns.append(nn.Sigmoid())
        #activation_fns.append(nn.Sigmoid())
        #activation_fns.append(nn.Softmax(dim=1))
        #self._time_dependent = nn.Linear(1, c_out,bias=False)
        self.layer = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns)
        self.Unique_elements = c_out

        
    def forward(self,t,data,edges,pos):
        Node_feats, Edge_index, polar_r,polar_angle = data, edges,pos[:,0],pos[:,1]
        cov = [[5.0, 0], [0, 1.0]]        
        Final_feature = torch.zeros((1,Node_feats.shape[1],20,20))
        edge_value_decomposed = []        
        dist = multivariate_normal.MultivariateNormal(torch.Tensor([0,1]), torch.Tensor(cov))
        Points = dist.sample((1000,))
        for idx_entry,node_val in enumerate(zip(Edge_index[0],Edge_index[1])):
            org_node = node_val[0].int()
            neigh_node = node_val[1].int()
            diff_factor = Node_feats[org_node] - Node_feats[neigh_node]
            x_temp = polar_r[neigh_node]*torch.ones(1000) + Points[:,0]
            y_temp = polar_angle[neigh_node]*torch.ones(1000) + Points[:,1]
            for idx_num,unique_elements in enumerate(diff_factor):
                density_val = plt.hist2d(x_temp.detach().numpy(), y_temp.detach().numpy(), bins=(20,20),density=True,range=[[0,10],[-3.14,3.14]])
                density_temp = unique_elements*torch.Tensor(density_val[0])
                if idx_num == 0:
                    adj_plane = density_temp
                else:
                    adj_plane = torch.cat((adj_plane,density_temp),0)
            if idx_entry ==0: 
                temp_plane = adj_plane.view(len(diff_factor),20,20)
            else:
                temp_plane = temp_plane + adj_plane.view(len(diff_factor),20,20)
                
            #for i in range(len(diff_factor)):
            #    figure = plt.gcf()
            #   figure.set_size_inches(8, 6)
            #    plt.imshow(temp_plane[i].detach().numpy(),cmap=plt.cm.jet)
            #    plt.savefig("Element"+str(i)+".png", dpi=100)
                
            
            if  idx_entry +1 == len(Edge_index[0]):
                Final_feature = torch.cat((Final_feature,temp_plane.view(1,len(diff_factor),20,20)),0)
            
            elif Edge_index[0][idx_entry+1] -  Edge_index[0][idx_entry] > 0:
                Final_feature_temp = temp_plane.view(1,len(diff_factor),20,20)
                Final_feature = torch.cat((Final_feature,Final_feature_temp),0)
                
               
        dx = Final_feature[0:len(Node_feats)].float()
        tt = torch.ones_like(dx[:, :1, :, :]) * t
        dx = torch.cat([tt, dx], 1)
        for l, layer in enumerate(self.layer):
            dx = layer(dx)
            if l ==0: dx = self.activation_fns[l](dx)


        return dx 
    
    
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