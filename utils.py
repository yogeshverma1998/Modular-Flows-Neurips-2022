import torch
import torch.nn as nn
import pandas as pd
import math
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import pandas as pd
import numpy as np
from rdkit import Chem
import os
import astropy
from astropy.coordinates import cartesian_to_spherical
from rdkit.Chem import AllChem, Draw
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF
from pysmiles import read_smiles
import matplotlib 
from torch_geometric.data import Data
import matplotlib
import networkx as nx
from pysmiles import write_smiles, fill_valence
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import rdMolTransforms
from rdkit.Chem.Draw import rdMolDraw2D, rdDepictor
import lib.layers.wrappers.cnf_regularization as reg_lib
import lib.spectral_norm as spectral_norm
import lib.layers as layers
from lib.layers.odefunc import divergence_bf, divergence_approx
from train_misc import standard_normal_logprob
from train_misc import set_cnf_options, count_nfe, count_parameters, count_total_time
from train_misc import add_spectral_norm, spectral_norm_power_iteration
from train_misc import create_regularization_fns, get_regularization, append_regularization_to_log
from rdkit import Chem,DataStructs
from scipy.special import softmax
from graphflow_model_function import *
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
from scipy.stats import multivariate_normal
from torch_geometric.transforms import RadiusGraph
from torch_geometric.nn import radius_graph
from torch_geometric.nn import knn_graph


def get_map(Final_node_feature):
    x,y = np.mgrid[0:20:0.4,-4:4:0.4]
    pos = np.dstack((x, y))
    for idx_node,batch_num in enumerate(Final_node_feature):
        #for idx_channel,channel_num in enumerate(batch_num):
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111)
            ax2.contourf(x, y, batch_num.detach().numpy())
            plt.savefig("Element"+str(idx_node)+".png")
            
def get_trajectory_node_wise(z):
    final_density = []
    time_val = [i for i in range(z.shape[0])]
    for channel in range(z.shape[2]):
        one_node_density = []
        for time_step in range(z.shape[0]):
            time_density = z[time_step]
            one_node_density.append(time_density[0][channel].item())
            
        final_density.append(one_node_density)

    return np.array(final_density),time_val 
    
def get_2D_JT_cord(smiles,top_ring_attributes,rings_vocab):
    
    mol = Chem.MolFromSmiles(smiles)
    ring_system = GetRingSystems(mol)
    AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()
    atoms_belonging_to_ring = []
    ring_belonging_to_ring = []
    
    for ring in ring_system:
            submol = Chem.MolFragmentToSmiles(mol,atomsToUse=list(ring),kekuleSmiles=True)
            for list_ring in rings_vocab:
                mol1 = Chem.MolFromSmiles(submol)
                mol2 = Chem.MolFromSmiles(list_ring)
                fp1 = Chem.RDKFingerprint(mol1)
                fp2 = Chem.RDKFingerprint(mol2)
                if DataStructs.TanimotoSimilarity(fp1,fp2) == 1:
                    ring_belonging_to_ring.append(ring)
                    for atom in ring:
                        atoms_belonging_to_ring.append(atom)
                        
    
    
    x_cord = []
    y_cord = []

    for i, atom in enumerate(mol.GetAtoms()):
            positions = conf.GetAtomPosition(i)
            #print(atom.GetSymbol(), positions.x, positions.y, positions.z)
            if i in atoms_belonging_to_ring: continue
            x_cord.append(positions.x)
            y_cord.append(positions.y)

            
    for ring in ring_belonging_to_ring:
        positions = conf.GetAtomPosition(sorted(ring)[0])
        x_cord.append(positions.x)
        y_cord.append(positions.y)
        
        
    
    r = np.sqrt(np.array(x_cord)**2+np.array(y_cord)**2)
    t = np.arctan2(np.array(y_cord),np.array(x_cord))
    return r,t



def get_3D_JT_cord(smiles,top_ring_attributes,rings_vocab,rdkit_mol,conf):
    
    mol = Chem.MolFromSmiles(smiles)
    ring_system = GetRingSystems(mol)
    atoms_belonging_to_ring = []
    ring_belonging_to_ring = []
    
    for ring in ring_system:
            submol = Chem.MolFragmentToSmiles(mol,atomsToUse=list(ring),kekuleSmiles=True)
            for list_ring in rings_vocab:
                mol1 = Chem.MolFromSmiles(submol)
                mol2 = Chem.MolFromSmiles(list_ring)
                fp1 = Chem.RDKFingerprint(mol1)
                fp2 = Chem.RDKFingerprint(mol2)
                if DataStructs.TanimotoSimilarity(fp1,fp2) == 1:
                    ring_belonging_to_ring.append(ring)
                    for atom in ring:
                        atoms_belonging_to_ring.append(atom)
                        
    
    
    x_cord = []
    y_cord = []
    z_cord = []
    for i, atom in enumerate(rdkit_mol.GetAtoms()):
            positions = conf.GetAtomPosition(i)
            #print(atom.GetSymbol(), positions.x, positions.y, positions.z)
            if i in atoms_belonging_to_ring: continue
            x_cord.append(positions.x)
            y_cord.append(positions.y)
            z_cord.append(positions.z)
            
    for ring in ring_belonging_to_ring:
        positions = conf.GetAtomPosition(sorted(ring)[0])
        x_cord.append(positions.x)
        y_cord.append(positions.y)
        z_cord.append(positions.z)
        
        
    
    r,t,z = cartesian_to_spherical(np.array(x_cord).reshape(-1,1),np.array(y_cord).reshape(-1,1),np.array(z_cord).reshape(-1,1))
    return np.array(r),np.array(t),np.array(z) 



def get_3D_cylinder(conf,rdkit_mol):
    
    x_cord = []
    y_cord = []
    z_cord = []
    for i, atom in enumerate(rdkit_mol.GetAtoms()):
        positions = conf.GetAtomPosition(i)
        #print(atom.GetSymbol(), positions.x, positions.y, positions.z)
        x_cord.append(positions.x)
        y_cord.append(positions.y)
        z_cord.append(positions.z)
        
    r,t,z = cartesian_to_spherical(np.array(x_cord).reshape(-1,1),np.array(y_cord).reshape(-1,1),np.array(z_cord).reshape(-1,1))
    return np.array(r),np.array(t),np.array(z)


def get_2D_cylinder(smiles):
    rdkit_mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(rdkit_mol)
    conf = rdkit_mol.GetConformer()
    x_cord = []
    y_cord = []
    for i, atom in enumerate(rdkit_mol.GetAtoms()):
            positions = conf.GetAtomPosition(i)
            #print(atom.GetSymbol(), positions.x, positions.y, positions.z)
            x_cord.append(positions.x)
            y_cord.append(positions.y)

    r = np.sqrt(np.array(x_cord)**2+np.array(y_cord)**2)
    t = np.arctan2(np.array(y_cord),np.array(x_cord))
    return r,t



def get_ring_info(Smiles,rings_vocab,ring_system):
    mol = Chem.MolFromSmiles(Smiles)
    ring_system = GetRingSystems(mol)
    ring_true_system = []
    ring_true_vocab = []
    for idx_ring,ring in enumerate(ring_system):
        submol = Chem.MolFragmentToSmiles(mol,atomsToUse=list(ring),kekuleSmiles=True)
        for idx_vocab,old in enumerate(rings_vocab):
            mol1 = Chem.MolFromSmiles(submol)
            mol2 = Chem.MolFromSmiles(old)
            fp1 = Chem.RDKFingerprint(mol1)
            fp2 = Chem.RDKFingerprint(mol2)
            if DataStructs.TanimotoSimilarity(fp1,fp2) == 1: 
                ring_true_vocab.append(idx_vocab)
                ring_true_system.append(idx_ring)
    return ring_true_vocab,ring_true_system

def get_ring_bond_info(mol,ring_system):
    Bond_info = []
    Ring_info = []
    Ring_info_total = []
    Bond_info_total = []
    Org = []
    
    for bond in range(len(mol.GetBonds())):
            First = False
            Second = False
            present = -1
            temp = []
            a1 = mol.GetBonds()[bond].GetBeginAtom().GetIdx()
            a2 = mol.GetBonds()[bond].GetEndAtom().GetIdx()
            Org.append((a1,a2))
            for idx_ring,ring in enumerate(ring_system):
                if a1 in list(ring) and a2 in list(ring): 
                    Bond_info.append((a1,a2))
                    Ring_info.append(idx_ring)



    for edge in Org:
        if edge not in Bond_info:
            Bond_info.append(edge)
            Ring_info.append(-1)
            
    return zip(Bond_info,Ring_info) 

def get_trajectory(Z_start,Z_end):
    
    Z_start_flatten = Z_start.flatten()
    
    for node_num in range(len(Z_end)):
            for entry in range(Z_end.shape[1]):
                if Z_end[node_num][entry] < 0: Z_end[node_num][entry] = abs(Z_end[node_num][entry])
                
    Z_end = Z_end/Z_end.sum(axis=1,keepdims=1)
    plt.hist(Z_end.flatten(),density=True)
    plt.xticks([])
    plt.yticks([])
    plt.savefig("hist_end.png")
    plt.clf()
    Z_start_flatten = np.random.standard_normal(5000)
    plt.hist(Z_start_flatten,density=True)
    plt.xticks([])
    plt.yticks([])
    plt.savefig("hist_start.png")



def tensorize_molecule(edges,nodes,Unique_elements,rings_vocab,pos_node):
    

    num_nodes = len(nodes)
    total_labels = Unique_elements+rings_vocab
    Node_feature_input = np.zeros((num_nodes,len(total_labels)))
    #Node_feature_input = np.zeros((num_nodes,len(Unique_elements)))
    final = []
    for idx in range(num_nodes):
        final.append(np.random.standard_normal(size=(1,len(total_labels))))
        
    Node_feature_final = np.array(final).reshape(num_nodes,len(total_labels))
    #Node_feature_final = Node_feature_final/Node_feature_final.sum(axis=1,keepdims=1)
      
    for node in range(num_nodes):
        for unelm in range(len(total_labels)):
            if nodes[node] == total_labels[unelm]:
                Node_feature_input[node][unelm] = 1
                
    Node_feature_input = (1-0.1)*Node_feature_input + 0.1*np.random.uniform(0,1,size=(num_nodes,len(total_labels)))    
    Final_pred = torch.from_numpy(Node_feature_final)
    Final_input = torch.from_numpy(Node_feature_input)
    Position_features = torch.from_numpy(pos_node)    
    edge_start = []
    edge_finish = []
    for i in edges:
        edge_start.append(i[0])
        edge_finish.append(i[1])
    Edge_index = torch.tensor([edge_start,edge_finish])
    order = torch.ones((Edge_index.shape[1],1))
    data = Data(x=Final_input, edge_index=Edge_index,y=Final_pred,pos=Position_features,edge_attr=order)
    
    return data 


def tensorize_molecule_eval(edges,nodes,Unique_elements,rings_vocab,pos_node):
    

    num_nodes = len(nodes)
    total_labels = Unique_elements+rings_vocab
    Node_feature_input = np.zeros((num_nodes,len(total_labels)))
    final = []
    for idx in range(num_nodes):
        final.append(np.random.standard_normal(size=(1,len(total_labels))))
        
    Node_feature_final = np.array(final).reshape(num_nodes,len(total_labels))
      
    for node in range(num_nodes):
        for unelm in range(len(total_labels)):
            if nodes[node] == total_labels[unelm]:
                Node_feature_input[node][unelm] = 1 
                
    Node_feature_input = (1-0.1)*Node_feature_input + 0.1*np.random.uniform(0,1,size=(num_nodes,len(total_labels)))     
    Final_pred = torch.from_numpy(Node_feature_final)
    Final_input = torch.from_numpy(Node_feature_input)
    Position_features = torch.from_numpy(pos_node)    
    edge_start = []
    edge_finish = []
    for i in edges:
        edge_start.append(i[0])
        edge_finish.append(i[1])
    Edge_index = torch.tensor([edge_start,edge_finish])
    order = torch.ones((Edge_index.shape[1],1))
    data = Data(x=Final_pred, edge_index=Edge_index,y=Final_input,pos=Position_features,edge_attr=order)
    
    return data 







def get_decomposed_mol(smiles,top_rings_attributes,rings_vocab):
    
    Edges_mid = []
    Org = []
    Edges_in_ring = []
    Index_ring = []
    mol = Chem.MolFromSmiles(smiles)
    ring_system = GetRingSystems(mol)
    BR_info = get_ring_bond_info(mol,ring_system)
    
    ring_label_vocab,ring_label_mol = get_ring_info(smiles,rings_vocab,ring_system)
    #print("Ring system",ring_label_vocab,ring_label_mol)
    #print("Ring system atoms",ring_system)
    for bond in BR_info:
            Org.append(bond[0])    
            if bond[1] == -1:
                Edges_mid.append(bond[0])

            else:
                if bond[1] not in ring_label_mol:
                    Edges_mid.append(bond[0]) 

                else:
                    Edges_in_ring.append(bond[0])
                    Index_ring.append(bond[1])

    Edges_final = []
    for edge in Edges_mid:
        First = False
        Second = False
        for idx,i in enumerate(ring_label_mol):
            if edge[0] in list(ring_system[i]): 
                First = True
                idx1 = idx

            if edge[1] in list(ring_system[i]): 
                Second = True
                idx2 = idx

        if First == True and Second == True:
            Edges_final.append((list(ring_system[idx1])[0],list(ring_system[idx2])[0]))
        if Second == True and First == False:
            Edges_final.append((edge[0],list(ring_system[idx2])[0]))
        if Second == False and First == True:
            Edges_final.append((list(ring_system[idx1])[0],edge[1]))
        if Second == False and First == False:
            Edges_final.append((edge[0],edge[1]))  
            
    #print("Edges_final",Edges_final)
    smile_mol = read_smiles(smiles)
    node_atoms = list(smile_mol.nodes(data='element'))
    atom_final_node = []
    total_node_attributes = []
    for i,j in Edges_final:
        total_node_attributes.append(i)
        total_node_attributes.append(j)

    Unique_attributes = set(total_node_attributes)
    Node_attributes = []
    for idx,i in enumerate(Unique_attributes):
        Node_attributes.append([i,node_atoms[i][1],idx])

    for idx_node,node in enumerate(Node_attributes):
        for idx,i in enumerate(ring_label_mol): 
            if node[0] in list(ring_system[i]): Node_attributes[idx_node][1] = top_rings_attributes[ring_label_vocab[idx]]

    new_node_num = [num for num in range(len(Node_attributes))]
    new_edges = []
    for atom in Node_attributes:
        atom_final_node.append(atom[1])    

    for idx_edge, edge in enumerate(Edges_final):
        for start_node in Node_attributes:
            if edge[0] == start_node[0]:
                for end_node in Node_attributes:
                    if edge[1] == end_node[0]:
                        new_edges.append((start_node[2],end_node[2])) 
                        
    #print("Org",Org)
    #print("Edges_mid",Edges_mid)
    #print("Edges in ring",Edges_in_ring)
    #print("Edges final",Edges_final)
    #print("Index ring",Index_ring)
    #print("New Edges",new_edges)
    #print("Tree Node attributes",atom_final_node)
    polar_r,polar_angle = get_2D_JT_cord(smiles,top_rings_attributes,rings_vocab)
    node_pos_features = np.concatenate((polar_r.reshape(-1,1),polar_angle.reshape(-1,1)),axis=1)
    
    return new_edges,atom_final_node,node_pos_features

def get_decomposed_mol_3D(smiles,top_rings_attributes,rings_vocab,rdkit_mol,conf):
    
    Edges_mid = []
    Org = []
    Edges_in_ring = []
    Index_ring = []
    mol = Chem.MolFromSmiles(smiles)
    ring_system = GetRingSystems(mol)
    BR_info = get_ring_bond_info(mol,ring_system)
    
    ring_label_vocab,ring_label_mol = get_ring_info(smiles,rings_vocab,ring_system)
    #print("Ring system",ring_label_vocab,ring_label_mol)
    #print("Ring system atoms",ring_system)
    for bond in BR_info:
            Org.append(bond[0])    
            if bond[1] == -1:
                Edges_mid.append(bond[0])

            else:
                if bond[1] not in ring_label_mol:
                    Edges_mid.append(bond[0]) 

                else:
                    Edges_in_ring.append(bond[0])
                    Index_ring.append(bond[1])

    Edges_final = []
    for edge in Edges_mid:
        First = False
        Second = False
        for idx,i in enumerate(ring_label_mol):
            if edge[0] in list(ring_system[i]): 
                First = True
                idx1 = idx

            if edge[1] in list(ring_system[i]): 
                Second = True
                idx2 = idx

        if First == True and Second == True:
            Edges_final.append((list(ring_system[idx1])[0],list(ring_system[idx2])[0]))
        if Second == True and First == False:
            Edges_final.append((edge[0],list(ring_system[idx2])[0]))
        if Second == False and First == True:
            Edges_final.append((list(ring_system[idx1])[0],edge[1]))
        if Second == False and First == False:
            Edges_final.append((edge[0],edge[1]))  
            
    #print("Edges_final",Edges_final)
    smile_mol = read_smiles(smiles)
    node_atoms = list(smile_mol.nodes(data='element'))
    atom_final_node = []
    total_node_attributes = []
    for i,j in Edges_final:
        total_node_attributes.append(i)
        total_node_attributes.append(j)

    Unique_attributes = set(total_node_attributes)
    Node_attributes = []
    for idx,i in enumerate(Unique_attributes):
        Node_attributes.append([i,node_atoms[i][1],idx])

    for idx_node,node in enumerate(Node_attributes):
        for idx,i in enumerate(ring_label_mol): 
            if node[0] in list(ring_system[i]): Node_attributes[idx_node][1] = top_rings_attributes[ring_label_vocab[idx]]

    new_node_num = [num for num in range(len(Node_attributes))]
    new_edges = []
    for atom in Node_attributes:
        atom_final_node.append(atom[1])    

    for idx_edge, edge in enumerate(Edges_final):
        for start_node in Node_attributes:
            if edge[0] == start_node[0]:
                for end_node in Node_attributes:
                    if edge[1] == end_node[0]:
                        new_edges.append((start_node[2],end_node[2])) 
                        
    #print("Org",Org)
    #print("Edges_mid",Edges_mid)
    #print("Edges in ring",Edges_in_ring)
    #print("Edges final",Edges_final)
    #print("Index ring",Index_ring)
    #print("New Edges",new_edges)
    #print("Tree Node attributes",atom_final_node)
    polar_r,polar_angle,polar_z = get_3D_JT_cord(smiles,top_rings_attributes,rings_vocab,rdkit_mol,conf)
    node_pos_features = np.concatenate((polar_r.reshape(-1,1),polar_angle.reshape(-1,1),polar_z.reshape(-1,1)),axis=1)
    
    return new_edges,atom_final_node,node_pos_features

def get_greater(index,list_val):
    descend_order = []
    list_val_update = []
    for idx, val in enumerate(list_val):
        if index > val:
            descend_order.append(idx)
            list_val_update.append(val)
    return descend_order,list_val_update






def check_validity_JT(data,pred,Unique_elements,top_ring_attributes,num,itr,rings_vocab):
    
    for node_num in range(len(pred)):
            for entry in range(len(Unique_elements+top_ring_attributes)):
                if pred[node_num][entry] < 0: pred[node_num][entry] = abs(pred[node_num][entry])

    pred = pred/pred.sum(axis=1,keepdims=1)
    #print("Final",pred)
    #pred = softmax(pred, axis=1)
    Pred =  np.argmax(pred,axis=1)
    conec = data.edge_index.numpy()
    tree_conec = []
    for i,j in zip(conec[0],conec[1]):
        tree_conec.append((i,j))
    total_elements = Unique_elements + top_ring_attributes
    pred_mol = []
    for k in Pred:
        pred_mol.append(str(total_elements[k]))
        
    position = []
    
    for atom in pred_mol:
        if atom in top_ring_attributes: position.append(True)
        else: position.append(False)
        
    node_index = [i for i in range(len(pred_mol))] 
    #print(pred_mol,position,tree_conec)
    num = 0
    ring_idx = 0
    edges = []
    node_attributes = []
    num = 0
    
    
    for atom,is_ring,idx_node in zip(pred_mol,position,node_index):
        if is_ring:
            for idx,ring in enumerate(top_ring_attributes):
                if atom == ring: 
                    submol = read_smiles(rings_vocab[idx])
                    submol_edges = list(submol.edges(data='order'))
                    for entry in range(len(submol_edges)):
                        edges.append((submol_edges[entry][0] + num,submol_edges[entry][1] + num,submol_edges[entry][2]))
                    for node in submol.nodes(data='element'):
                        node_attributes.append(node[1])
                    num = num + len(submol.nodes(data='element'))-1
                    ring_idx = ring_idx +1
                
            if position[ring_idx]: edges.append((edges[len(edges)-1][0]+1,edges[len(edges)-1][1]+1,1,"Add"))
        
        else:
            node_attributes.append(pred_mol[idx_node])
            if ring_idx == 0:
                for i,j in tree_conec:
                    if i==ring_idx : 
                        edges.append((i,j,1,"Else"))
                    
            else:
                for i,j in tree_conec:
                    if  True not in position and ring_idx + 1 == len(node_index): continue
                    if j==ring_idx : 
                        edges.append((num,num+1,1,"Else"))

            num = num + 1
            ring_idx = ring_idx +1
            
    print(pred_mol,edges)        
    mol = nx.Graph()
    edge_list =[]
    for i in edges:
        edge_list.append((i[0],i[1]))
    
    
    mol.add_edges_from(edge_list)
    gen = [] 
    gen_smiles = ""
    for i in edges:
        mol.edges[i[0], i[1]]['order'] = i[2]

    for idx,atom in enumerate(node_attributes):
        gen_smiles = gen_smiles+str(atom)
        mol.nodes[idx]['element'] = str(atom)
    #print("Edges ",edges)    
    #print(pred_mol,len(pred_mol),node_attributes,len(node_attributes),mol.nodes(data='element'))
    fill_valence(mol,respect_hcount=True,max_bond_order=3) 
    
    m = Chem.MolFromSmiles(write_smiles(mol),sanitize=True)
    if m is not None: 
        #Draw.MolToFile(m,'generated_molecule_images/QM9/molecule_test' + str(num) + "_" + str(itr) + ".png")
        valid = 1
    if m is None:
        valid = 0
        
    return valid


def check_validity_JT_lkl(data,pred,Unique_elements,top_ring_attributes,num,itr,rings_vocab):
    
    
    conec = data.edge_index.numpy()
    tree_conec = []
    for i,j in zip(conec[0],conec[1]):
        tree_conec.append((i,j))
    total_elements = Unique_elements + top_ring_attributes
    
    for sample in range(1000):
        pred_mol = []


        for node_num in range(len(pred)):
                for entry in range(len(Unique_elements+top_ring_attributes)):
                    if pred[node_num][entry] < 0: pred[node_num][entry] = abs(pred[node_num][entry])

                prob = pred[node_num]/pred[node_num].sum(axis=0,keepdims=1)
                atom = np.random.choice(total_elements, size = 1, p = prob, replace = False)
                pred_mol.append(str(atom[0]))

        position = []

        for atom in pred_mol:
            if atom in top_ring_attributes: position.append(True)
            else: position.append(False)

        node_index = [i for i in range(len(pred_mol))] 
        #print(pred_mol,position,tree_conec)
        num = 0
        ring_idx = 0
        edges = []
        node_attributes = []
        num = 0


        for atom,is_ring,idx_node in zip(pred_mol,position,node_index):
            if is_ring:
                for idx,ring in enumerate(top_ring_attributes):
                    if atom == ring: 
                        submol = read_smiles(rings_vocab[idx])
                        submol_edges = list(submol.edges(data='order'))
                        for entry in range(len(submol_edges)):
                            edges.append((submol_edges[entry][0] + num,submol_edges[entry][1] + num,submol_edges[entry][2]))
                        for node in submol.nodes(data='element'):
                            node_attributes.append(node[1])
                        num = num + len(submol.nodes(data='element'))-1
                        ring_idx = ring_idx +1

                if position[ring_idx]: edges.append((edges[len(edges)-1][0]+1,edges[len(edges)-1][1]+1,1,"Add"))

            else:
                node_attributes.append(pred_mol[idx_node])
                if ring_idx == 0:
                    for i,j in tree_conec:
                        if i==ring_idx : 
                            edges.append((i,j,1,"Else"))

                else:
                    for i,j in tree_conec:
                        if  True not in position and ring_idx + 1 == len(node_index): continue
                        if j==ring_idx : 
                            edges.append((num,num+1,1,"Else"))

                num = num + 1
                ring_idx = ring_idx +1

        print(pred_mol,tree_conec)        
        print(node_attributes,len(node_attributes))
        print(edges)
        mol = nx.Graph()
        edge_list =[]
        for i in edges:
            edge_list.append((i[0],i[1]))


        mol.add_edges_from(edge_list)
        gen = [] 
        gen_smiles = ""
        for i in edges:
            mol.edges[i[0], i[1]]['order'] = i[2]

        for idx,atom in enumerate(node_attributes):
            gen_smiles = gen_smiles+str(atom)
            mol.nodes[idx]['element'] = str(atom)
        #print("Edges ",edges)    
        #print(pred_mol,len(pred_mol),node_attributes,len(node_attributes),mol.nodes(data='element'))
        fill_valence(mol,respect_hcount=True,max_bond_order=3) 
        print("Sample ",sample,node_attributes)
        m = Chem.MolFromSmiles(write_smiles(mol),sanitize=True)
        if m is not None: 
            #Draw.MolToFile(m,'generated_molecule_images/QM9/molecule_test' + str(num) + "_" + str(itr) + ".png")
            valid = 1
        if m is None:
            valid = 0

    return valid




def check_validity_ZINC(data,pred,Unique_elements,num,itr,order):
    mol = nx.Graph()
    edges = data.edge_index.numpy()
    edge_list = []
    valid = 0
    for i in range(len(edges[0])):
        edge_list.append((edges[0][i],edges[1][i]))
        
    gen = [] 
    gen_smiles = ""
    #print(pred)
    pred = softmax(pred, axis=1)
    #print(pred)
    mol.add_edges_from(edge_list)
    Pred =  np.argmax(pred,axis=1)
    
    for k in Pred:
        gen.append(str(Unique_elements[k]))
        
    for idx in range(len(gen)):
        gen_smiles = gen_smiles+str(gen[idx])
        mol.nodes[idx]['element'] = str(gen[idx])
        
    for i in range(len(edges[0])):
        mol.edges[edges[0][i], edges[1][i]]['order'] = order[i].numpy()[0]
        
    #print(gen)
    fill_valence(mol,respect_hcount=True,max_bond_order=3)
    m = Chem.MolFromSmiles(write_smiles(mol),sanitize=True)
    print(m)
    if m is not None: 
        #Draw.MolToFile(m,'generated_molecule_images/QM9/molecule_test' + str(num) + "_" + str(itr) + ".png")
        valid = 1
    if m is None:
        valid = 0
    return write_smiles(mol),valid



def check_validity_QM9(data,pred,Unique_elements,num,itr,order):
    mol = nx.Graph()
    edges = data.edge_index.numpy()
    edge_list = []
    valid = 0
    for i in range(len(edges[0])):
        edge_list.append((edges[0][i],edges[1][i]))
        
    gen = [] 
    gen_smiles = ""
    #print(pred)
    pred = softmax(pred, axis=1)
    #print(pred)
    mol.add_edges_from(edge_list)
    Pred =  np.argmax(pred,axis=1)
    
    for k in Pred:
        gen.append(str(Unique_elements[k]))
        
    for idx in range(len(gen)):
        gen_smiles = gen_smiles+str(gen[idx])
        mol.nodes[idx]['element'] = str(gen[idx])
        
    for i in range(len(edges[0])):
        mol.edges[edges[0][i], edges[1][i]]['order'] = order[i].numpy()[0]
        
    #print(gen)
    fill_valence(mol,respect_hcount=True,max_bond_order=3)
    m = Chem.MolFromSmiles(write_smiles(mol),sanitize=True)
    print(m)
    if m is not None: 
        #Draw.MolToFile(m,'generated_molecule_images/QM9/molecule_test' + str(num) + "_" + str(itr) + ".png")
        valid = 1
    if m is None:
        valid = 0
    return write_smiles(mol),valid


def reconstruct_valid_JT(data,pred,Unique_elements,top_ring_attributes,num,itr,rings_vocab):
    
    position = []
    #print(len(Unique_elements),pred.shape)
    gen = pred/pred.sum(axis=1,keepdims=1)
    total_elements = Unique_elements + top_ring_attributes
    Pred =  np.argmax(gen,axis=1)
    pred = []
   
    for k in Pred:
        #print(k)
        pred.append(str(total_elements[k]))
    
    edges = data.edge_index.numpy()
    tree_conec = []
    for i in range(len(edges[0])):
        tree_conec.append((edges[0][i],edges[1][i]))
    #ßprint(tree_conec)
    node_index = [i for i in range(len(pred))]
    
    for atom in pred:
        if atom in top_ring_attributes: position.append(True)
        else: position.append(False)
    
    num = 0
    ring_idx = 0
    edges = []
    node_attributes = []
    num = 0
    sub_ring_edges = []
    starting_pos = []
    ending_pos = []
    org_node = []
    decomposed_node_attributes = []
    
    
    
    for i in pred:
        if i not in top_ring_attributes:
            decomposed_node_attributes.append(i)
        
        else:
            for idx,ring in enumerate(top_ring_attributes): 
                if i == ring:
                    submol = read_smiles(rings_vocab[idx])
                    elements = nx.get_node_attributes(submol, name = "element")
                    for k in range(len(elements)):
                        decomposed_node_attributes.append(elements[k])
                        
    for atom,is_ring,idx_node in zip(pred,position,node_index):
        if is_ring:
            for idx,ring in enumerate(top_ring_attributes):
                if atom == ring:
                    submol = read_smiles(rings_vocab[idx])
                    submol_edges = list(submol.edges(data='order'))
                    org_node.append(idx_node)
                    sub_ring_edges.append((idx_node,submol_edges,len(submol.nodes)))
                    
                    
    #print(sub_ring_edges)
    update_edges = []
    final_edges = []
    edge_start = []
    edge_start_final = []
    edge_end = []
    edge_end_final = []
    for i,j in tree_conec:
        edge_start.append(i)
        edge_end.append(j)
    
    ending_values = []
    for entry,entry_values in enumerate(sub_ring_edges):
    
        if entry == 0:
            starting = entry_values[0]
            starting_pos.append(starting)
            ending_pos.append(entry_values[0] + entry_values[2] - 1)
            #print(starting)
        else:
            n_ring_atoms = 0
            curr_node = entry_values[0]
            for hist_entry in range(0,entry):
                prev_node =  sub_ring_edges[hist_entry][0]
                prev_num_atoms = sub_ring_edges[hist_entry][2]
                n_ring_atoms = n_ring_atoms + prev_num_atoms
            
            correction_factor = n_ring_atoms + curr_node - entry
            starting_pos.append(correction_factor)
            ending_pos.append(correction_factor + entry_values[2]- 1)
            #print(correction_factor)
            
    #print("Starting and ending pos",starting_pos,ending_pos)
    #print(org_node)
    #print(tree_conec)
    old_new_connections = []
    for i,j in tree_conec:
        if i in org_node and j in org_node: continue
        greater_index_e1,node_val1 = get_greater(i,org_node)
        greater_index_e2,node_val2 = get_greater(j,org_node)
        
        if len(greater_index_e1) == 0 and len(greater_index_e2) == 0: continue
    
        if len(greater_index_e1) == 0 and len(greater_index_e2) > 0:
            ring_index = greater_index_e2[len(greater_index_e2)-1]
            old_node_val = node_val2[len(node_val2)-1] 
            factor = ending_pos[ring_index] + j - old_node_val
            #print(i,j,factor)
            old_new_connections.append((j,factor))
        
        if len(greater_index_e1) > 0 and len(greater_index_e2) == 0:
            ring_index = greater_index_e1[len(greater_index_e1)-1]
            old_node_val = node_val1[len(node_val1)-1] 
            factor = ending_pos[ring_index] + j - old_node_val
            #print(i,j,factor)
            old_new_connections.append((i,factor))
    
        if len(greater_index_e1) > 0 and len(greater_index_e2) > 0:
            
            if i in org_node:
                ring_index2 = greater_index_e2[len(greater_index_e2)-1] 
                old_node_val2 = node_val2[len(node_val2)-1]
                factor2 = ending_pos[ring_index2] + j - old_node_val2
                #print(i,j,factor2,"i in ring")
                old_new_connections.append((j,factor2))
             
            if j in org_node:
                ring_index1 = greater_index_e1[len(greater_index_e1)-1] 
                old_node_val1 = node_val1[len(node_val1)-1]
                factor1 = ending_pos[ring_index1] + i - old_node_val1
                #print(i,j,factor1,"j in ring")
                old_new_connections.append((i,factor1))
                
            if i not in org_node and j not in org_node:
                ring_index1 = greater_index_e1[len(greater_index_e1)-1] 
                ring_index2 = greater_index_e2[len(greater_index_e2)-1]
                old_node_val1 = node_val1[len(node_val1)-1]
                old_node_val2 = node_val2[len(node_val2)-1]
                factor1 = ending_pos[ring_index1] + i - old_node_val1
                factor2 = ending_pos[ring_index2] + j - old_node_val2
                #print(i,j,factor1,factor2,"Both not in ring")
                old_new_connections.append((i,factor1))
                old_new_connections.append((j,factor2))
                
                
    #print(list(set(old_new_connections)))
    #print(edge_start)
    #print(edge_end)
    for idx,entry_edge in enumerate(zip(edge_start,edge_end)):
        for idx_org,org_val in enumerate(org_node):
            if entry_edge[0] == org_val:
                edge_start[idx] = ending_pos[idx_org]

            if entry_edge[1] == org_val:
                edge_end[idx] = starting_pos[idx_org]
            
        for idx_old,old_map in enumerate(old_new_connections):
            if old_map[0] == entry_edge[0]:
                edge_start[idx] = old_map[1]
            
            if old_map[0] == entry_edge[1]:
                edge_end[idx] = old_map[1]   

    new_ring_edges = []
    for entry,entry_values in enumerate(sub_ring_edges):
        for edge_entries in entry_values[1]:
            new_ring_edges.append((edge_entries[0] + starting_pos[entry],edge_entries[1]+starting_pos[entry],edge_entries[2]))
    max_i = 0
    max_j = 0 
    for i,j in zip(edge_start,edge_end):
        final_edges.append((i,j,1))
        if max_i<i:
            max_i = i
        if max_j<j:
            max_j=j
            
    
    final_edges = final_edges + new_ring_edges
    #print("Edges after fully updating node values",final_edges)
    #print("Nodes",decomposed_node_attributes)
    final_edges,correc_syn = correct(final_edges,decomposed_node_attributes)
    return final_edges,correc_syn,decomposed_node_attributes



def check_JT(final_edges,decomposed_node_attributes):
    mol = nx.Graph()
    edge_list = []
    for i in final_edges:
        edge_list.append((i[0],i[1]))
    mol.add_edges_from(edge_list)
    for idx in range(len(decomposed_node_attributes)):
        mol.nodes[idx]['element'] = str(decomposed_node_attributes[idx])

    #for i in final_edges:
    #    mol.edges[i[0], i[1]]['order'] = i[2]

    fill_valence(mol,respect_hcount=True,max_bond_order=3)
    m = Chem.MolFromSmiles(write_smiles(mol),sanitize=True)
    print(m)
    if m is not None: 
        #Draw.MolToFile(m,'generated_molecule_images/QM9/molecule_test' + str(num) + "_" + str(itr) + ".png")
        valid = 1
    if m is None:
        valid = 0
    return write_smiles(mol),valid
    
    
def check_JT_final(final_edges,decomposed_node_attributes):
    mol = Chem.RWMol()

    # add atoms to mol and keep track of index
    node_to_idx = {}
    for i in range(len(decomposed_node_attributes)):
        a = Chem.Atom(decomposed_node_attributes[i])
        molIdx = mol.AddAtom(a)
        node_to_idx[i] = molIdx
    
    for ix,iy,bond in set(final_edges):
            # only traverse half the matrix
            # add relevant bond type (there are many more of these)
            if bond == 0:
                continue
            elif bond == 1:
                bond_type = Chem.rdchem.BondType.SINGLE
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
            elif bond == 2:
                bond_type = Chem.rdchem.BondType.DOUBLE
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

    # Convert RWMol to Mol object
    mol = mol.GetMol()
    smiles_gen = Chem.MolToSmiles(mol)
    m = Chem.MolFromSmiles(Chem.MolToSmiles(mol),sanitize=True)
    print(m)
    if m is not None: 
        #Draw.MolToFile(m,'generated_molecule_images/QM9/molecule_test' + str(num) + "_" + str(itr) + ".png")
        valid = 1
    if m is None:
        valid = 0
    return Chem.MolToSmiles(mol),valid



def correct(final_edges,nodes):
    final_mod_edges = final_edges.copy()
    start_node = [i for i,_,_ in final_edges]
    end_node = [j for _,j,_ in final_edges]
    correct_syn = False
    for node_num in range(len(nodes)):
        if node_num in start_node or node_num in end_node: continue
        final_mod_edges.append((node_num,node_num+1,1))
        correct_syn  = True
    return final_mod_edges,correct_syn    
    

def check_novelty(gen_smiles, train_smiles):
    if len(gen_smiles) == 0:
        novel_ratio = 0.
    else:
        duplicates = [1 for mol in gen_smiles if mol in train_smiles]
        novel = len(gen_smiles) - sum(duplicates)
        novel_ratio = novel*100./len(gen_smiles)
    return novel_ratio



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


def Average(lst):
    return sum(lst) / len(lst)


def get_evaluated(pred,true):
    correct = 0
    wrong = 0
    total = 0
    pred_arg = np.argmax(pred,axis=1)
    true_arg = torch.argmax(true,dim=1)
    for i,j in zip(pred_arg,true_arg):
        if i == j:
            correct = correct + 1
        else:
            wrong = wrong + 1
        total = total + 1
    
    return float(correct/total), float(wrong/total)
        




def reverse_transform(model,data_samples,ntimes=101, memory=0.01, device='cpu'):
    z_samples = data_samples.x
    with torch.no_grad():
        logp_samples = torch.sum(standard_normal_logprob(z_samples), 1, keepdim=True)
        t = 0
        for cnf in model.chain:
            end_time = (cnf.sqrt_end_time * cnf.sqrt_end_time)
            integration_times = torch.linspace(0, end_time, ntimes)
            z_traj, _ = cnf(data_samples, logp_samples, integration_times=integration_times, reverse=True)
            z_traj = z_traj.cpu().numpy()
            t += 1
            
    return z_traj

def build_model_tabular(c_in,c_out,args,regularization_fns=None):

    hidden_dims = tuple(map(int, args.dims.split("-")))

    def build_cnf():
        diffeq = GraphFlow(c_in,c_out)
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
        return cnf

    chain = [build_cnf() for _ in range(args.num_blocks)]
    if args.batch_norm:
        bn_layers = [layers.MovingBatchNorm1d(dims, bn_lag=args.bn_lag) for _ in range(args.num_blocks)]
        bn_chain = [layers.MovingBatchNorm1d(dims, bn_lag=args.bn_lag)]
        for a, b in zip(chain, bn_layers):
            bn_chain.append(a)
            bn_chain.append(b)
        chain = bn_chain
    model = layers.SequentialFlow(chain)

    set_cnf_options(args, model)

    return model

def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2




def get_unique(Smiles):
    Unique_elements = []
    dall = []
    Adj = []
    for i in Smiles:
        mol = read_smiles(i)
        elements = nx.get_node_attributes(mol, name = "element")
        for k in range(len(elements)):
            dall.append(elements[k])
    
    for val in dall: 
        if val in Unique_elements: 
            continue 
        else:
            Unique_elements.append(val)

    return Unique_elements



def get_graph_data_with_polar_3D(Smiles,Unique_elements,conf,rdkit_mol):

    NF_final = []
    NF_input = []
    mol = read_smiles(Smiles)
    labels = []
    elements = nx.get_node_attributes(mol, name = "element")
    num_nodes = len(elements)
    Node_feature_input = np.zeros((num_nodes,len(Unique_elements)))
    final = []
    for idx in range(num_nodes):
        final.append(np.random.standard_normal(size=(1,len(Unique_elements))))
        
    Node_feature_final = np.array([final]).reshape(num_nodes,len(Unique_elements))
      
    for node in range(num_nodes):
        for unelm in range(len(Unique_elements)):
            if elements[node] == Unique_elements[unelm]:
                Node_feature_input[node][unelm] = 1
                
    Node_feature_input = (1-0.1)*Node_feature_input + 0.1*np.random.uniform(0,1,size=(num_nodes,len(Unique_elements)))
    polar_r,polar_angle,polar_z = get_3D_cylinder(conf,rdkit_mol)
    node_pos_features = np.concatenate((polar_r.reshape(-1,1),polar_angle.reshape(-1,1),polar_z.reshape(-1,1)),axis=1)
    
    Final_pred = torch.from_numpy(Node_feature_final)
    Final_input = torch.from_numpy(Node_feature_input)
    Pos_node = torch.from_numpy(node_pos_features)
        
    Total_edges = list(mol.edges(data='order'))
    edge_start = []
    edge_finish = []
    order = []
    
    for i in Total_edges:
        edge_start.append(i[0])
        edge_finish.append(i[1])
        order.append(i[2])
        
    #node_index = [num for num in range(num_nodes)]
    #for idx_node in node_index:
    #    if idx_node not in edge_start:
    #        for idx_second_node,edge_val in enumerate(edge_finish):
    #            if idx_node == edge_val:
    #                edge_start.append(idx_node)
    #                edge_finish.append(edge_start[idx_second_node])
                 
                    
    #Sorted_Edges = sorted(zip(edge_start,edge_finish))
    #edge_start = [i for i,_ in Sorted_Edges]
    #edge_finish = [i for _,i in Sorted_Edges]
    Edge_index = torch.tensor([edge_start,edge_finish])
    data = Data(x=Final_input, edge_index=Edge_index,y=Final_pred,pos=Pos_node,edge_attr=torch.tensor(order).view(-1,1))
    
    return data,order


def get_graph_data_with_polar_2D(Smiles,Unique_elements):

    NF_final = []
    NF_input = []
    mol = read_smiles(Smiles)
    labels = []
    elements = nx.get_node_attributes(mol, name = "element")
    num_nodes = len(elements)
    Node_feature_input = np.zeros((num_nodes,len(Unique_elements)))
    final = []
    for idx in range(num_nodes):
        final.append(np.random.standard_normal(size=(1,len(Unique_elements))))
        
    Node_feature_final = np.array([final]).reshape(num_nodes,len(Unique_elements))
      
    for node in range(num_nodes):
        for unelm in range(len(Unique_elements)):
            if elements[node] == Unique_elements[unelm]:
                Node_feature_input[node][unelm] = 1
                
    Node_feature_input = (1-0.1)*Node_feature_input + 0.1*np.random.uniform(0,1,size=(num_nodes,len(Unique_elements)))
    polar_r,polar_angle = get_2D_cylinder(Smiles)
    node_pos_features = np.concatenate((polar_r.reshape(-1,1),polar_angle.reshape(-1,1)),axis=1)
    
    Final_pred = torch.from_numpy(Node_feature_final)
    Final_input = torch.from_numpy(Node_feature_input)
    Pos_node = torch.from_numpy(node_pos_features)
        
    Total_edges = list(mol.edges(data='order'))
    edge_start = []
    edge_finish = []
    order = []
    
    for i in Total_edges:
        edge_start.append(i[0])
        edge_finish.append(i[1])
        order.append(i[2])

    Edge_index = torch.tensor([edge_start,edge_finish])
    data = Data(x=Final_input, edge_index=Edge_index,y=Final_pred,pos=Pos_node,edge_attr=torch.tensor(order).view(-1,1))
    
    return data,order


def get_graph_data_with_polar_2D_eval(Smiles,Unique_elements):

    NF_final = []
    NF_input = []
    mol = read_smiles(Smiles)
    labels = []
    elements = nx.get_node_attributes(mol, name = "element")
    num_nodes = len(elements)
    Node_feature_input = np.zeros((num_nodes,len(Unique_elements)))
    final = []
    for idx in range(num_nodes):
        final.append(np.random.standard_normal(size=(1,len(Unique_elements))))
        
    Node_feature_final = np.array([final]).reshape(num_nodes,len(Unique_elements))
      
    for node in range(num_nodes):
        for unelm in range(len(Unique_elements)):
            if elements[node] == Unique_elements[unelm]:
                Node_feature_input[node][unelm] = 1
                
    Node_feature_input = (1-0.1)*Node_feature_input + 0.1*np.random.uniform(0,1,size=(num_nodes,len(Unique_elements)))
    polar_r,polar_angle = get_2D_cylinder(Smiles)
    node_pos_features = np.concatenate((polar_r.reshape(-1,1),polar_angle.reshape(-1,1)),axis=1)
    
    Final_pred = torch.from_numpy(Node_feature_final)
    Final_input = torch.from_numpy(Node_feature_input)
    Pos_node = torch.from_numpy(node_pos_features)
        
    Total_edges = list(mol.edges(data='order'))
    edge_start = []
    edge_finish = []
    order = []
    
    for i in Total_edges:
        edge_start.append(i[0])
        edge_finish.append(i[1])
        order.append(i[2])
        
    #node_index = [num for num in range(num_nodes)]
    #for idx_node in node_index:
    #    if idx_node not in edge_start:
    #        for idx_second_node,edge_val in enumerate(edge_finish):
    #            if idx_node == edge_val:
    #                edge_start.append(idx_node)
    #                edge_finish.append(edge_start[idx_second_node])
                 
                    
    #Sorted_Edges = sorted(zip(edge_start,edge_finish))
    #edge_start = [i for i,_ in Sorted_Edges]
    #edge_finish = [i for _,i in Sorted_Edges]
    Edge_index = torch.tensor([edge_start,edge_finish])
    data = Data(x=Final_pred, edge_index=Edge_index,y=Final_input,pos=Pos_node,edge_attr=torch.tensor(order).view(-1,1))
    
    return data,order




def get_graph_data_with_polar_3D_eval(Smiles,Unique_elements,conf,rdkit_mol):

    NF_final = []
    NF_input = []
    mol = read_smiles(Smiles)
    labels = []
    elements = nx.get_node_attributes(mol, name = "element")
    num_nodes = len(elements)
    Node_feature_input = np.zeros((num_nodes,len(Unique_elements)))
    final = []
    for idx in range(num_nodes):
        final.append(np.random.standard_normal(size=(1,len(Unique_elements))))
        
    Node_feature_final = np.array([final]).reshape(num_nodes,len(Unique_elements))
      
    for node in range(num_nodes):
        for unelm in range(len(Unique_elements)):
            if elements[node] == Unique_elements[unelm]:
                Node_feature_input[node][unelm] = 1
                
    Node_feature_input = (1-0.1)*Node_feature_input + 0.1*np.random.uniform(0,1,size=(num_nodes,len(Unique_elements)))
    polar_r,polar_angle,polar_z = get_3D_cylinder(conf,rdkit_mol)
    node_pos_features = np.concatenate((polar_r.reshape(-1,1),polar_angle.reshape(-1,1),polar_z.reshape(-1,1)),axis=1)
    
    Final_pred = torch.from_numpy(Node_feature_final)
    Final_input = torch.from_numpy(Node_feature_input)
    Pos_node = torch.from_numpy(node_pos_features)
        
    Total_edges = list(mol.edges(data='order'))
    edge_start = []
    edge_finish = []
    order = []
    
    for i in Total_edges:
        edge_start.append(i[0])
        edge_finish.append(i[1])
        order.append(i[2])
        
    #node_index = [num for num in range(num_nodes)]
    #for idx_node in node_index:
    #    if idx_node not in edge_start:
    #        for idx_second_node,edge_val in enumerate(edge_finish):
    #            if idx_node == edge_val:
    #                edge_start.append(idx_node)
    #                edge_finish.append(edge_start[idx_second_node])
                 
                    
    #Sorted_Edges = sorted(zip(edge_start,edge_finish))
    #edge_start = [i for i,_ in Sorted_Edges]
    #edge_finish = [i for _,i in Sorted_Edges]
    Edge_index = torch.tensor([edge_start,edge_finish])
    data = Data(x=Final_pred, edge_index=Edge_index,y=Final_input,pos=Pos_node,edge_attr=torch.tensor(order).view(-1,1))
    
    return data,order

def alternate_strips(batch_size):
    node_index = []
    prob_val= []
    node_pos = []
    x1 = np.random.rand(batch_size) * 4 - 2
    x2_ = np.random.rand(batch_size) - np.random.randint(0, 5, batch_size) * 2 + 4
    x2 = x2_ + (np.floor(x1) % 1)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-2,2))
    
    x2 = min_max_scaler.fit_transform(x2.reshape(-1,1)).flatten()
    final_node_features = np.concatenate([x1[:, None], x2[:, None]], 1) * 2
    Edge_index = radius_graph(torch.Tensor(final_node_features), r=0, loop=False)
    
    x_cord = final_node_features[:,0]
    y_cord = final_node_features[:,1]
    
    r = np.sqrt(np.array(x_cord)**2+np.array(y_cord)**2)
    t = np.arctan2(np.array(y_cord),np.array(x_cord))
    node_pos_features = np.concatenate((r.reshape(-1,1),t.reshape(-1,1)),axis=1)
    #Node_feature_input = np.concatenate((Node_feature_input,node_pos_features),axis=1)
    
    Node_feature_final = np.random.standard_normal(size=(len(final_node_features),2))
    #Node_feature_final = Node_feature_final/Node_feature_final.sum(axis=1,keepdims=1)
    #Node_feature_final = np.concatenate((Node_feature_final,node_pos_features),axis=1)
    
    Final_pred = torch.from_numpy(Node_feature_final).float()
    Final_input = torch.from_numpy(final_node_features).float()
    Pos_node = torch.from_numpy(node_pos_features).float()
    
    data = Data(x=Final_input, edge_index=Edge_index,y=Final_pred,pos=Pos_node)
    return data

def alternate_strips_eval(batch_size):
    node_index = []
    prob_val= []
    node_pos = []
    x1 = np.random.rand(batch_size) * 4 - 2
    x2_ = np.random.rand(batch_size) - np.random.randint(0, 5, batch_size) * 2 + 4
    x2 = x2_ + (np.floor(x1) % 1)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-2,2))
    
    x2 = min_max_scaler.fit_transform(x2.reshape(-1,1)).flatten()
    final_node_features = np.concatenate([x1[:, None], x2[:, None]], 1) * 2
    Edge_index = radius_graph(torch.Tensor(final_node_features), r=0, loop=False)
    
    x_cord = final_node_features[:,0]
    y_cord = final_node_features[:,1]
    
    r = np.sqrt(np.array(x_cord)**2+np.array(y_cord)**2)
    t = np.arctan2(np.array(y_cord),np.array(x_cord))
    node_pos_features = np.concatenate((r.reshape(-1,1),t.reshape(-1,1)),axis=1)
    #Node_feature_input = np.concatenate((Node_feature_input,node_pos_features),axis=1)
    
    Node_feature_final = np.random.standard_normal(size=(len(final_node_features),2))
    #Node_feature_final = Node_feature_final/Node_feature_final.sum(axis=1,keepdims=1)
    #Node_feature_final = np.concatenate((Node_feature_final,node_pos_features),axis=1)
    
    Final_pred = torch.from_numpy(Node_feature_final).float()
    Final_input = torch.from_numpy(final_node_features).float()
    Pos_node = torch.from_numpy(node_pos_features).float()
    
    data = Data(x=Final_pred, edge_index=Edge_index,y=Final_input,pos=Pos_node)
    return data



def alternate_checker_data(batch_size): 
    node_index = []
    prob_val= []
    node_pos = []
    x1 = np.random.rand(batch_size) * 4 - 2
    x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
    x2 = x2_ + (np.floor(x1) % 2)
    final_node_features = np.concatenate([x1[:, None], x2[:, None]], 1) * 2
    
    Edge_index = radius_graph(torch.Tensor(final_node_features), r=0, loop=False)
    
    x_cord = final_node_features[:,0]
    y_cord = final_node_features[:,1]
    
    r = np.sqrt(np.array(x_cord)**2+np.array(y_cord)**2)
    t = np.arctan2(np.array(y_cord),np.array(x_cord))
    node_pos_features = np.concatenate((r.reshape(-1,1),t.reshape(-1,1)),axis=1)
    #Node_feature_input = np.concatenate((Node_feature_input,node_pos_features),axis=1)
    
    final = []
    num_nodes = len(final_node_features)
    for idx in range(num_nodes):
        final.append(np.random.standard_normal(size=(1,2)))
        
    Node_feature_final = np.array([final]).reshape(num_nodes,2)
    #Node_feature_final = Node_feature_final/Node_feature_final.sum(axis=1,keepdims=1)
    #Node_feature_final = np.concatenate((Node_feature_final,node_pos_features),axis=1)
    
    Final_pred = torch.from_numpy(Node_feature_final).float()
    Final_input = torch.from_numpy(final_node_features).float()
    Pos_node = torch.from_numpy(node_pos_features).float()
    
    data = Data(x=Final_input, edge_index=Edge_index,y=Final_pred,pos=Pos_node)
    return data





def alternate_checker_data_eval(batch_size): 
    node_index = []
    prob_val= []
    node_pos = []
    x1 = np.random.rand(batch_size) * 4 - 2
    x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
    x2 = x2_ + (np.floor(x1) % 2)
    final_node_features = np.concatenate([x1[:, None], x2[:, None]], 1) * 2
    
    Edge_index = radius_graph(torch.Tensor(final_node_features), r=0, loop=False)
    
    x_cord = final_node_features[:,0]
    y_cord = final_node_features[:,1]
    
    r = np.sqrt(np.array(x_cord)**2+np.array(y_cord)**2)
    t = np.arctan2(np.array(y_cord),np.array(x_cord))
    node_pos_features = np.concatenate((r.reshape(-1,1),t.reshape(-1,1)),axis=1)
    #Node_feature_input = np.concatenate((Node_feature_input,node_pos_features),axis=1)
    
    final = []
    num_nodes = len(final_node_features)
    for idx in range(num_nodes):
        final.append(np.random.standard_normal(size=(1,2)))
        
    Node_feature_final = np.array([final]).reshape(num_nodes,2)
    #Node_feature_final = Node_feature_final/Node_feature_final.sum(axis=1,keepdims=1)
    #Node_feature_final = np.concatenate((Node_feature_final,node_pos_features),axis=1)
    
    Final_pred = torch.from_numpy(Node_feature_final).float()
    Final_input = torch.from_numpy(final_node_features).float()
    Pos_node = torch.from_numpy(node_pos_features).float()
    
    data = Data(x=Final_pred, edge_index=Edge_index,y=Final_input,pos=Pos_node)
    return data






def alternate_strip_data_grid(): 
    node_index = []
    prob_value= []
    node_pos = []
    prob_value = []
    x_position = []
    y_position = []
    for idx_node in range(20*20):
        x_pos = int(idx_node/20)
        y_pos = int(idx_node%20) 
        layer_val = int(idx_node/40)
        if layer_val%2== 0:
            x_position.append(x_pos)
            y_position.append(y_pos)
            prob_value.append([0,1])
            
        else:
            x_position.append(x_pos)
            y_position.append(y_pos)
            prob_value.append([1,0])
            
       
    pos_features = np.concatenate((np.array(x_position).reshape(-1,1),np.array(y_position).reshape(-1,1)),axis=1)
    pos_features = (pos_features/20)*2 - np.ones((pos_features.shape[0],pos_features.shape[1]))
    Edge_index = knn_graph(torch.Tensor(pos_features), k=2, loop=False)
    
    
    r = np.sqrt(np.array(x_position)**2+np.array(y_position)**2)
    t = np.arctan2(np.array(y_position),np.array(x_position))
    node_pos_features = np.concatenate((r.reshape(-1,1),t.reshape(-1,1)),axis=1)
    final_node_features = np.array(prob_value).reshape(len(prob_value),2)
    final_node_features = (1-0.1)*final_node_features + 0.1*np.random.uniform(0,1,size=(len(prob_value),2))
    #Node_feature_input = np.concatenate((Node_feature_input,node_pos_features),axis=1)
    
    final = []
    num_nodes = len(final_node_features)
    for idx in range(num_nodes):
        final.append(np.random.standard_normal(size=(1,2)))
        
    Node_feature_final = np.array([final]).reshape(num_nodes,2)
    
    Final_pred = torch.from_numpy(Node_feature_final).float()
    Final_input = torch.from_numpy(final_node_features).float()
    Pos_node = torch.from_numpy(pos_features).float()
    order = torch.ones(Edge_index.shape[1]).view(-1,1)
    
    data = Data(x=Final_input, edge_index=Edge_index,y=Final_pred,pos=Pos_node,edge_attr=order)
    return data





def get_mapping_strip(final_result):
    transform = np.argmax(final_result,axis=1).reshape(20,20).numpy()
    fig, ax = plt.subplots(figsize=(10,10))
    x = np.linspace(0,19,20)
    y = np.linspace(0,19,20)
    x1,y1 = np.meshgrid(x,y)
    ax.scatter(x1,y1,c=transform,s=800)
    plt.axis("off")
    plt.xlim([-1,4])
    plt.ylim([-1,4])
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
    plt.savefig(str(cwd) + "/data/stripes_evaluation.png")

    
    
def get_mapping_big_chess(final_result):
    transform = np.argmax(final_result,axis=1).reshape(16,16).numpy()
    fig, ax = plt.subplots(figsize=(10,10))
    x = np.linspace(0,15,16)
    y = np.linspace(0,15,16)
    x1,y1 = np.meshgrid(x,y)
    ax.scatter(x1,y1,c=transform,s=800)
    plt.axis("off")
    plt.xlim([-1,4])
    plt.ylim([-1,4])
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
    plt.savefig(str(cwd) + "/data/bigchess_evaluation.png")
    
    

    
def get_mapping_small_chess(final_result):
    
    transform = np.argmax(final_result,axis=1).reshape(4,4).numpy()
    fig, ax = plt.subplots(figsize=(10,10))
    x = np.linspace(0,3,4)
    y = np.linspace(0,3,4)
    x1,y1 = np.meshgrid(x,y)
    ax.scatter(x1,y1,c=transform,s=800)
    plt.axis("off")
    plt.xlim([-1,4])
    plt.ylim([-1,4])
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
    plt.savefig(str(cwd) + "/data/smallchess_evaluation.png")
    
    
def alternate_strip_data_grid_eval(): 
    node_index = []
    prob_value= []
    node_pos = []
    prob_value = []
    x_position = []
    y_position = []
    for idx_node in range(20*20):
        x_pos = int(idx_node/20)
        y_pos = int(idx_node%20) 
        layer_val = int(idx_node/40)
        if layer_val%2== 0:
            x_position.append(x_pos)
            y_position.append(y_pos)
            prob_value.append([0,1])
            
        else:
            prob_value.append([1,0])
            x_position.append(x_pos)
            y_position.append(y_pos)
            
       
    pos_features = np.concatenate((np.array(x_position).reshape(-1,1),np.array(y_position).reshape(-1,1)),axis=1)
    pos_features = (pos_features/20)*2 - np.ones((pos_features.shape[0],pos_features.shape[1]))
    Edge_index = knn_graph(torch.Tensor(pos_features), k=4, loop=False)
    
    
    r = np.sqrt(np.array(x_position)**2+np.array(y_position)**2)
    t = np.arctan2(np.array(y_position),np.array(x_position))
    node_pos_features = np.concatenate((r.reshape(-1,1),t.reshape(-1,1)),axis=1)
    final_node_features = np.array(prob_value).reshape(len(prob_value),2)
    final_node_features = (1-0.1)*final_node_features + 0.1*np.random.uniform(0,1,size=(len(prob_value),2))
    #Node_feature_input = np.concatenate((Node_feature_input,node_pos_features),axis=1)
    
    final = []
    num_nodes = len(final_node_features)
    for idx in range(num_nodes):
        final.append(np.random.standard_normal(size=(1,2)))
        
    Node_feature_final = np.array([final]).reshape(num_nodes,2)
    
    Final_pred = torch.from_numpy(Node_feature_final).float()
    Final_input = torch.from_numpy(final_node_features).float()
    Pos_node = torch.from_numpy(pos_features).float()
    order = torch.ones(Edge_index.shape[1]).view(-1,1)
    
    data = Data(x=Final_pred, edge_index=Edge_index,y=Final_input,pos=Pos_node,edge_attr=order)
    return data

def alternate_checker_data_grid(): 
    node_index = []
    prob_value= []
    node_pos = []
    prob_value = []
    x_position = []
    y_position = []
    for idx_node in range(16*16):
        x_pos = int(idx_node/16)
        y_pos = int(idx_node%16) 
        if (int(x_pos/4) + int(y_pos/4))%2 == 0:
            prob_value.append([0,1])
            x_position.append(x_pos)
            y_position.append(y_pos)
            
        else:
            prob_value.append([1,0])
            x_position.append(x_pos)
            y_position.append(y_pos)
       
    pos_features = np.concatenate((np.array(x_position).reshape(-1,1),np.array(y_position).reshape(-1,1)),axis=1)
    pos_features = (pos_features/15)*2 #- np.ones((pos_features.shape[0],pos_features.shape[1]))
    Edge_index = knn_graph(torch.Tensor(pos_features), k=4, loop=False)
    
    
    r = np.sqrt(np.array(x_position)**2+np.array(y_position)**2)
    t = np.arctan2(np.array(y_position),np.array(x_position))
    node_pos_features = np.concatenate((r.reshape(-1,1),t.reshape(-1,1)),axis=1)
    final_node_features = np.array(prob_value).reshape(len(prob_value),2)
    final_node_features = (1-0.1)*final_node_features + 0.1*np.random.uniform(0,1,size=(len(prob_value),2))
    #Node_feature_input = np.concatenate((Node_feature_input,node_pos_features),axis=1)
    
    final = []
    num_nodes = len(final_node_features)
    for idx in range(num_nodes):
        final.append(np.random.standard_normal(size=(1,2)))
        
    Node_feature_final = np.array([final]).reshape(num_nodes,2)
    #Node_feature_final = Node_feature_final/Node_feature_final.sum(axis=1,keepdims=1)
    #Node_feature_final = np.concatenate((Node_feature_final,node_pos_features),axis=1)
    
    Final_pred = torch.from_numpy(Node_feature_final).float()
    Final_input = torch.from_numpy(final_node_features).float()
    Pos_node = torch.from_numpy(pos_features).float()
    order = torch.ones(Edge_index.shape[1]).view(-1,1)
    
    data = Data(x=Final_input, edge_index=Edge_index,y=Final_pred,pos=Pos_node,edge_attr=order)
    return data



def alternate_checker_data_grid_eval(): 
    node_index = []
    prob_value= []
    node_pos = []
    prob_value = []
    x_position = []
    y_position = []
    for idx_node in range(16*16):
        x_pos = int(idx_node/16)
        y_pos = int(idx_node%16) 
        if (int(x_pos/4) + int(y_pos/4))%2 == 0:
            prob_value.append([0,1])
            x_position.append(x_pos)
            y_position.append(y_pos)
            
        else:
            prob_value.append([1,0])
            x_position.append(x_pos)
            y_position.append(y_pos)
       
    pos_features = np.concatenate((np.array(x_position).reshape(-1,1),np.array(y_position).reshape(-1,1)),axis=1)
    pos_features = (pos_features/15)*2 #- np.ones((pos_features.shape[0],pos_features.shape[1]))
    Edge_index = knn_graph(torch.Tensor(pos_features), k=4, loop=False)
    
    
    r = np.sqrt(np.array(x_position)**2+np.array(y_position)**2)
    t = np.arctan2(np.array(y_position),np.array(x_position))
    node_pos_features = np.concatenate((r.reshape(-1,1),t.reshape(-1,1)),axis=1)
    final_node_features = np.array(prob_value).reshape(len(prob_value),2)
    final_node_features = (1-0.1)*final_node_features + 0.1*np.random.uniform(0,1,size=(len(prob_value),2))
    #Node_feature_input = np.concatenate((Node_feature_input,node_pos_features),axis=1)
    
    final = []
    num_nodes = len(final_node_features)
    for idx in range(num_nodes):
        final.append(np.random.standard_normal(size=(1,2)))
        
    Node_feature_final = np.array([final]).reshape(num_nodes,2)
    #Node_feature_final = Node_feature_final/Node_feature_final.sum(axis=1,keepdims=1)
    #Node_feature_final = np.concatenate((Node_feature_final,node_pos_features),axis=1)
    
    Final_pred = torch.from_numpy(Node_feature_final).float()
    Final_input = torch.from_numpy(final_node_features).float()
    Pos_node = torch.from_numpy(pos_features).float()
    order = torch.ones(Edge_index.shape[1]).view(-1,1)
    
    data = Data(x=Final_pred, edge_index=Edge_index,y=Final_input,pos=Pos_node,edge_attr=order)
    return data







def alternate_checker_data_grid_mod(): 
    node_index = []
    prob_value= []
    node_pos = []
    prob_value = []
    x_position = []
    y_position = []
    for idx_node in range(4*4):
        x_pos = int(idx_node/4)
        y_pos = int(idx_node%4) 
        if (int(x_pos) + int(y_pos))%2 == 0:
            x_position.append(x_pos)
            y_position.append(y_pos)
            prob_value.append([1,0])
            
        else:
            prob_value.append([0,1])
            x_position.append(x_pos)
            y_position.append(y_pos)
       
    pos_features = np.concatenate((np.array(x_position).reshape(-1,1),np.array(y_position).reshape(-1,1)),axis=1)
    pos_features = (pos_features/3)*2 - np.ones((pos_features.shape[0],pos_features.shape[1]))
    Edge_index = knn_graph(torch.Tensor(pos_features), k=4, loop=False)
    
    
    r = np.sqrt(np.array(x_position)**2+np.array(y_position)**2)
    t = np.arctan2(np.array(y_position),np.array(x_position))
    node_pos_features = np.concatenate((r.reshape(-1,1),t.reshape(-1,1)),axis=1)
    final_node_features = np.array(prob_value).reshape(len(prob_value),2)
    final_node_features = (1-0.1)*final_node_features + 0.1*np.random.uniform(0,1,size=(len(prob_value),2))
    #Node_feature_input = np.concatenate((Node_feature_input,node_pos_features),axis=1)
    
    final = []
    num_nodes = len(final_node_features)
    for idx in range(num_nodes):
        final.append(np.random.standard_normal(size=(1,2)))
        
    Node_feature_final = np.array([final]).reshape(num_nodes,2)
    
    Final_pred = torch.from_numpy(Node_feature_final).float()
    Final_input = torch.from_numpy(final_node_features).float()
    Pos_node = torch.from_numpy(pos_features).float()
    order = torch.ones(Edge_index.shape[1]).view(-1,1)
    
    data = Data(x=Final_input, edge_index=Edge_index,y=Final_pred,pos=Pos_node,edge_attr=order)
    return data

def alternate_checker_data_grid_mod_eval(): 
    node_index = []
    prob_value= []
    node_pos = []
    prob_value = []
    x_position = []
    y_position = []
    for idx_node in range(4*4):
        x_pos = int(idx_node/4)
        y_pos = int(idx_node%4) 
        if (int(x_pos) + int(y_pos))%2 == 0:
            x_position.append(x_pos)
            y_position.append(y_pos)
            prob_value.append([1,0])
            
        else:
            prob_value.append([0,1])
            x_position.append(x_pos)
            y_position.append(y_pos)
       
    pos_features = np.concatenate((np.array(x_position).reshape(-1,1),np.array(y_position).reshape(-1,1)),axis=1)
    pos_features = (pos_features/3)*2 - np.ones((pos_features.shape[0],pos_features.shape[1]))
    Edge_index = knn_graph(torch.Tensor(pos_features), k=4, loop=False)
    
    
    r = np.sqrt(np.array(x_position)**2+np.array(y_position)**2)
    t = np.arctan2(np.array(y_position),np.array(x_position))
    node_pos_features = np.concatenate((r.reshape(-1,1),t.reshape(-1,1)),axis=1)
    final_node_features = np.array(prob_value).reshape(len(prob_value),2)
    final_node_features = (1-0.1)*final_node_features + 0.1*np.random.uniform(0,1,size=(len(prob_value),2))
    #Node_feature_input = np.concatenate((Node_feature_input,node_pos_features),axis=1)
    
    final = []
    num_nodes = len(final_node_features)
    for idx in num_nodes:
        final.append(np.random.standard_normal(size=(1,2)))
        
    Node_feature_final = np.array([final]).reshape(num_nodes,2)
    
    Final_pred = torch.from_numpy(Node_feature_final).float()
    Final_input = torch.from_numpy(final_node_features).float()
    Pos_node = torch.from_numpy(pos_features).float()
    order = torch.ones(Edge_index.shape[1]).view(-1,1)
    
    
    data = Data(x=Final_pred, edge_index=Edge_index,y=Final_input,pos=Pos_node,edge_attr=order)
    return data

