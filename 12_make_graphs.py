#_______________________ Explanation #________________________________________
# Program to make the same KOS for every .mol2 and 
# construct latent graphs from .mol2 files and Order data
# Dependencies
# MDAnalysis
# numpy, pandas, mendeleev
# Torch dependencies
# Pytorch 1.7.0+cu101
# Pytorch geometric
#_____________________________________________________________________________


#_____________________________________________________________________________
#_____________________________________________________________________________


#_______________________ Importing Libraries #________________________________
print("IMPORTING LIBRARIES")
import os
import json

import pandas as pd
import numpy as np
import mendeleev as me
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.data import DataLoader
from itertools import combinations
 
#_____________________________________________________________________________


#____________________________ Quaternion functions #__________________________
def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return w, x, y, z
def q_conjugate(q):
    w, x, y, z = q
    return (w, -x, -y, -z)
def qv_mult(q1, v1):
    q2 = (0.0,) + v1
    return q_mult(q_mult(q1, q2), q_conjugate(q1))[1:]
#_____________________________________________________________________________


#_________________________ Useful functions for Data processing #______________
def helper_for_KOS(in_pth):
    tripos_ATOM=False
    mol2_atom_lines=[]
    with open(in_pth) as fh:
        for line in fh:
            if "ATOM" in line:
                tripos_ATOM = True
            if "BOND" in line:
                tripos_ATOM = False
                tripos_BONDS = True
            if tripos_ATOM and not "ATOM" in line:
                mol2_atom_lines.append(line.split())
    return mol2_atom_lines

def helper_rotate_and_transpose_to_KOS(mol2_atom_lines):
    coordinates=np.array(mol2_atom_lines)[:,2:5].astype(float)
    moltypes=[i.split(".")[0] for i in np.array(mol2_atom_lines)[:,5]]

    
    # Calculate Z-Vector, LONGEST vector
    zvec,zi,zy=[],[],[]
    for i in np.array(coordinates):
        for j in np.array(coordinates):
            vec=j-i
            if np.linalg.norm(vec)>np.linalg.norm(zvec):
                zvec,zi,zj=vec,i,j
    ## SCALE the new vectors
    zvec=np.array(zvec)/np.linalg.norm(zvec)
    
    # Calculate Y-Vector, SECOND-LONGEST vector, perpendicular to Z
    p_yvec=[]
    for j in np.array(coordinates):
        #Get vector to molecule
        mvec=j-zi
        #Get projection vector
        pvec=(np.dot(mvec,zvec)/(np.linalg.norm(zvec)))*zvec/np.linalg.norm(zvec)
        #Get y vector, positive
        vec=mvec-pvec
        if np.linalg.norm(vec)>np.linalg.norm(p_yvec):
            p_yvec=vec
    ## SCALE the new vectors
    p_yvec=np.array(p_yvec)/np.linalg.norm(p_yvec)
    
    # Calculate X-Vector, (Perpendicular Z,Y)
    p_xvec=[]
    for j in np.array(coordinates):
        #Get vector to molecule
        mvec=j-zi
        #Get x unit vector
        xuni=np.cross(zvec,p_yvec)/np.linalg.norm(np.cross(zvec,p_yvec))
        #Get x_vector
        vec=np.dot(mvec,xuni)*xuni
        if np.linalg.norm(vec)>np.linalg.norm(p_xvec):
            p_xvec=vec
    ## SCALE the new vectors
    p_xvec=np.array(p_xvec)/np.linalg.norm(p_xvec)
    
    ## SCALE the new vectors
    zvec=np.array(zvec)/np.linalg.norm(zvec)
    p_yvec=np.array(p_yvec)/np.linalg.norm(p_yvec)
    p_xvec=np.array(p_xvec)/np.linalg.norm(p_xvec)
    
    
    ##ROTATION
    #Calculate KOS matrixes
    KOS_M_1=np.array([[1,0,0],[0,1,0],[0,0,1]])
    KOS_M_2=np.array([list(p_xvec),list(p_yvec),list(zvec)])
    #Calculate rotation matrix between KOS matrixes
    R=np.matmul(np.linalg.inv(KOS_M_1),KOS_M_2)
    #Apply rotation
    rot_coordinates=[]
    for c in coordinates:
        rot_coordinates.append(np.matmul(R,np.array(c)))
    
    """
    ## ROTATION ONE - rotate KOS vectors onto each other
    # Make vector of mol KOS
    p_1=(zvec+p_yvec+p_xvec)/np.linalg.norm(zvec+p_yvec+p_xvec)
    z_1,y_1,x_1=zvec/np.linalg.norm(zvec),p_yvec/np.linalg.norm(p_yvec),p_xvec/np.linalg.norm(p_xvec)
    
    # Make vector of unit KOS
    p_2=np.array([1,1,1])/np.linalg.norm([1,1,1])
    
    # Get rotation quaternion between the two KOS
    #angle = np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))
    angle = -np.arctan2(np.linalg.norm(np.cross(p_1, p_2)), np.dot(p_1, p_2))
    #cos_theta=np.dot(p_1,p_2)/(np.linalg.norm(p_1)*np.linalg.norm(p_2))
    
    
    
    v=np.cross(p_1,p_2)
    v=v/np.linalg.norm(v)
    rot_quat=(np.cos(angle/2),v[0]*np.sin(angle/2),v[1]*np.sin(angle/2),v[2]*np.sin(angle/2))
    
    # Apply the first rotation
    z_1_r,y_1_r,x_1_r=qv_mult(rot_quat, tuple(z_1)),qv_mult(rot_quat, tuple(y_1)),qv_mult(rot_quat, tuple(x_1))
    
    first_rot_coordinates=[]
    for c in coordinates:
        first_rot_coordinates.append(qv_mult(rot_quat, tuple(c)))
        
    ## ROTATION TWO - rotate the KOS around its own axes to match vectors
    beta_1=np.arccos(np.dot(np.array([0,0,1]),p_2)/(np.linalg.norm(np.array([0,0,1]))*np.linalg.norm(p_2)))
    beta_2=np.arccos(np.dot(z_1_r,p_2)/(np.linalg.norm(z_1_r)*np.linalg.norm(p_2)))
    
    k_1=np.sin(beta_1)*np.array([0,0,1])
    k_2=np.sin(beta_2)*np.array(z_1_r)
    
    #=np.arccos(np.dot(k_1,k_2)/(np.linalg.norm(k_1)*np.linalg.norm(k_2)))
    angle = -np.arctan2(np.linalg.norm(np.cross(k_1, k_2)), np.dot(k_1, k_2))
    
    
    # Get rotation quaternion No 2:
    rot_quat_2=(np.cos(angle/2),p_2[0]*np.sin(angle/2),p_2[1]*np.sin(angle/2),p_2[2]*np.sin(angle/2))
    
    # Apply secondary rotation
    second_rot_coordinates=[]
    for c in first_rot_coordinates:
        second_rot_coordinates.append(qv_mult(rot_quat_2,tuple(c)))"""
        
    # Calculate CENTER OF MASS and TRANSLATE
    molweights=[me.element(i).atomic_weight for i in moltypes]
    CoM=np.dot(np.array(molweights),np.array(rot_coordinates))/np.sum(molweights)
    translate_coordinates=rot_coordinates-CoM
        
    ## WRITE TO MOL2 ATOM LINES
    new_mol2_atom_lines=[i for i in mol2_atom_lines]
    test=[]
    for lin in range(len(new_mol2_atom_lines)):
        for c in range(len(new_mol2_atom_lines)):
            new_mol2_atom_lines[lin][2]=translate_coordinates[lin][0]
            new_mol2_atom_lines[lin][3]=translate_coordinates[lin][1]
            new_mol2_atom_lines[lin][4]=translate_coordinates[lin][2]
    
    return new_mol2_atom_lines

def convert_to_same_KOS(in_pth,out_pth):
    mol_list=os.listdir(in_pth)
    n=0
    for mol in mol_list:
        print("Converting {} of {}".format(n,len(mol_list)))
        mol2_atom_lines=helper_for_KOS(in_pth+"/"+mol)
        new_mol2_atom_lines=helper_rotate_and_transpose_to_KOS(mol2_atom_lines)
        
        new_mol=open(out_pth+"/"+mol,"w")
        tripos_ATOM=False
        with open(in_pth+"/"+mol) as fh:
            for line in fh:
                if "ATOM" in line:
                    tripos_ATOM=True
                    new_mol.write(line)
                    l=0
                if "BOND" in line:
                    tripos_ATOM=False
                if tripos_ATOM==False:
                    new_mol.write(line)
                if tripos_ATOM and not "ATOM" in line:
                    # Transposed and rotated data gets written here
                    new_mol.write("{:>7s}".format(new_mol2_atom_lines[l][0])) # Atom Number
                    new_mol.write("{:1s}".format(" "))  # Space
                    new_mol.write("{:8s}".format(new_mol2_atom_lines[l][1])) # Atom Name
                    new_mol.write("{:10.4f}".format(new_mol2_atom_lines[l][2])) # X
                    new_mol.write("{:10.4f}".format(new_mol2_atom_lines[l][3])) # Y
                    new_mol.write("{:10.4f}".format(new_mol2_atom_lines[l][4])) # Z
                    new_mol.write("{:1s}".format(" "))  # Space
                    new_mol.write("{:5s}".format(new_mol2_atom_lines[l][5])) # Atom Type
                    new_mol.write("{:>4s}".format(new_mol2_atom_lines[l][6])) # Chain
                    new_mol.write("{:1s}".format(" "))  # Space
                    new_mol.write("{:1s}".format(" "))  # Space
                    new_mol.write("{:4s}".format(new_mol2_atom_lines[l][7]))  # Residue
                    new_mol.write("{:>14s}".format(new_mol2_atom_lines[l][8])) # Charge
                    new_mol.write("\n")
                    l+=1
        n+=1
                    
        new_mol.close()

def load_data(pth):
    #load_data converts data from mol2 files. Reads out type, position, charge, bonds
    typ=[]
    position=[]
    charge=[]
    bonds=[]

    tripos_ATOM=False
    tripos_BONDS=False

    with open(pth) as fh:
        for line in fh:
            if "ATOM" in line:
                tripos_ATOM = True
            if "BOND" in line:
                tripos_ATOM = False
                tripos_BONDS = True
            if tripos_ATOM and not "ATOM" in line:
                # This is necessary, as atom names can be weird.
                # The type has a format like C.3, where we take the first part ["C","3"]
                typ.append(line.split()[5].split(".")[0])
                position.append([float(i) for i in line.split()[2:5]])
                charge.append(float(line.split()[-1]))
            if tripos_BONDS and not "BOND" in line:
                bonds.append(line.split()[1:])
    return [typ,position,charge,bonds]


def convert_to_graph_data(raw_list):
    #convert_to_graph_data converts the raw data of the list into graph data
    #-Node - Features:
    ##[see below]
    #-Edge - Features:
    ##[distance, bond - type(one - hot)]
    NO_NODE_FEATURES=0
    NO_EDGE_FEATURES=0
    graph_data = []
    bond_dic = {"1": [1, 0, 0, 0, 0, 0], "2": [0, 1, 0, 0, 0, 0], "3": [0, 0, 1, 0, 0, 0], 
                "ar": [0, 0, 0, 1, 0, 0], "am": [0, 0, 0, 0, 1, 0], "n": [0, 0, 0, 0, 0, 1]}

    for i in range(len(raw_list)):
        if i%1==0:
            print(str(i)+"/"+str(len(raw_list)))
        node_features = []
        edge_index = []
        edge_attributes = []

        # Construct node features
        for at in range(len(raw_list[i][0])):
            temp_nf = []  # temporary node features

            temp_nf.append(float(raw_list[i][2][at]))  # charge

            temp_atom_id=raw_list[i][0][at] # Get the atom name/id like "C"
            temp_me_atom=me.element((temp_atom_id)) # Get the mendeleev object that holds all features

            temp_nf.append(temp_me_atom.atomic_number) # atomic number
            temp_nf.append(temp_me_atom.atomic_radius) # atomic radius
            temp_nf.append(temp_me_atom.atomic_volume) # atomic volume
            temp_nf.append(temp_me_atom.atomic_weight) # atomic weight

            temp_nf.append(temp_me_atom.covalent_radius) # covalent radius
            temp_nf.append(temp_me_atom.dipole_polarizability) # dipole polarizability
            temp_nf.append(temp_me_atom.electron_affinity) # electron affinity
            temp_nf.append(temp_me_atom.electrons) # number of electrons

            temp_nf.append(temp_me_atom.electrophilicity()) #electrophillicity index
            temp_nf.append(temp_me_atom.en_pauling) # electronegativity acc. to pauling
            temp_nf.append(temp_me_atom.neutrons) # number of neutrons
            temp_nf.append(temp_me_atom.protons) # number of protons

            temp_nf.append(temp_me_atom.vdw_radius) # van der waals radius

            temp_nf.append(raw_list[i][1][at][0])  # x pos
            temp_nf.append(raw_list[i][1][at][1])  # y pos
            temp_nf.append(raw_list[i][1][at][2])  # z pos

            node_features.append(temp_nf)

        
        ### Construct edge indices and features(attributes)
        #Construct all possible edges
        edge_combs=[list(j) for j in combinations(range(len(raw_list[i][0])),2)]
        #Get bond indice list
        bond_indices=[[int(j)-1 for j in k[:2]] for k in raw_list[i][3]]
        #Iterate over every edge
        for edge in edge_combs:
            #print(edge)
            #print([int(j)-1 for j in raw_list[i][3][0][:-1]])
            #Lists of temporary edge indices and features
            temp_ei = []  # temporary edge index
            temp_ef = []  # temporary edge features
            
            #Is the graph edge a bond between atoms
            if edge in bond_indices:
                #Find out TYPE of bond
                bond_type=bond_dic[raw_list[i][3][bond_indices.index(edge)][-1]]
            #Or isn't there a bond
            if edge not in bond_indices:
                #Type is "none":"n"
                bond_type=bond_dic["n"]
                
            #Appending edge index twice and reversed for bidirectional edge
            temp_ei.append(edge)
            temp_ei.append(edge[::-1])
            
            #Get first bond position
            r1 = np.array(raw_list[i][1][edge[0]])
            #Get second bond position
            r2 = np.array(raw_list[i][1][edge[1]])
            
            #Appending edge features twice because of bidirectional edges
            temp_ef.append([np.linalg.norm(r1 - r2)] + bond_type)
            temp_ef.append([np.linalg.norm(r1 - r2)] + bond_type)
            
            #Appending to graph
            for ei in temp_ei:
                edge_index.append(ei)
            for ef in temp_ef:
                edge_attributes.append(ef)
            
        
        """# Construct edge indices and features(attributes)
        for bd in range(len(raw_list[i][3])):
            temp_ei = []  # temporary edge index
            temp_ef = []  # temporary edge features

            temp_ei.append([int(j)-1 for j in raw_list[i][3][bd][:-1]])
            # Appending twice, once reversed for bidirectional edges
            temp_ei.append([int(j)-1 for j in raw_list[i][3][bd][:-1]][::-1])

            r1 = np.array(raw_list[i][1][int(raw_list[i][3][bd][0])-1])  # First bond position for distance
            r2 = np.array(raw_list[i][1][int(raw_list[i][3][bd][1])-1])  # Second bond position for distance
            bond_type = bond_dic[raw_list[i][3][bd][-1]]

            temp_ef.append([np.linalg.norm(r1 - r2)] + bond_type)  # Appending twice because of bidirectional edges
            temp_ef.append([np.linalg.norm(r1 - r2)] + bond_type)

            for ei in temp_ei:
                edge_index.append(ei)
            for ef in temp_ef:
                edge_attributes.append(ef)"""

        #Tensorize the data
        node_features=torch.tensor(node_features,dtype=torch.float)
        edge_index=torch.tensor(edge_index,dtype=torch.long).t().contiguous()
        edge_attributes=torch.tensor(edge_attributes,dtype=torch.float)

        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attributes)
        graph_data.append(data)
    
    NO_NODE_FEATURES=len(temp_nf)
    NO_EDGE_FEATURES=len(temp_ef[0])

    #Create empty graph with self edge. Used if molecules are not present in the mixture.
    node_features=torch.zeros(2,NO_NODE_FEATURES,dtype=torch.float)#-1
    edge_index=torch.tensor([[0,1],[1,0]],dtype=torch.long)
    edge_attributes=torch.zeros(2,NO_EDGE_FEATURES,dtype=torch.float)#-1
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attributes)
    graph_data.append(data)

    return graph_data

def scale_graph_data(latent_graph_list):
    #Iterate through graph list to get stacked NODE and EDGE features
    node_stack=[]
    edge_stack=[]

    for g in latent_graph_list[:-1]:
        node_stack.append(g.x)          #Append node features
        edge_stack.append(g.edge_attr)  #Append edge features

    node_cat=torch.cat(node_stack,dim=0)
    edge_cat=torch.cat(edge_stack,dim=0)

    #Calculate NODE feature MEAN
    node_mean=node_cat.mean(dim=0)
    #Calculate NODE feature STD
    node_std=node_cat.std(dim=0,unbiased=False)
    #Calculate EDGE feature MEAN
    edge_mean=edge_cat.mean(dim=0)
    #Calculate EDGE feature STD
    edge_std=edge_cat.std(dim=0,unbiased=False)

    #Apply zero-mean, unit variance scaling, append scaled graph to list
    latent_graph_list_sc=[]
    for g in latent_graph_list[:-1]:
        x_sc=g.x-node_mean
        x_sc/=node_std
        ea_sc=g.edge_attr-edge_mean
        ea_sc/=edge_std
        temp_graph=Data(x=x_sc,edge_index=g.edge_index,edge_attr=ea_sc)
        latent_graph_list_sc.append(temp_graph)

    latent_graph_list_sc.append(latent_graph_list[-1])

    return latent_graph_list_sc

def save_graph_data(path,datalist,datadict):
    for key in datadict.keys():
        graph = datalist[datadict[key]]   # get graph
        torch.save(graph.x, path + "/" + key + "_x.pt")
        torch.save(graph.edge_index, path + "/" + key + "_ei.pt")
        torch.save(graph.edge_attr, path + "/" + key + "_ea.pt")

def load_graph_data(path,datadict):
    datalist = []
    for key in datadict.keys():
        x = torch.load(path + "/" + key + "_x.pt")
        ei = torch.load(path + "/" + key + "_ei.pt")
        ea = torch.load(path + "/" + key + "_ea.pt")

        graph = Data(x=x,edge_index=ei,edge_attr=ea)
        datalist.append(graph)
    return datalist

#_____________________________________________________________________________



#_________________ Transpose and Rotate: 1_raw->1_rawKOS #____________________
RAW_PTH="1_raw"
RAWKOS_PTH="1_rawKOS"
if not os.path.exists(RAWKOS_PTH):
    os.mkdir(RAWKOS_PTH)

if len(os.listdir(RAWKOS_PTH))==0:
    print("CONVERTING DATA TO SAME KOS")
    convert_to_same_KOS(RAW_PTH,RAWKOS_PTH)
else:
    print("SKIPPING CONVERSION")
#_____________________________________________________________________________


#_________________ Graph generation: 1_rawKOS->2_latent_graphs #______________
#Read in all the raw mol2 files
PROC_PATH="2_latent_graphs"
if not os.path.exists(PROC_PATH):
    os.mkdir(PROC_PATH)

raw_lo_paths=[RAWKOS_PTH+"\\"+i for i in os.listdir(RAWKOS_PTH)]


#Create dictionary to get the mapping of the latent graph data to molecule name
dict_items=[[os.listdir(RAWKOS_PTH)[i][:-5],i]for i in range(len(os.listdir(RAWKOS_PTH)))]
LATENT_GRAPH_DICT={key:value for (key,value) in dict_items}
#Append empty graph to dict
LATENT_GRAPH_DICT["0"]=-1

#If latent graphs are not constructed, construct them
if len(os.listdir(PROC_PATH))==0:
    print("GRAPH CONSTRUCTION")
    #Read in the data in the mol2 files: atom type, position, charge and bonds
    data_raw=[]
    for mol in raw_lo_paths:
       data_raw.append(load_data(mol))

    #Construct latent graphs
    latent_graph_data=convert_to_graph_data(data_raw)
    #Scale latent graphs
    latent_graph_data_sc=scale_graph_data(latent_graph_data)
    #Save graphs
    save_graph_data(PROC_PATH, latent_graph_data_sc, LATENT_GRAPH_DICT)
    #Save dictionary
    with open('latent_graph_dict.json', 'w') as fp:
        json.dump(LATENT_GRAPH_DICT, fp, sort_keys=True, indent=4)
    
else:
    print("SKIPPING GRAPH CONSTRUCTION")
    #Load data
    #latent_graph_data_sc=load_graph_data(PROC_PATH,LATENT_GRAPH_DICT)
    #with open('data.json', 'r') as fp:
    #    data = json.load(fp)
#_____________________________________________________________________________








