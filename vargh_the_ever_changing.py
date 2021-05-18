#_______________________ Explanation #________________________________________
# Program to test out if GNNs(graph neural networks) are a viable way to predict order parameters.
# Torch dependencies:
# Pytorch 1.7.1+cu101
# Pytorch geometric
#_____________________________________________________________________________

#_____________________________________________________________________________
#_____________________________________________________________________________


#_______________________ Importing Libraries #________________________________
print("IMPORTING LIBRARIES")
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.data import DataLoader

from torch.nn import Sequential as Seq, Linear as Lin, ReLU, LeakyReLU, BatchNorm1d
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.nn import MetaLayer

##_____IMPORT DICTIONARIES
from dicts.atomic_mass import atomic_mass
from dicts.atomic_number import atomic_number
from dicts.atomic_radius import atomic_radius
from dicts.electronegativity import electronegativity
from dicts.covalent_radius import covalent_radius
from dicts.electron_affinity import electron_affinity
#_____________________________________________________________________________

#_________________________Useful functions for Data processing________________
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

def RZFN(val):
    #RZFN-Return zero if None, name is self explanatory
    #Fix for dictionary values that are None
    if val is None:
        return 0
    else:
        return val


def convert_to_graph_data(raw_list):
    #convert_to_graph_data converts the raw data of the list into graph data
    #-Node - Features:
    ##[charge, atomic_mass, atomic_number, atomic_radius, covalent_radius, electron_affinity, electronegativity]
    #-Edge - Features:
    ##[distance, bond - type(one - hot)]
    graph_data = []
    bond_dic = {"1": [1, 0, 0, 0, 0], "2": [0, 1, 0, 0, 0], "3": [0, 0, 1, 0, 0], "ar": [0, 0, 0, 1, 0],
                "am": [0, 0, 0, 0, 1]}

    for i in range(len(raw_list)):
        node_features = []
        edge_index = []
        edge_attributes = []

        # Construct node features
        for at in range(len(raw_list[i][0])):
            temp_nf = []  # temporary node features

            temp_nf.append(float(raw_list[i][2][at]))  # charge

            temp_nf.append(RZFN(atomic_mass[raw_list[i][0][at]]))  # atomic_mass
            temp_nf.append(RZFN(atomic_number[raw_list[i][0][at]]))  # atomic_number
            temp_nf.append(RZFN(atomic_radius[raw_list[i][0][at]]))  # atomic_radius

            temp_nf.append(RZFN(covalent_radius[raw_list[i][0][at]]))  # covalent_radius
            temp_nf.append(RZFN(electron_affinity[raw_list[i][0][at]]))  # electron_affinity
            temp_nf.append(RZFN(electronegativity[raw_list[i][0][at]]))  # electronegativity

            node_features.append(temp_nf)

        # Construct edge indices and features(attributes)
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
                edge_attributes.append(ef)

        #Tensorize the data
        node_features=torch.tensor(node_features,dtype=torch.float)
        edge_index=torch.tensor(edge_index,dtype=torch.long).t().contiguous()
        edge_attributes=torch.tensor(edge_attributes,dtype=torch.float)

        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attributes)
        graph_data.append(data)

    #Create empty graph with self edge. Used if molecules are not present in the mixture.
    node_features=torch.zeros(2,NO_NODE_FEATURES_ONE,dtype=torch.float)-1
    edge_index=torch.tensor([[0,1],[1,0]],dtype=torch.long)
    edge_attributes=torch.zeros(2,NO_EDGE_FEATURES_ONE,dtype=torch.float)-1
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attributes)
    graph_data.append(data)

    return graph_data

def scale_graph_data(latent_graph_list):
    #Iterate through graph list to get stacked NODE and EDGE features
    node_stack=[]
    edge_stack=[]

    for g in latent_graph_list:
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
    for g in latent_graph_list:
        x_sc=g.x-node_mean
        x_sc/=node_std
        ea_sc=g.edge_attr-edge_mean
        ea_sc/=edge_std
        temp_graph=Data(x=x_sc,edge_index=g.edge_index,edge_attr=ea_sc)
        latent_graph_list_sc.append(temp_graph)

    return latent_graph_list_sc

#_____________________________________________________________________________

#_________________________Make latent graph data______________________________
print("MAKE LATENT GRAPH DATA")
#Read in all the raw mol2 files
RAW_PATH="1_raw"
raw_lo_paths=[RAW_PATH+"\\"+i for i in os.listdir(RAW_PATH)]

#Read in the data in the mol2 files: atom type, position, charge and bonds
data_raw=[]
for mol in raw_lo_paths:
    data_raw.append(load_data(mol))

NO_NODE_FEATURES_ONE=7
NO_EDGE_FEATURES_ONE=6

#Construct latent graphs
latent_graph_data=convert_to_graph_data(data_raw)
#Scale latent graphs
latent_graph_data_sc=scale_graph_data(latent_graph_data)

#Create dictionary to get the mapping of the latent graph data to molecule name
dict_items=[[os.listdir(RAW_PATH)[i][:-5],i]for i in range(len(os.listdir(RAW_PATH)))]
latent_graph_dict={key:value for (key,value) in dict_items}
#Append empty graph to dict
latent_graph_dict["0"]=-1
#_____________________________________________________________________________

#_________________________Define Data Loader Function_________________________
print("CONSTRUCT DATALOADER FUNCTION")

#Load order parameter data
DB_order=pd.read_csv("data/DB_order_parameters_POPC.csv",sep=",",header=0)
# Shuffle
DB_order=DB_order.sample(frac=1)
#Potential multiplicator for faster convergence
MULT=1

def data_loader(database,batch_size=4,shuffle=True):
    #-- This loader creates Databatches from a Database, run on either the training or validation set!

    #Establish empty lists
    graph_list_p1=[]
    graph_list_p2=[]
    graph_list_pm=[]
    target_list=[]
    Temperature_list=[]

    #Iterate through database and construct latent graphs of the molecules, as well as the target tensors
    for itm in range(len(database)):
        graph_p1 = latent_graph_data_sc[latent_graph_dict[database["Molecule1"].iloc[itm]]]
        graph_p2 = latent_graph_data_sc[latent_graph_dict[database["Molecule2"].iloc[itm]]]
        graph_pm = latent_graph_data_sc[latent_graph_dict[database["Membrane"].iloc[itm]]]

        graph_p1.y = torch.tensor([float(database["molp1"].iloc[itm])])
        graph_p2.y = torch.tensor([float(database["molp2"].iloc[itm])])
        graph_pm.y = torch.tensor([float(database["molpm"].iloc[itm])])

        graph_list_p1.append(graph_p1)
        graph_list_p2.append(graph_p2)
        graph_list_pm.append(graph_pm)

        target=torch.tensor([database[str(i)].iloc[itm]*MULT for i in range(2,17)],dtype=torch.float)
        target_list.append(target)

        Temperature_list.append(database["Temperature"].iloc[itm])



    #Generate a shuffled integer list that then produces distinct batches
    idxs=torch.randperm(len(database)).tolist()
    if shuffle==False:
        idxs=sorted(idxs)

    #Generate the batches
    batch_p1=[]
    batch_p2=[]
    batch_pm=[]
    batch_target=[]
    batch_Temperature=[]

    for b in range(len(database))[::batch_size]:
        #Creating empty Batch objects
        B_p1 = Batch()
        B_p2 = Batch()
        B_pm = Batch()

        #Creating empty lists that will be concatenated later (advanced minibatching)
        stack_x_p1,stack_x_p2,stack_x_pm=[],[],[]
        stack_ei_p1,stack_ei_p2,stack_ei_pm=[],[],[]
        stack_ea_p1,stack_ea_p2,stack_ea_pm=[],[],[]
        stack_y_p1,stack_y_p2,stack_y_pm=[],[],[]
        stack_btc_p1,stack_btc_p2,stack_btc_pm=[],[],[]

        stack_target=[]
        stack_Temperature=[]

        #Creating the batch shifts, that shift the edge indices
        b_shift_p1=0
        b_shift_p2=0
        b_shift_pm=0

        for i in range(batch_size):
            if ((b+i)<len(database)):
                stack_x_p1.append(graph_list_p1[idxs[b + i]].x)
                stack_x_p2.append(graph_list_p2[idxs[b + i]].x)
                stack_x_pm.append(graph_list_pm[idxs[b + i]].x)

                stack_ei_p1.append(graph_list_p1[idxs[b + i]].edge_index+b_shift_p1)
                stack_ei_p2.append(graph_list_p2[idxs[b + i]].edge_index+b_shift_p2)
                stack_ei_pm.append(graph_list_pm[idxs[b + i]].edge_index+b_shift_pm)

                #Updating the shifts
                b_shift_p1+=graph_list_p1[idxs[b + i]].x.size()[0]
                b_shift_p2+=graph_list_p2[idxs[b + i]].x.size()[0]
                b_shift_pm+=graph_list_pm[idxs[b + i]].x.size()[0]

                stack_ea_p1.append(graph_list_p1[idxs[b + i]].edge_attr)
                stack_ea_p2.append(graph_list_p2[idxs[b + i]].edge_attr)
                stack_ea_pm.append(graph_list_pm[idxs[b + i]].edge_attr)

                stack_y_p1.append(graph_list_p1[idxs[b + i]].y)
                stack_y_p2.append(graph_list_p2[idxs[b + i]].y)
                stack_y_pm.append(graph_list_pm[idxs[b + i]].y)

                stack_btc_p1.append(
                    torch.full([graph_list_p1[idxs[b + i]].x.size()[0]], fill_value=int(i), dtype=torch.long))
                #FILL VALUE IS PROBABLY JUST i! (OR NOT)
                stack_btc_p2.append(
                    torch.full([graph_list_p2[idxs[b + i]].x.size()[0]], fill_value=int(i), dtype=torch.long))
                stack_btc_pm.append(
                    torch.full([graph_list_pm[idxs[b + i]].x.size()[0]], fill_value=int(i), dtype=torch.long))

                stack_target.append(target_list[idxs[b + i]])
                stack_Temperature.append(Temperature_list[int(b+i)])

        #print(stack_y_p1)
        #print(stack_x_p1)

        B_p1.edge_index=torch.cat(stack_ei_p1,dim=1)
        B_p1.x=torch.cat(stack_x_p1,dim=0)
        B_p1.edge_attr=torch.cat(stack_ea_p1,dim=0)
        B_p1.y=torch.cat(stack_y_p1,dim=0)
        B_p1.batch=torch.cat(stack_btc_p1,dim=0)

        B_p2.edge_index = torch.cat(stack_ei_p2, dim=1)
        B_p2.x = torch.cat(stack_x_p2, dim=0)
        B_p2.edge_attr = torch.cat(stack_ea_p2, dim=0)
        B_p2.y = torch.cat(stack_y_p2, dim=0)
        B_p2.batch = torch.cat(stack_btc_p2, dim=0)

        B_pm.edge_index = torch.cat(stack_ei_pm, dim=1)
        B_pm.x = torch.cat(stack_x_pm, dim=0)
        B_pm.edge_attr = torch.cat(stack_ea_pm, dim=0)
        B_pm.y = torch.cat(stack_y_pm, dim=0)
        B_pm.batch = torch.cat(stack_btc_pm, dim=0)

        B_target = torch.stack(stack_target,dim=0)

        B_Temperature=torch.Tensor(stack_Temperature)

        #Appending batches
        batch_p1.append(B_p1)
        batch_p2.append(B_p2)
        batch_pm.append(B_pm)
        batch_target.append(B_target)
        batch_Temperature.append(B_Temperature)

    #Return
    return [batch_p1,batch_p2,batch_pm,batch_target,batch_Temperature]
#_____________________________________________________________________________

#_________________________Define MessagePassing MetaLayers____________________
def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class EdgeModel_ONE(torch.nn.Module):
    def __init__(self):
        super(EdgeModel_ONE, self).__init__()
        hidden = HIDDEN_EDGE_ONE
        in_channels = HID_EDGE_ENC+2*HID_NODE_ENC
        self.edge_mlp = Seq(Lin(in_channels, hidden), LeakyReLU(), BatchNorm1d(hidden),
                            Lin(hidden, HID_EDGE_ENC)).apply(init_weights)

    def forward(self, src, dest, edge_attr, u, batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr], 1)
        return self.edge_mlp(out)
class EdgeModel_TWO(torch.nn.Module):
    def __init__(self):
        super(EdgeModel_TWO, self).__init__()
        hidden = HIDDEN_EDGE_TWO
        in_channels = NO_EDGE_FEATURES_TWO+2*NO_NODE_FEATURES_TWO
        self.edge_mlp = Seq(Lin(in_channels, hidden), LeakyReLU(), BatchNorm1d(hidden),
                            Lin(hidden, NO_EDGE_FEATURES_TWO)).apply(init_weights)

    def forward(self, src, dest, edge_attr, u, batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr], 1)
        return self.edge_mlp(out)

class NodeModel_ONE(torch.nn.Module):
    def __init__(self):
        super(NodeModel_ONE, self).__init__()
        hidden=HIDDEN_NODE_ONE
        in_channels_1 = HID_EDGE_ENC+HID_NODE_ENC
        in_channels_2 = hidden+HID_NODE_ENC
        self.node_mlp_1 = Seq(Lin(in_channels_1, hidden), LeakyReLU(), BatchNorm1d(hidden),
                              Lin(hidden, hidden)).apply(init_weights)
        self.node_mlp_2 = Seq(Lin(in_channels_2, hidden), LeakyReLU(), BatchNorm1d(hidden),
                              Lin(hidden, HID_NODE_ENC)).apply(init_weights)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_add(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)
class NodeModel_TWO(torch.nn.Module):
    def __init__(self):
        super(NodeModel_TWO, self).__init__()
        hidden=HIDDEN_NODE_TWO
        in_channels_1 = NO_EDGE_FEATURES_TWO+NO_NODE_FEATURES_TWO
        in_channels_2 = hidden+NO_NODE_FEATURES_TWO
        self.node_mlp_1 = Seq(Lin(in_channels_1, hidden), LeakyReLU(), BatchNorm1d(hidden),
                              Lin(hidden, hidden)).apply(init_weights)
        self.node_mlp_2 = Seq(Lin(in_channels_2, hidden), LeakyReLU(), BatchNorm1d(hidden),
                              Lin(hidden, NO_NODE_FEATURES_TWO)).apply(init_weights)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_add(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)

class GlobalModel_ONE(torch.nn.Module):
    def __init__(self):
        super(GlobalModel_ONE, self).__init__()
        hidden = HIDDEN_GRAPH_ONE
        in_channels_1=HID_NODE_ENC+HID_EDGE_ENC
        in_channels_2=hidden+HID_NODE_ENC

        self.global_mlp_1 = Seq(Lin(in_channels_1, hidden), LeakyReLU(), BatchNorm1d(hidden),
                                Lin(hidden, hidden)).apply(init_weights)
        self.global_mlp_2 = Seq(Lin(in_channels_2, hidden), LeakyReLU(), BatchNorm1d(hidden),
                                Lin(hidden, hidden), LeakyReLU(), BatchNorm1d(hidden),
                                #Lin(hidden, hidden), LeakyReLU(), BatchNorm1d(hidden),
                                Lin(hidden, NO_GRAPH_FEATURES_ONE)).apply(init_weights)

        in_mlp = NO_GRAPH_FEATURES_ONE+HID_NODE_ENC
        out_mlp = NO_GRAPH_FEATURES_ONE

        self.global_mlp = Seq(Lin(in_mlp, hidden), LeakyReLU(), BatchNorm1d(hidden),
                              Lin(hidden, out_mlp)).apply(init_weights)

    #def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        #print(f"X: {x.shape}")
        #print(f"U: {u.shape}")
        #print(f"B: {batch.shape}")
        #print(f"SM: {scatter_mean(x, batch, dim=0, dim_size=u.size(0)).shape}")
        #print("\n")
        #out = torch.cat([u, scatter_mean(x, batch, dim=0)], dim=1)
        #return self.global_mlp(out)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row,col=edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.global_mlp_1(out)
        out = scatter_add(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([scatter_add(x, batch, dim=0),scatter_add(out,batch,dim=0)], dim=1)
        return self.global_mlp_2(out)

class GlobalModel_TWO(torch.nn.Module):
    def __init__(self):
        super(GlobalModel_TWO, self).__init__()
        hidden = HIDDEN_GRAPH_TWO
        in_channels_1=NO_NODE_FEATURES_TWO+NO_EDGE_FEATURES_TWO
        in_channels_2=hidden+NO_NODE_FEATURES_TWO

        self.global_mlp_1 = Seq(Lin(in_channels_1, hidden), LeakyReLU(), BatchNorm1d(hidden),
                                Lin(hidden, hidden)).apply(init_weights)
        self.global_mlp_2 = Seq(Lin(in_channels_2, hidden), LeakyReLU(), BatchNorm1d(hidden),
                                #Lin(hidden, hidden), LeakyReLU(), BatchNorm1d(hidden),
                                #Lin(hidden, hidden), LeakyReLU(), BatchNorm1d(hidden),
                                Lin(hidden, NO_GRAPH_FEATURES_TWO)).apply(init_weights)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row,col=edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.global_mlp_1(out)
        out = scatter_add(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([scatter_add(x, batch, dim=0),scatter_add(out,batch,dim=0)], dim=1)
        return self.global_mlp_2(out)

#_____________________________________________________________________________

#_________________________Construct Graph Network_____________________________
print("CONSTRUCT GRAPH NETWORK")
class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN,self).__init__()
        self.meta1 = MetaLayer(EdgeModel_ONE(), NodeModel_ONE(), GlobalModel_ONE())
        self.meta2 = MetaLayer(EdgeModel_ONE(), NodeModel_ONE(), GlobalModel_ONE())
        self.meta3 = MetaLayer(EdgeModel_ONE(), NodeModel_ONE(), GlobalModel_ONE())

        self.enc_edge_memb=Seq(Lin(NO_EDGE_FEATURES_ONE,HID_EDGE_ENC), LeakyReLU(), BatchNorm1d(HID_EDGE_ENC),
                               #Lin(HID_EDGE_ENC, HID_EDGE_ENC), LeakyReLU(), BatchNorm1d(HID_EDGE_ENC),
                               #Lin(HID_EDGE_ENC, HID_EDGE_ENC), LeakyReLU(), BatchNorm1d(HID_EDGE_ENC),
                               Lin(HID_EDGE_ENC, HID_EDGE_ENC)).apply(init_weights)
        self.enc_edge_mol=Seq(Lin(NO_EDGE_FEATURES_ONE,HID_EDGE_ENC), LeakyReLU(), BatchNorm1d(HID_EDGE_ENC),
                               #Lin(HID_EDGE_ENC, HID_EDGE_ENC), LeakyReLU(), BatchNorm1d(HID_EDGE_ENC),
                               #Lin(HID_EDGE_ENC, HID_EDGE_ENC), LeakyReLU(), BatchNorm1d(HID_EDGE_ENC),
                               Lin(HID_EDGE_ENC, HID_EDGE_ENC)).apply(init_weights)

        self.enc_node_memb = Seq(Lin(NO_NODE_FEATURES_ONE, HID_NODE_ENC), LeakyReLU(),BatchNorm1d(HID_NODE_ENC),
                                 #Lin(HID_NODE_ENC, HID_NODE_ENC), LeakyReLU(), BatchNorm1d(HID_NODE_ENC),
                                 #Lin(HID_NODE_ENC, HID_NODE_ENC), LeakyReLU(), BatchNorm1d(HID_NODE_ENC),
                                 Lin(HID_NODE_ENC, HID_NODE_ENC)).apply(init_weights)
        self.enc_node_mol = Seq(Lin(NO_NODE_FEATURES_ONE, HID_NODE_ENC), LeakyReLU(), BatchNorm1d(HID_NODE_ENC),
                                #Lin(HID_NODE_ENC, HID_NODE_ENC), LeakyReLU(), BatchNorm1d(HID_NODE_ENC),
                                #Lin(HID_NODE_ENC, HID_NODE_ENC), LeakyReLU(), BatchNorm1d(HID_NODE_ENC),
                                Lin(HID_NODE_ENC, HID_NODE_ENC)).apply(init_weights)

        self.a_linear = Seq(Lin((NO_GRAPH_FEATURES_ONE+1),NO_GRAPH_FEATURES_ONE), LeakyReLU(),
                            BatchNorm1d(NO_GRAPH_FEATURES_ONE),
                            #Lin(NO_GRAPH_FEATURES_ONE,NO_GRAPH_FEATURES_ONE), LeakyReLU(),
                            #BatchNorm1d(NO_GRAPH_FEATURES_ONE),
                            Lin(NO_GRAPH_FEATURES_ONE,(NO_GRAPH_FEATURES_ONE//2))).apply(init_weights)
        self.b_linear = Seq(Lin((NO_GRAPH_FEATURES_ONE+1),NO_GRAPH_FEATURES_ONE), LeakyReLU(),
                            BatchNorm1d(NO_GRAPH_FEATURES_ONE),
                            #Lin(NO_GRAPH_FEATURES_ONE,NO_GRAPH_FEATURES_ONE), LeakyReLU(),
                            #BatchNorm1d(NO_GRAPH_FEATURES_ONE),
                            Lin(NO_GRAPH_FEATURES_ONE,(NO_GRAPH_FEATURES_ONE//2))).apply(init_weights)
        self.c_linear = Seq(Lin((NO_GRAPH_FEATURES_ONE+1),NO_GRAPH_FEATURES_ONE), LeakyReLU(),
                            BatchNorm1d(NO_GRAPH_FEATURES_ONE),
                            #Lin(NO_GRAPH_FEATURES_ONE,NO_GRAPH_FEATURES_ONE), LeakyReLU(),
                            #BatchNorm1d(NO_GRAPH_FEATURES_ONE),
                            Lin(NO_GRAPH_FEATURES_ONE,(NO_GRAPH_FEATURES_ONE//2))).apply(init_weights)

        self.d_linear = Seq(Lin(3*(NO_GRAPH_FEATURES_ONE//2),NO_GRAPH_FEATURES_ONE), LeakyReLU(),
                            BatchNorm1d(NO_GRAPH_FEATURES_ONE),
                            Lin(NO_GRAPH_FEATURES_ONE, NO_GRAPH_FEATURES_ONE), LeakyReLU(),
                            BatchNorm1d(NO_GRAPH_FEATURES_ONE),
                            Lin(NO_GRAPH_FEATURES_ONE, NO_GRAPH_FEATURES_ONE), LeakyReLU(),
                            BatchNorm1d(NO_GRAPH_FEATURES_ONE),
                            #Lin(NO_GRAPH_FEATURES_ONE, NO_GRAPH_FEATURES_ONE), LeakyReLU(),
                            #BatchNorm1d(NO_GRAPH_FEATURES_ONE),
                            Lin(NO_GRAPH_FEATURES_ONE,15)).apply(init_weights)

    def forward(self,plist):
        # Here we instantiate the three molecular graphs for the first level GNN
        p1,p2,pm,Temperature=plist
        x_p1,ei_p1,ea_p1,u_p1,btc_p1=p1.x,p1.edge_index,p1.edge_attr,p1.y,p1.batch
        x_p2,ei_p2,ea_p2,u_p2,btc_p2=p2.x,p2.edge_index,p2.edge_attr,p2.y,p2.batch
        x_pm,ei_pm,ea_pm,u_pm,btc_pm=pm.x,pm.edge_index,pm.edge_attr,pm.y,pm.batch

        # Embed the node and edge features
        enc_x_p1 = self.enc_node_mol(x_p1)
        enc_x_p2 = self.enc_node_mol(x_p2)
        enc_x_pm = self.enc_node_memb(x_pm)

        enc_ea_p1 = self.enc_edge_mol(ea_p1)
        enc_ea_p2 = self.enc_edge_mol(ea_p2)
        enc_ea_pm = self.enc_edge_mol(ea_pm)

        #Create the empty molecular graphs for feature extraction, graph level one
        u1=torch.full(size=(BATCH_SIZE, NO_GRAPH_FEATURES_ONE), fill_value=0.1, dtype=torch.float).to(device)
        u2=torch.full(size=(BATCH_SIZE, NO_GRAPH_FEATURES_ONE), fill_value=0.1, dtype=torch.float).to(device)
        um=torch.full(size=(BATCH_SIZE, NO_GRAPH_FEATURES_ONE), fill_value=0.1, dtype=torch.float).to(device)

        # Now the first level rounds of message passing are performed, molecular features are constructed
        for _ in range(NO_MP_ONE):
            enc_x_p1, enc_ea_p1, u1 = self.meta1(x=enc_x_p1, edge_index=ei_p1, edge_attr=enc_ea_p1, u=u1, batch=btc_p1)
            enc_x_p2, enc_ea_p2, u2 = self.meta1(x=enc_x_p2, edge_index=ei_p2, edge_attr=enc_ea_p2, u=u2, batch=btc_p2)
            enc_x_pm, enc_ea_pm, um = self.meta1(x=enc_x_pm, edge_index=ei_pm, edge_attr=enc_ea_pm, u=um, batch=btc_pm)

        # --- SECOND LEVEL NETWORK
        # Create the edge features for the second level graphs from the "u"s of the first level
        p1_div_pm = u_p1 / u_pm
        p2_div_pm = u_p2 / u_pm

        # Create the second level network inputs
        p1_div_pm = torch.transpose(torch.unsqueeze(p1_div_pm, 0), 0, 1)
        p2_div_pm = torch.transpose(torch.unsqueeze(p2_div_pm, 0), 0, 1)
        T2 = torch.transpose(torch.unsqueeze(Temperature, 0), 0, 1)

        a = torch.cat([p1_div_pm, u1], dim=1)
        b = torch.cat([p2_div_pm, u2], dim=1)
        c = torch.cat([T2, um], dim=1)

        a = self.a_linear(a)
        b = self.b_linear(b)
        c = self.c_linear(c)

        d = torch.cat([a,b,c], dim=1)
        d = self.d_linear(d)

        return d

#_____________________________________________________________________________

#________________________Define a CallbackOption______________________________
#_____________________________________________________________________________

#________________________Training parameters__________________________________
NO_GRAPH_FEATURES_ONE=128
NO_GRAPH_FEATURES_TWO=64

NO_NODE_FEATURES_TWO=NO_GRAPH_FEATURES_ONE

NO_EDGE_FEATURES_TWO=32

HID_NODE_ENC=64
HID_EDGE_ENC=64

NO_MP_ONE=4
NO_MP_TWO=2

HIDDEN_NODE_ONE,HIDDEN_NODE_TWO=64,64
HIDDEN_EDGE_ONE,HIDDEN_EDGE_TWO=64,64
HIDDEN_GRAPH_ONE,HIDDEN_GRAPH_TWO=64,64


NUM_EPOCHS = 1000
BATCH_SIZE = 30
SPLIT = 0.75
TRAIN_SET = DB_order[:int(SPLIT*len(DB_order))]
VAL_SET = DB_order[int(SPLIT*len(DB_order)):]

"""train_loader=DataLoader(latent_graph_data_sc[:int(len(latent_graph_data_sc)*0.75)],batch_size=BATCH_SIZE,shuffle=False)
for batch in train_loader:
    print(batch.batch)
    print(batch.edge_index)
    print("\n")

batch_p1,batch_p2,batch_pm,batch_target,batch_Temperature=data_loader(TRAIN_SET, batch_size=BATCH_SIZE,
                                                                          shuffle=False)
for btc in range(len(batch_p1)):
    p1=batch_p1[btc]#.to(device)
    print(p1.batch)
    print(p1.edge_index)

exit()"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"DEVICE: {device}")
#device="cpu"

model=GNN().to(device)
optimizer=torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
crit=torch.nn.MSELoss()

def train():
    #model.train()

    loss_all = 0

    batch_p1,batch_p2,batch_pm,batch_target,batch_Temperature=data_loader(TRAIN_SET, batch_size=BATCH_SIZE,
                                                                          shuffle=True)

    for btc in range(len(batch_p1)):
        p1=batch_p1[btc].to(device)
        p2=batch_p2[btc].to(device)
        pm=batch_pm[btc].to(device)
        targ=batch_target[btc].to(device)
        Temperature=batch_Temperature[btc].to(device)

        out=model([p1,p2,pm,Temperature])
        loss=crit(out,targ)
        #testl=np.mean(np.abs(out.detach().cpu().numpy()-targ.detach().cpu().numpy()))
        optimizer.zero_grad()#set_to_none=True

        loss.backward()
        #Gradient clipping (?)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.001)

        loss_all+=loss.item()
        optimizer.step()
    return loss_all

def evaluate(SET):
    model.eval()

    set_loss = []

    with torch.no_grad():
        batch_p1,batch_p2,batch_pm,batch_target,batch_Temperature=data_loader(SET, batch_size=BATCH_SIZE,
                                                                              shuffle=True)


        for btc in range(len(batch_p1)):
            p1 = batch_p1[btc].to(device)
            p2 = batch_p2[btc].to(device)
            pm = batch_pm[btc].to(device)
            targ = batch_target[btc].detach().cpu().numpy()
            Temperature = batch_Temperature[btc].to(device)

            out = model([p1, p2, pm, Temperature]).detach().cpu().numpy()

            set_loss.append(np.mean(np.abs(out-targ)))

    return np.sum(set_loss)

def plots(SET,name):
    if not os.path.exists("plots"):
        os.mkdir("plots")
    if not os.path.exists("plots\\"+name):
        os.mkdir("plots\\"+name)

    with torch.no_grad():
        batch_p1,batch_p2,batch_pm,batch_target,batch_Temperature=data_loader(SET, batch_size=BATCH_SIZE, shuffle=True)

    for btc in range(len(batch_p1)):
        p1 = batch_p1[btc].to(device)
        p2 = batch_p2[btc].to(device)
        pm = batch_pm[btc].to(device)
        targ = batch_target[btc].detach().cpu().numpy()
        Temperature = batch_Temperature[btc].to(device)

        out = model([p1, p2, pm, Temperature]).detach().cpu().numpy()

        for i in range(BATCH_SIZE)[::2]:
            fig, axs = plt.subplots(1, 2, sharey=True)

            axs[i % 2].plot(list(range(2, 17)), targ[i], label="True")
            axs[i % 2].plot(list(range(2, 17)), out[i], label="Fake")

            axs[(i+1) % 2].plot(list(range(2, 17)), targ[i+1], label="True")
            axs[(i+1) % 2].plot(list(range(2, 17)), out[i+1], label="Fake")
            #axs[i//2][i%2].plot(list(range(2, 17)), out[i], label="Fake")
            for ax in axs:
                ax.legend(loc=3)
                ax.set_ylim([0,0.3*MULT])
            fig.tight_layout()
            fig.savefig("plots/"+name+"/"+"vargh_B"+str(btc)+"_"+str(i))
            fig.clf()
            plt.close()


loss_list=[]
train_accs_list=[]
val_accs_list=[]

model.train()
for epoch in range(1,NUM_EPOCHS+1):
    loss=train()
    train_acc = evaluate(TRAIN_SET)
    val_acc = evaluate(VAL_SET)
    loss_list.append(loss)
    train_accs_list.append(train_acc)
    val_accs_list.append(val_acc)
    print('Epoch: {:03d}, Loss: {:.5f}, Train Err: {:.5f}, Val Err: {:.5f}'.format(epoch, loss, train_acc, val_acc))

plots(TRAIN_SET,"train")
plots(VAL_SET,"val")

perf_data=pd.DataFrame(np.transpose([list(range(NUM_EPOCHS)),loss_list,train_accs_list,val_accs_list]),
                       columns=["Epoch","Loss","Train_Err","Val_Err"])

palette = sns.color_palette("rocket_r")

fig,axs=plt.subplots(1,2,figsize=(8,4),)
sns.lineplot(data=perf_data,x="Epoch",y="Loss",ax=axs[0],label="Loss",palette=palette)
sns.lineplot(data=perf_data,x="Epoch",y="Train_Err",ax=axs[1],label="Train_Err",palette=palette)
sns.lineplot(data=perf_data,x="Epoch",y="Val_Err",ax=axs[1],label="Val_Err",palette=palette)
axs[0].set_yscale("log")
axs[1].set_yscale("log")
plt.legend()
fig.tight_layout()
fig.savefig("vargh_out.png")
#model = GraphNet().to(device)
#data = dataset[0].to(device)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
#_____________________________________________________________________________

#___________________________Training__________________________________________
#_____________________________________________________________________________
















