"""
Script to construct graphs from the .mol2 files for later use.
-Save each graph individually, as well as construct a pytorch-geometric dataset.

-Node-Features:
[charge,atomic_mass,atomic_number,atomic_radius,covalent_radius,electron_affinity,electronegativity]

-Edge-Features:
[distance,bond-type(one-hot)]

-Graph-Features:
Zero-array of shape [128,]
"""

##----IMPORTS
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from tqdm import tqdm
import numpy as np
import pandas as pd
import os

##----IMPORT DICTIONARIES
from dicts.atomic_mass import atomic_mass
from dicts.atomic_number import atomic_number
from dicts.atomic_radius import atomic_radius
from dicts.electronegativity import electronegativity
from dicts.covalent_radius import covalent_radius
from dicts.electron_affinity import electron_affinity



##----LOAD IN DATA
RAW_PATH="1_raw"
fl_list=[RAW_PATH+"\\"+i for i in os.listdir(RAW_PATH)]

def load_data(pth):
    typ=[]
    position=[]
    charge=[]
    bonds=[]

    tripos_ATOM=False
    tripos_BONDS=False

    with open(pth) as fh:
        for line in fh:
            if "ATOM" in line:
                tripos_ATOM=True
            if "BOND" in line:
                tripos_ATOM=False
                tripos_BONDS=True
            if tripos_ATOM and not "ATOM" in line:
                typ.append(line.split()[1])
                position.append([float(i) for i in line.split()[2:5]])
                charge.append(float(line.split()[-1]))
            if tripos_BONDS and not "BOND" in line:
                bonds.append(line.split()[1:])
    return [typ,position,charge,bonds]

data_raw=[]
for mol in fl_list:
    data_raw.append(load_data(mol))

print(data_raw[0][0])
print(data_raw[0][1])
print(data_raw[0][2])
print(data_raw[0][3])
##----DEFINE InMemoryDataset CLASS FOR ALL GRAPHS
class IMDataset_ALL(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(IMDataset_ALL, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['2_latent_graphs/2HMols.dataset']
    def download(self):
        pass
    def process(self):
        data_list = []

        bond_dic={"1":[1,0,0,0],"2":[0,1,0,0],"3":[0,0,1,0],"ar":[0,0,0,1]}

        for i in range(data_raw):
            node_features=[]
            edge_index=[]
            edge_attributes=[]

            #Construct node features
            for at in range(len(data_raw[i][0])):
                temp_nf=[]                                               #temporary node features

                temp_nf.append(float(data_raw[i][2][at]))                # charge

                temp_nf.append(atomic_mass[data_raw[i][0][at]])          # atomic_mass
                temp_nf.append(atomic_number[data_raw[i][0][at]])        # atomic_number
                temp_nf.append(atomic_radius[data_raw[i][0][at]])        # atomic_radius

                temp_nf.append(covalent_radius[data_raw[i][0][at]])      # covalent_radius
                temp_nf.append(electron_affinity[data_raw[i][0][at]])    # electron_affinity
                temp_nf.append(electronegativity[data_raw[i][0][at]])    # electronegativity

                node_features.append(temp_nf)

            #Construct edge indice and features(attributes)
            for bd in range(len(data_raw[i][3])):
                temp_ei=[]      #temporary edge index
                temp_ef=[]      #temporary edge features

                temp_ei.append([int(i) for i in data_raw[i][3][bd][1:]])    #Appending twice, once reversed for bidirectional edges
                temp_ei.append([int(i) for i in data_raw[i][3][bd][1:]][::-1])

                r1=np.array(data_raw[i][1][int(data_raw[i][3][bd][1])-1]) #First bond position
                r2=np.array(data_raw[i][1][int(data_raw[i][3][bd][2])-1]) #Second bond position
                bond_type=bond_dic[data_raw[i][3][bd][0]]

                temp_ef.append([np.linalg.norm(r1-r2)]+bond_type)   #Aüüending twice because of bidirectiona edges
                temp_ef.append([np.linalg.norm(r1-r2)]+bond_type)

                edge_index.append(temp_ei)
                edge_attributes.append(temp_ef)


            data = Data(x=node_features, edge_index=edge_index, edge_attributes=edge_attributes)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

IMDataset_ALL()

##----DEFINE InMemoryDataset CLASS FOR SINGLE GRAPH

"""class IMDataset_ALL(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(IMDataset_ALL, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['2_latent_graphs/2HMols.dataset']
    def download(self):
        pass
    def process(self):
        data_list = []

        # process by session_id
        grouped = df.groupby('session_id')
        for session_id, group in tqdm(grouped):
            sess_item_id = LabelEncoder().fit_transform(group.item_id)
            group = group.reset_index(drop=True)
            group['sess_item_id'] = sess_item_id
            node_features = group.loc[group.session_id == session_id, ['sess_item_id', 'item_id']].sort_values(
                'sess_item_id').item_id.drop_duplicates().values

            node_features = torch.LongTensor(node_features).unsqueeze(1)
            target_nodes = group.sess_item_id.values[1:]
            source_nodes = group.sess_item_id.values[:-1]

            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
            x = node_features

            y = torch.FloatTensor([group.label.values[0]])

            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])"""



