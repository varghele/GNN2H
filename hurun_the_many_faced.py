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

from torch.nn import Sequential as Seq, Linear as Lin, ReLU, LeakyReLU
from torch_scatter import scatter_mean
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
    bond_dic = {"1": [1, 0, 0, 0, 0], "2": [0, 1, 0, 0, 0], "3": [0, 0, 1, 0, 0], "ar": [0, 0, 0, 1, 0],"am": [0, 0, 0, 0, 1]}

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

            temp_ei.append([int(j)-1 for j in raw_list[i][3][bd][:-1]])  # Appending twice, once reversed for bidirectional edges
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
    node_features=torch.zeros(2,NO_NODE_FEATURES,dtype=torch.float)-1
    edge_index=torch.tensor([[0,1],[1,0]],dtype=torch.long)
    edge_attributes=torch.zeros(2,NO_EDGE_FEATURES,dtype=torch.float)-1
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
NO_GRAPH_FEATURES=64
NO_NODE_FEATURES=7
NO_EDGE_FEATURES=6
NO_MP=2
BATCH_SIZE=4

print("MAKE LATENT GRAPH DATA")
#Read in all the raw mol2 files
RAW_PATH="1_raw"
raw_lo_paths=[RAW_PATH+"\\"+i for i in os.listdir(RAW_PATH)]

#Read in the data in the mol2 files: atom type, position, charge and bonds
data_raw=[]
for mol in raw_lo_paths:
    data_raw.append(load_data(mol))

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


#_________________________Transform data to I/O pairs_________________________
print("TRANSFORM DATA TO I/O PAIRS")
#Load order parameter data
DB_order=pd.read_csv("data/DB_order_parameters_POPC.csv",sep=",",header=0)
# Shuffle
DB_order=DB_order.sample(frac=1)

graph_data=[]
for itm in range(len(DB_order)):
    graph_p1 = latent_graph_data_sc[latent_graph_dict[DB_order["Molecule1"].iloc[itm]]]
    graph_p2 = latent_graph_data_sc[latent_graph_dict[DB_order["Molecule2"].iloc[itm]]]
    graph_pm = latent_graph_data_sc[latent_graph_dict[DB_order["Membrane"].iloc[itm]]]

    delta_edge_1 = graph_p1.x.shape[0]  #stacking adjacency matrixes diagonally, requires adding the number of NODES
    delta_edge_2 = graph_p2.x.shape[0]  #don't mistake this for the edge shape(again)

    graph_x = torch.cat((graph_p1.x, graph_p2.x, graph_pm.x), dim=0)
    graph_ei = torch.cat(
        (graph_p1.edge_index, graph_p2.edge_index + delta_edge_1, graph_pm.edge_index + delta_edge_1 + delta_edge_2),
        dim=1)
    graph_ea = torch.cat((graph_p1.edge_attr, graph_p2.edge_attr, graph_pm.edge_attr), dim=0)

    graph_y = torch.zeros((1, 19))
    graph_y[0][0] = DB_order["molp1"].iloc[itm]
    graph_y[0][1] = DB_order["molp2"].iloc[itm]
    graph_y[0][2] = DB_order["molpm"].iloc[itm]
    graph_y[0][3] = DB_order["Temperature"].iloc[itm]
    for i in range(2,17):
        graph_y[0][i+2] = DB_order[str(i)].iloc[itm]

    data=Data(x=graph_x,edge_index=graph_ei,edge_attr=graph_ea,y=graph_y)

    graph_data.append(data)



#_____________________________________________________________________________

#_________________________Define MessagePassing MetaLayers____________________
class EdgeModel(torch.nn.Module):
    def __init__(self):
        super(EdgeModel, self).__init__()
        in_channels = NO_EDGE_FEATURES+2*NO_NODE_FEATURES+NO_GRAPH_FEATURES
        hidden=64
        self.edge_mlp = Seq(Lin(in_channels, hidden), LeakyReLU(), Lin(hidden, NO_EDGE_FEATURES))

    def forward(self, src, dest, edge_attr, u, batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        #print(f"src: {src.shape}")
        #print(f"dest: {dest.shape}")
        #print(f"edge_attr: {edge_attr.shape}")
        #print(f"u: {u.shape}")

        #print(src.type)
        #print(batch)
        out = torch.cat([src, dest, edge_attr, u[batch]], 1)
        #print(f"out: {out.shape}")
        return self.edge_mlp(out)

class NodeModel(torch.nn.Module):
    def __init__(self):
        super(NodeModel, self).__init__()
        in_channels_1 = NO_EDGE_FEATURES+NO_NODE_FEATURES
        hidden=32
        in_channels_2 = hidden+NO_NODE_FEATURES+NO_GRAPH_FEATURES
        self.node_mlp_1 = Seq(Lin(in_channels_1, hidden), LeakyReLU(), Lin(hidden, hidden))
        self.node_mlp_2 = Seq(Lin(in_channels_2, hidden), LeakyReLU(), Lin(hidden, NO_NODE_FEATURES))

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        print(out.shape)
        #print(x.shape)
        #print(u[batch].shape)
        out = torch.cat([x, out, u[batch]], dim=1)
        #print(out.shape)
        return self.node_mlp_2(out)

class GlobalModel(torch.nn.Module):
    def __init__(self):
        super(GlobalModel, self).__init__()
        in_channels=NO_NODE_FEATURES+NO_GRAPH_FEATURES
        hidden=64
        self.global_mlp = Seq(Lin(in_channels, hidden), LeakyReLU(), Lin(hidden, NO_GRAPH_FEATURES))

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        #print(scatter_mean(x, batch, dim=0).shape)
        out = torch.cat([u, scatter_mean(x, batch, dim=0)], dim=1)
        return self.global_mlp(out)
#_____________________________________________________________________________

#_________________________Construct Graph Network_____________________________
print("CONSTRUCT GRAPH NETWORK")
class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN,self).__init__()
        #self.meta = MetaLayer(EdgeModel().to(device), NodeModel().to(device), GlobalModel().to(device)).to(device)#
        self.meta = MetaLayer(EdgeModel(), NodeModel(), GlobalModel())

        self.mlp1 = torch.nn.Linear(NO_GRAPH_FEATURES*2, NO_GRAPH_FEATURES)
        self.act1 = torch.nn.LeakyReLU()
        self.bn1 = torch.nn.BatchNorm1d(NO_GRAPH_FEATURES)

        self.mlp2 = torch.nn.Linear(NO_GRAPH_FEATURES, NO_GRAPH_FEATURES)
        self.act2 = torch.nn.LeakyReLU()
        self.bn2 = torch.nn.BatchNorm1d(NO_GRAPH_FEATURES)

        self.mlp3 = torch.nn.Linear(NO_GRAPH_FEATURES, NO_GRAPH_FEATURES)
        self.act3 = torch.nn.LeakyReLU()
        self.bn3 = torch.nn.BatchNorm1d(NO_GRAPH_FEATURES)

        self.mlp4 = torch.nn.Linear(NO_GRAPH_FEATURES, 15)

    def forward(self,batch):
        # Here we instantiate the three molecular graphs for the first level GNN
        x, ei, ea, y = batch.x, batch.edge_index, batch.edge_attr, batch.y

        # Create graph level features
        #print(f"x: {x.shape}")
        #print(f"ei: {ei.shape}")
        #print(f"ea: {ea.shape}")
        #print(f"y: {y.shape}")

        u=torch.zeros((BATCH_SIZE,NO_GRAPH_FEATURES)).to(device)
        Tetal = torch.zeros((BATCH_SIZE, NO_GRAPH_FEATURES)).to(device)
        for j in range(NO_GRAPH_FEATURES//BATCH_SIZE):
            for i in range(BATCH_SIZE):
                Tetal[:,j+i]=y[:,i] #only works for BS 4, whatever its only a test


        #u[:, 0] = y[:, 0]
        #u[:, 1] = y[:, 1]
        #u[:, 2] = y[:, 2]
        #u[:, 3] = y[:, 3]

        #u=u.to(device)

        #print(batch.batch)

        for _ in range(NO_MP):
            x, ea, u = self.meta(x=x, edge_index=ei, edge_attr=ea, u=u, batch=batch.batch)

        cat=torch.cat([u,Tetal],dim=1)

        u2=self.mlp1(cat)
        u2=self.act1(u2)
        u2=self.bn1(u2)

        u2 = self.mlp2(u)
        u2 = self.act2(u2)
        u2 = self.bn2(u2)

        u2 = self.mlp3(u)
        u2 = self.act3(u2)
        u2 = self.bn3(u2)

        u2=self.mlp4(u2)


        return u2

#_____________________________________________________________________________

#________________________Define a CallbackOption______________________________
#_____________________________________________________________________________

#________________________Training parameters__________________________________
NUM_EPOCHS = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device="cpu"
device="cuda:0"

def train():
    model.train()

    loss_all=0

    for data in train_loader:
        data=data.to(device)
        optimizer.zero_grad()
        out=model(data)
        target=data.y[:,4:].to(device)
        loss=crit(out,target)
        loss.backward()
        loss_all+=data.num_graphs*loss.item()
        optimizer.step()
    return loss_all/int(len(graph_data)*split)

model=GNN().to(device)
optimizer=torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
crit=torch.nn.MSELoss()

split=0.8
print(np.shape(graph_data))
print(int(len(graph_data)*split))


train_loader=DataLoader(graph_data[:int(len(graph_data)*split)],batch_size=BATCH_SIZE,shuffle=True)
valid_loader=DataLoader(graph_data[int(len(graph_data)*split):],batch_size=BATCH_SIZE,shuffle=True)
def evaluate(loader):
    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():
        for data in loader:

            data = data.to(device)
            pred = model(data).detach().cpu().numpy()

            targ = data.y[:,4:].detach().cpu().numpy()
            predictions.append(pred)
            targets.append(targ)
    return np.sum((pred-targ)**2)

def plots(loader,name):
    if not os.path.exists("plots"):
        os.mkdir("plots")
    if not os.path.exists("plots\\"+name):
        os.mkdir("plots\\"+name)

    batch=0
    for data in loader:
        data=data.to(device)
        pred=model(data).detach().cpu().numpy()
        targ = data.y[:,4:].detach().cpu().numpy()

        fig,axs=plt.subplots(2,2,sharey=True)

        for i in range(BATCH_SIZE):
            axs[i//2][i%2].plot(list(range(2,17)),targ[i],label="True")
            axs[i//2][i%2].plot(list(range(2, 17)), pred[i], label="Fake")
        for axk in axs:
            for ax in axk:
                ax.legend(loc=3)
                ax.set_ylim([0,0.3])
        fig.tight_layout()
        fig.savefig("plots/"+name+"/"+str(batch))
        fig.clf()
        plt.close()
        batch+=1

loss_list=[]
train_accs_list=[]
val_accs_list=[]

for epoch in range(1,NUM_EPOCHS+1):
    loss=train()
    train_acc = evaluate(train_loader)
    val_acc = evaluate(valid_loader)
    loss_list.append(loss)
    train_accs_list.append(train_acc)
    val_accs_list.append(val_acc)
    #test_acc = evaluate(test_loader)
    #print('Epoch: {:03d}, Loss: {:.5f}, Train Auc: {:.5f}, Val Auc: {:.5f}, Test Auc: {:.5f}'.
    #      format(epoch, loss, train_acc, val_acc, test_acc))
    print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Val Acc: {:.5f}'.format(epoch, loss, train_acc, val_acc))

plots(train_loader,"train")
plots(valid_loader,"val")

perf_data=pd.DataFrame(np.transpose([list(range(NUM_EPOCHS)),loss_list,train_accs_list,val_accs_list]),columns=["Epoch","Loss","Train_Acc","Val_Acc"])

palette = sns.color_palette("rocket_r")

fig,axs=plt.subplots(1,2,figsize=(8,4),)
sns.lineplot(data=perf_data,x="Epoch",y="Loss",ax=axs[0],label="Loss",palette=palette)
sns.lineplot(data=perf_data,x="Epoch",y="Train_Acc",ax=axs[1],label="Train_Acc",palette=palette)
sns.lineplot(data=perf_data,x="Epoch",y="Val_Acc",ax=axs[1],label="Val_Acc",palette=palette)
axs[0].set_yscale("log")
axs[1].set_yscale("log")
plt.legend()
fig.tight_layout()
fig.savefig("hurun_out.png")
#model = GraphNet().to(device)
#data = dataset[0].to(device)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
#_____________________________________________________________________________

#___________________________Training__________________________________________
#_____________________________________________________________________________
















