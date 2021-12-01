import torch
from torch_geometric.data import Data


def load_graph_data(path,datadict):
    datalist = []
    for key in datadict.keys():
        x = torch.load(path + "/" + key + "_x.pt")
        ei = torch.load(path + "/" + key + "_ei.pt")
        ea = torch.load(path + "/" + key + "_ea.pt")

        graph = Data(x=x,edge_index=ei,edge_attr=ea)
        datalist.append(graph)
    return datalist
