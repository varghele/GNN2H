import torch
from torch_geometric.data import Batch

from torch.nn import Sequential as Seq, Linear as Lin, LeakyReLU, BatchNorm1d, LayerNorm
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.nn import MetaLayer


NO_NODE_FEATURES_ONE=17
NO_EDGE_FEATURES_ONE=7

NO_GRAPH_FEATURES_ONE=128
NO_GRAPH_FEATURES_TWO=64

NO_NODE_FEATURES_TWO=NO_GRAPH_FEATURES_ONE

NO_EDGE_FEATURES_TWO=32

ENCODING_NODE_1=64
ENCODING_EDGE_1=32
ENCODING_EDGE_2=NO_EDGE_FEATURES_TWO

HIDDEN_NODE_ONE,HIDDEN_NODE_TWO=128,128
HIDDEN_EDGE_ONE,HIDDEN_EDGE_TWO=64,64
HIDDEN_GRAPH_ONE,HIDDEN_GRAPH_TWO=128,128

device="cuda:0"

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class EdgeModel_ONE(torch.nn.Module):
    def __init__(self):
        super(EdgeModel_ONE, self).__init__()
        hidden = HIDDEN_EDGE_ONE
        in_channels = ENCODING_EDGE_1+2*ENCODING_NODE_1
        self.edge_mlp = Seq(Lin(in_channels, ENCODING_EDGE_1)).apply(init_weights)

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
        self.edge_mlp = Seq(Lin(in_channels, NO_EDGE_FEATURES_TWO)).apply(init_weights)

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
        in_channels_1 = ENCODING_EDGE_1+ENCODING_NODE_1
        in_channels_2 = hidden+ENCODING_NODE_1
        self.node_mlp_1 = Seq(Lin(in_channels_1, hidden)).apply(init_weights)
        self.node_mlp_2 = Seq(Lin(in_channels_2, ENCODING_NODE_1)).apply(init_weights)

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
        self.node_mlp_1 = Seq(Lin(in_channels_1, hidden)).apply(init_weights)
        self.node_mlp_2 = Seq(Lin(in_channels_2, NO_NODE_FEATURES_TWO)).apply(init_weights)

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
        in_channels=ENCODING_NODE_1+ENCODING_EDGE_1

        self.global_mlp = Seq(Lin(in_channels, NO_GRAPH_FEATURES_ONE)).apply(init_weights)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row,col=edge_index
        node_aggregate = scatter_add(x, batch, dim=0)
        edge_aggregate = scatter_add(edge_attr, batch[col], dim=0)
        out = torch.cat([node_aggregate, edge_aggregate], dim=1)
        return self.global_mlp(out)

class GlobalModel_TWO(torch.nn.Module):
    def __init__(self):
        super(GlobalModel_TWO, self).__init__()
        hidden = HIDDEN_GRAPH_TWO
        in_channels=NO_EDGE_FEATURES_TWO+NO_NODE_FEATURES_TWO

        self.global_mlp = Seq(Lin(in_channels, NO_GRAPH_FEATURES_TWO)).apply(init_weights)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row,col=edge_index
        node_aggregate = scatter_add(x, batch, dim=0)
        edge_aggregate = scatter_add(edge_attr, batch[col], dim=0)
        out = torch.cat([node_aggregate, edge_aggregate], dim=1)
        return self.global_mlp(out)

class GNN_FULL_CLASS(torch.nn.Module):
    def __init__(self, BATCH_SIZE, NO_MP_ONE, NO_MP_TWO):
        super(GNN_FULL_CLASS,self).__init__()
        self.meta1 = MetaLayer(EdgeModel_ONE(), NodeModel_ONE(), GlobalModel_ONE())
        self.meta2 = MetaLayer(EdgeModel_ONE(), NodeModel_ONE(), GlobalModel_ONE())
        self.meta3 = MetaLayer(EdgeModel_TWO(), NodeModel_TWO(), GlobalModel_TWO())

        self.encoding_edge_1=Seq(Lin(NO_EDGE_FEATURES_ONE, ENCODING_EDGE_1)).apply(init_weights)

        self.encoding_node_1 = Seq(Lin(NO_NODE_FEATURES_ONE, ENCODING_NODE_1)).apply(init_weights)


        self.encoding_edge_2 = Seq(Lin(2, NO_EDGE_FEATURES_TWO)).apply(init_weights)

        self.encoding_node_2 = Seq(Lin(NO_GRAPH_FEATURES_ONE, NO_GRAPH_FEATURES_ONE)).apply(init_weights)


        self.mlp_last = Seq(Lin(NO_GRAPH_FEATURES_TWO, NO_GRAPH_FEATURES_TWO), LeakyReLU(),
                            LayerNorm(NO_GRAPH_FEATURES_TWO),
                            Lin(NO_GRAPH_FEATURES_TWO, 15)).apply(init_weights)

        self.batch_size = BATCH_SIZE
        self.no_mp_one = NO_MP_ONE
        self.no_mp_two = NO_MP_TWO



    def forward(self,plist):
        # Here we instantiate the three molecular graphs for the first level GNN
        p1,p2,pm,Temperature=plist
        x_p1,ei_p1,ea_p1,u_p1,btc_p1=p1.x,p1.edge_index,p1.edge_attr,p1.y,p1.batch
        x_p2,ei_p2,ea_p2,u_p2,btc_p2=p2.x,p2.edge_index,p2.edge_attr,p2.y,p2.batch
        x_pm,ei_pm,ea_pm,u_pm,btc_pm=pm.x,pm.edge_index,pm.edge_attr,pm.y,pm.batch

        # Embed the node and edge features
        enc_x_p1 = self.encoding_node_1(x_p1)
        enc_x_p2 = self.encoding_node_1(x_p2)
        enc_x_pm = self.encoding_node_1(x_pm)

        enc_ea_p1 = self.encoding_edge_1(ea_p1)
        enc_ea_p2 = self.encoding_edge_1(ea_p2)
        enc_ea_pm = self.encoding_edge_1(ea_pm)

        #Create the empty molecular graphs for feature extraction, graph level one
        u1=torch.full(size=(self.batch_size, NO_GRAPH_FEATURES_ONE), fill_value=0.1, dtype=torch.float).to(device)
        u2=torch.full(size=(self.batch_size, NO_GRAPH_FEATURES_ONE), fill_value=0.1, dtype=torch.float).to(device)
        um=torch.full(size=(self.batch_size, NO_GRAPH_FEATURES_ONE), fill_value=0.1, dtype=torch.float).to(device)

        # Now the first level rounds of message passing are performed, molecular features are constructed
        for _ in range(self.no_mp_one):
            enc_x_p1, enc_ea_p1, u1 = self.meta1(x=enc_x_p1, edge_index=ei_p1, edge_attr=enc_ea_p1, u=u1, batch=btc_p1)
            enc_x_p2, enc_ea_p2, u2 = self.meta1(x=enc_x_p2, edge_index=ei_p2, edge_attr=enc_ea_p2, u=u2, batch=btc_p2)
            enc_x_pm, enc_ea_pm, um = self.meta1(x=enc_x_pm, edge_index=ei_pm, edge_attr=enc_ea_pm, u=um, batch=btc_pm)


        # --- GRAPH GENERATION SECOND LEVEL

        #Encode the nodes second level
        u1 = self.encoding_node_2(u1)
        u2 = self.encoding_node_2(u2)
        um = self.encoding_node_2(um)

        # Instantiate new Batch object for second level graph
        nu_Batch = Batch()

        # Create edge indices for the second level (no connection between p1 and p2
        nu_ei = []
        temp_ei = torch.tensor([[0, 2, 1, 2], [2, 0, 2, 1]], dtype=torch.long)
        for b in range(self.batch_size):
            nu_ei.append(temp_ei + b * 3)  # +3 because this is the number of nodes in the second stage graph
        nu_Batch.edge_index = torch.cat(nu_ei, dim=1)

        # Create the edge features for the second level graphs from the first level "u"s
        p1_div_pm = u_p1 / u_pm
        p2_div_pm = u_p2 / u_pm
        #p1_div_p2 = u_p1 / (u_p2-1)

        # Concatenate the temperature and molecular percentages
        concat_T_p1pm = torch.transpose(torch.stack([Temperature,p1_div_pm]),0,1)
        concat_T_p2pm = torch.transpose(torch.stack([Temperature,p2_div_pm]),0,1)

        # Encode the edge features
        concat_T_p1pm = self.encoding_edge_2(concat_T_p1pm)
        concat_T_p2pm = self.encoding_edge_2(concat_T_p2pm)

        nu_ea=[]
        for b in range(self.batch_size):
            temp_ea=[]
            # Appending twice because of bidirectional edges
            temp_ea.append(concat_T_p1pm[b])
            temp_ea.append(concat_T_p1pm[b])
            temp_ea.append(concat_T_p2pm[b])
            temp_ea.append(concat_T_p2pm[b])
            nu_ea.append(torch.stack(temp_ea, dim=0))
        nu_Batch.edge_attr = torch.cat(nu_ea, dim=0)

        # Create new nodes in the batch
        nu_x = []
        for b in range(self.batch_size):
            temp_x = []
            temp_x.append(u1[b])
            temp_x.append(u2[b])
            temp_x.append(um[b])
            nu_x.append(torch.stack(temp_x, dim=0))
        nu_Batch.x = torch.cat(nu_x, dim=0)

        # Create new graph level target
        gamma = torch.full(size=(self.batch_size, NO_GRAPH_FEATURES_TWO), fill_value=0.1, dtype=torch.float)
        nu_Batch.y = gamma
        # Create new batch
        nu_btc = []
        for b in range(self.batch_size):
            nu_btc.append(torch.full(size=[3], fill_value=(int(b)), dtype=torch.long))
        nu_Batch.batch = torch.cat(nu_btc, dim=0)

        nu_Batch = nu_Batch.to(device)

        # --- MESSAGE PASSING LVL 2
        for _ in range(self.no_mp_two):
            nu_Batch.x, nu_Batch.edge_attr, nu_Batch.y = self.meta3(x=nu_Batch.x, edge_index=nu_Batch.edge_index,
                                                                    edge_attr=nu_Batch.edge_attr, u=nu_Batch.y,
                                                                    batch=nu_Batch.batch)

        c = self.mlp_last(nu_Batch.y)

        return c