import torch

from meta_modules.edge_models import EdgeModel
from meta_modules.node_models import NodeModel
from meta_modules.graph_models import GlobalModel

from graph_modules.encoding import EncodingMLP
from graph_modules.encoding import LastMLP

from torch_geometric.nn import MetaLayer
from torch_geometric.data import Batch


class GNN(torch.nn.Module):
    def __init__(self, BATCH_SIZE,
                 NO_EF_ONE, NO_EF_TWO,
                 NO_NF_ONE,
                 NO_GF_ONE, NO_GF_TWO,
                 NO_LAYERS, ENCODING_SIZE,
                 NO_MP_ONE, NO_MP_TWO, NORM, device):
        super(GNN, self).__init__()

        def init_weights(m):
            if type(m) == torch.nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        # Define Edge models stage 1 and 2, node encoding is twice the size of edge encoding
        # Stage 2 edges are encoded temperature and molecular percentages
        self.EdgeModel_ONE = EdgeModel(ENCODING_SIZE, ENCODING_SIZE*2, NO_LAYERS, NORM)
        self.EdgeModel_TWO = EdgeModel(NO_EF_TWO, NO_GF_ONE, NO_LAYERS, NORM)

        # Define Node models stage 1 and 2,
        # nodes stage 2 are encoded graphs stage 1
        self.NodeModel_ONE = NodeModel(ENCODING_SIZE, ENCODING_SIZE*2, NO_LAYERS, NORM)
        self.NodeModel_TWO = NodeModel(NO_EF_TWO, NO_GF_ONE, NO_LAYERS, NORM)

        # Graph Stage 1 feeds off of encodings, Graph stage 2 has defined edge size
        self.GlobalModel_ONE = GlobalModel(ENCODING_SIZE, ENCODING_SIZE*2, NO_GF_ONE, NO_LAYERS, NORM)
        self.GlobalModel_TWO = GlobalModel(NO_EF_TWO, NO_GF_ONE, NO_GF_TWO, NO_LAYERS, NORM)

        # Meta1 and Meta2 are SAME models, for molecules and membrane
        # Meta3 is same structure, different model for second stage MP
        self.meta1 = MetaLayer(self.EdgeModel_ONE, self.NodeModel_ONE, self.GlobalModel_ONE)
        self.meta2 = MetaLayer(self.EdgeModel_ONE, self.NodeModel_ONE, self.GlobalModel_ONE)
        self.meta3 = MetaLayer(self.EdgeModel_TWO, self.NodeModel_TWO, self.GlobalModel_TWO)

        # Encoding for edges stage 1 and 2
        self.encoding_edge_1 = EncodingMLP(NO_EF_ONE, ENCODING_SIZE, NO_LAYERS, NORM).apply(init_weights)
        self.encoding_edge_2 = EncodingMLP(2, NO_EF_TWO, NO_LAYERS, NORM).apply(init_weights)

        # Encodings for nodes stage 1 and 2, node 2 encoding maybe not necessary
        self.encoding_node_1 = EncodingMLP(NO_NF_ONE, ENCODING_SIZE*2, NO_LAYERS, NORM).apply(init_weights)
        self.encoding_node_2 = EncodingMLP(NO_GF_ONE, NO_GF_ONE, NO_LAYERS, NORM).apply(init_weights)

        # Setting up last stage MLP to pass graph features through
        self.last_stage_mlp = LastMLP(NO_GF_TWO, 15, NO_LAYERS, NORM).apply(init_weights)


        # Setting batch size and message passing rounds and graph features for self
        self.batchsize = BATCH_SIZE
        self.no_mp1 = NO_MP_ONE
        self.no_mp2 = NO_MP_TWO

        self.no_gf_one = NO_GF_ONE
        self.no_gf_two = NO_GF_TWO

        # Setting up internal device variable to push GNN to GPU
        self.device = device


    def forward(self, plist):
        # Here we instantiate the three molecular graphs for the first level GNN
        p1, p2, pm, Temperature = plist
        x_p1, ei_p1, ea_p1, u_p1, btc_p1 = p1.x, p1.edge_index, p1.edge_attr, p1.y, p1.batch
        x_p2, ei_p2, ea_p2, u_p2, btc_p2 = p2.x, p2.edge_index, p2.edge_attr, p2.y, p2.batch
        x_pm, ei_pm, ea_pm, u_pm, btc_pm = pm.x, pm.edge_index, pm.edge_attr, pm.y, pm.batch

        # Embed the node and edge features
        enc_x_p1 = self.encoding_node_1(x_p1)
        enc_x_p2 = self.encoding_node_1(x_p2)
        enc_x_pm = self.encoding_node_1(x_pm)

        enc_ea_p1 = self.encoding_edge_1(ea_p1)
        enc_ea_p2 = self.encoding_edge_1(ea_p2)
        enc_ea_pm = self.encoding_edge_1(ea_pm)

        # Create the empty molecular graphs for feature extraction, graph level one
        u1 = torch.full(size=(self.batchsize, self.no_gf_one), fill_value=0.1, dtype=torch.float).to(self.device)
        u2 = torch.full(size=(self.batchsize, self.no_gf_one), fill_value=0.1, dtype=torch.float).to(self.device)
        um = torch.full(size=(self.batchsize, self.no_gf_one), fill_value=0.1, dtype=torch.float).to(self.device)

        # Now the first level rounds of message passing are performed, molecular features are constructed
        for _ in range(self.no_mp1):
            enc_x_p1, enc_ea_p1, u1 = self.meta1(x=enc_x_p1, edge_index=ei_p1, edge_attr=enc_ea_p1, u=u1, batch=btc_p1)
            enc_x_p2, enc_ea_p2, u2 = self.meta1(x=enc_x_p2, edge_index=ei_p2, edge_attr=enc_ea_p2, u=u2, batch=btc_p2)
            enc_x_pm, enc_ea_pm, um = self.meta1(x=enc_x_pm, edge_index=ei_pm, edge_attr=enc_ea_pm, u=um, batch=btc_pm)

        # --- GRAPH GENERATION SECOND LEVEL

        # Encode the nodes second level
        u1 = self.encoding_node_2(u1)
        u2 = self.encoding_node_2(u2)
        um = self.encoding_node_2(um)

        # Instantiate new Batch object for second level graph
        nu_Batch = Batch()

        # Create edge indices for the second level (no connection between p1 and p2
        nu_ei = []
        temp_ei = torch.tensor([[0, 2, 1, 2], [2, 0, 2, 1]], dtype=torch.long)
        for b in range(self.batchsize):
            nu_ei.append(temp_ei + b * 3)  # +3 because this is the number of nodes in the second stage graph
        nu_Batch.edge_index = torch.cat(nu_ei, dim=1)

        # Create the edge features for the second level graphs from the first level "u"s
        p1_div_pm = u_p1 / u_pm
        p2_div_pm = u_p2 / u_pm
        # p1_div_p2 = u_p1 / (u_p2-1)

        # Concatenate the temperature and molecular percentages
        concat_T_p1pm = torch.transpose(torch.stack([Temperature, p1_div_pm]), 0, 1)
        concat_T_p2pm = torch.transpose(torch.stack([Temperature, p2_div_pm]), 0, 1)

        # Encode the edge features
        concat_T_p1pm = self.encoding_edge_2(concat_T_p1pm)
        concat_T_p2pm = self.encoding_edge_2(concat_T_p2pm)

        nu_ea = []
        for b in range(self.batchsize):
            temp_ea = []
            # Appending twice because of bidirectional edges
            temp_ea.append(concat_T_p1pm[b])
            temp_ea.append(concat_T_p1pm[b])
            temp_ea.append(concat_T_p2pm[b])
            temp_ea.append(concat_T_p2pm[b])
            nu_ea.append(torch.stack(temp_ea, dim=0))
        nu_Batch.edge_attr = torch.cat(nu_ea, dim=0)

        # Create new nodes in the batch
        nu_x = []
        for b in range(self.batchsize):
            temp_x = []
            temp_x.append(u1[b])
            temp_x.append(u2[b])
            temp_x.append(um[b])
            nu_x.append(torch.stack(temp_x, dim=0))
        nu_Batch.x = torch.cat(nu_x, dim=0)

        # Create new graph level target
        gamma = torch.full(size=(self.batchsize, self.no_gf_two), fill_value=0.1, dtype=torch.float)
        nu_Batch.y = gamma
        # Create new batch
        nu_btc = []
        for b in range(self.batchsize):
            nu_btc.append(torch.full(size=[3], fill_value=(int(b)), dtype=torch.long))
        nu_Batch.batch = torch.cat(nu_btc, dim=0)

        nu_Batch = nu_Batch.to(self.device)
        # Fix to push tensors to CUDA
        x_mp2 = nu_Batch.x.to(self.device)
        ei_mp2 = nu_Batch.edge_index.to(self.device)
        ea_mp2 = nu_Batch.edge_attr.to(self.device)
        u_mp2 = nu_Batch.y.to(self.device)
        batch_mp2 = nu_Batch.batch.to(self.device)

        # --- MESSAGE PASSING LVL 2
        for _ in range(self.no_mp2):
            x_mp2, ea_mp2, u_mp2 = self.meta3(x=x_mp2, edge_index=ei_mp2,
                                              edge_attr=ea_mp2, u=u_mp2,
                                              batch=batch_mp2)
        c = self.last_stage_mlp(u_mp2)

        return c
