import torch
from torch_geometric.data import Batch


def data_loader(database, latent_graph_dict, latent_graph_list, batch_size=4, shuffle=True):
    # -- This loader creates Databatches from a Database, run on either the training or validation set!

    # Establish empty lists
    graph_list_p1 = []
    graph_list_p2 = []
    graph_list_pm = []
    target_list = []
    Temperature_list = []

    # Iterate through database and construct latent graphs of the molecules, as well as the target tensors
    for itm in range(len(database)):
        graph_p1 = latent_graph_list[latent_graph_dict[database["Molecule1"].loc[itm]]]
        graph_p2 = latent_graph_list[latent_graph_dict[database["Molecule2"].loc[itm]]]
        graph_pm = latent_graph_list[latent_graph_dict[database["Membrane"].loc[itm]]]

        graph_p1.y = torch.tensor([float(database["molp1"].loc[itm])])
        graph_p2.y = torch.tensor([float(database["molp2"].loc[itm])])
        graph_pm.y = torch.tensor([float(database["molpm"].loc[itm])])

        graph_list_p1.append(graph_p1)
        graph_list_p2.append(graph_p2)
        graph_list_pm.append(graph_pm)

        target = torch.tensor([database[str(i)].loc[itm]*10 for i in range(2, 17)], dtype=torch.float)
        target_list.append(target)

        Temperature_list.append(database["Temperature"].loc[itm])

    # Generate a shuffled integer list that then produces distinct batches
    idxs = torch.randperm(len(database)).tolist()
    if shuffle==False:
        idxs = sorted(idxs)

    # Generate the batches
    batch_p1 = []
    batch_p2 = []
    batch_pm = []
    batch_target = []
    batch_Temperature = []

    for b in range(len(database))[::batch_size]:
        # Creating empty Batch objects
        B_p1 = Batch()
        B_p2 = Batch()
        B_pm = Batch()

        # Creating empty lists that will be concatenated later (advanced minibatching)
        stack_x_p1, stack_x_p2, stack_x_pm = [], [], []
        stack_ei_p1, stack_ei_p2, stack_ei_pm = [], [], []
        stack_ea_p1, stack_ea_p2, stack_ea_pm = [], [], []
        stack_y_p1, stack_y_p2, stack_y_pm = [], [], []
        stack_btc_p1, stack_btc_p2, stack_btc_pm = [], [], []

        stack_target = []
        stack_Temperature = []

        # Creating the batch shifts, that shift the edge indices
        b_shift_p1 = 0
        b_shift_p2 = 0
        b_shift_pm = 0

        for i in range(batch_size):
            if (b+i) < len(database):
                stack_x_p1.append(graph_list_p1[idxs[b + i]].x)
                stack_x_p2.append(graph_list_p2[idxs[b + i]].x)
                stack_x_pm.append(graph_list_pm[idxs[b + i]].x)

                stack_ei_p1.append(graph_list_p1[idxs[b + i]].edge_index+b_shift_p1)
                stack_ei_p2.append(graph_list_p2[idxs[b + i]].edge_index+b_shift_p2)
                stack_ei_pm.append(graph_list_pm[idxs[b + i]].edge_index+b_shift_pm)

                # Updating the shifts
                b_shift_p1 += graph_list_p1[idxs[b + i]].x.size()[0]
                b_shift_p2 += graph_list_p2[idxs[b + i]].x.size()[0]
                b_shift_pm += graph_list_pm[idxs[b + i]].x.size()[0]

                stack_ea_p1.append(graph_list_p1[idxs[b + i]].edge_attr)
                stack_ea_p2.append(graph_list_p2[idxs[b + i]].edge_attr)
                stack_ea_pm.append(graph_list_pm[idxs[b + i]].edge_attr)

                stack_y_p1.append(graph_list_p1[idxs[b + i]].y)
                stack_y_p2.append(graph_list_p2[idxs[b + i]].y)
                stack_y_pm.append(graph_list_pm[idxs[b + i]].y)

                stack_btc_p1.append(
                    torch.full([graph_list_p1[idxs[b + i]].x.size()[0]], fill_value=int(i), dtype=torch.long))
                # FILL VALUE IS PROBABLY JUST i! (OR NOT)
                stack_btc_p2.append(
                    torch.full([graph_list_p2[idxs[b + i]].x.size()[0]], fill_value=int(i), dtype=torch.long))
                stack_btc_pm.append(
                    torch.full([graph_list_pm[idxs[b + i]].x.size()[0]], fill_value=int(i), dtype=torch.long))

                stack_target.append(target_list[idxs[b + i]])
                stack_Temperature.append(Temperature_list[int(b+i)])

        # print(stack_y_p1)
        # print(stack_x_p1)

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

        B_target = torch.stack(stack_target, dim=0)

        B_Temperature = torch.Tensor(stack_Temperature)

        # Appending batches
        batch_p1.append(B_p1)
        batch_p2.append(B_p2)
        batch_pm.append(B_pm)
        batch_target.append(B_target)
        batch_Temperature.append(B_Temperature)

    # Return
    return [batch_p1, batch_p2, batch_pm, batch_target, batch_Temperature]
