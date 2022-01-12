import torch
import numpy as np

from graph_modules.graph_net_A import GNN_FULL_CLASS as GNN_A
from graph_modules.graph_net_B import GNN_FULL_CLASS as GNN_B
from graph_modules.graph_net_C import GNN_FULL_CLASS as GNN_C
from graph_modules.graph_net_V import GNN_FULL_CLASS as GNN_V
# ---------------------------------------------------------------------------------------------------------------------
from misc_modules.plotter import plotter, plot_loss_error
from misc_modules.load_graph_data import load_graph_data
# from misc_modules.train_and_evaluate import *
from misc_modules.data_loader import data_loader

import pandas as pd
from itertools import combinations

import sys
import os
import time


# --------- Get device to run this on
gpu = 0
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(device)
print(f"DEVICE: {device}")

# ---------------------------------------------------------------------------------------------------------------------
# --------- Train and Eval funcs have to be defined here, don't work otherwise
def train(batch_size, device, database, latent_graph_dict, latent_graph_list):
    # Set model to training mode
    #model.train()
    # Set up loss
    loss_avg = 0

    # Get batches from data loader
    batch_p1, batch_p2, batch_pm, batch_target, batch_Temperature = data_loader(database, latent_graph_dict,
                                                                                latent_graph_list, batch_size,
                                                                                shuffle=True
                                                                                )
    for btc in range(len(batch_p1)):
        # Get molecules 1 and two, as well as the membrane in batches, together with targets and temperature
        p1 = batch_p1[btc].to(device)
        p2 = batch_p2[btc].to(device)
        pm = batch_pm[btc].to(device)
        targ = batch_target[btc].to(device)
        Temperature = batch_Temperature[btc].to(device)

        # Forward pass and gradient descent
        out = model([p1, p2, pm, Temperature])
        loss = criterion(out, targ)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        loss_avg += loss.item()

        # Empty CUDA cache
        torch.cuda.empty_cache()

    return loss_avg/len(batch_p1)


def evaluate(batch_size, device, database, latent_graph_dict, latent_graph_list):
    # Set model to evaluation
    #model.eval()
    # Set up loss
    set_loss = []

    with torch.no_grad():
        # Get batches from data loader
        batch_p1, batch_p2, batch_pm, batch_target, batch_Temperature = data_loader(database, latent_graph_dict,
                                                                                    latent_graph_list, batch_size,
                                                                                    shuffle=False
                                                                                    )
        for btc in range(len(batch_p1)):
            # Get molecules 1 and two, as well as the membrane in batches, together with targets and temperature
            p1 = batch_p1[btc].to(device)
            p2 = batch_p2[btc].to(device)
            pm = batch_pm[btc].to(device)
            targ = batch_target[btc].detach().cpu().numpy()
            Temperature = batch_Temperature[btc].to(device)

            # Calculate loss
            out = model([p1, p2, pm, Temperature]).detach().cpu().numpy()
            #loss = criterion(out, targ)

            #loss_avg += loss.item()
            set_loss.append(np.mean(np.abs(out - targ)))

            # Empty CUDA cache
            torch.cuda.empty_cache()

    return np.sum(set_loss)/len(set_loss)
# ---------------------------------------------------------------------------------------------------------------------


# --------- Load the main INPUT training file
# INPUT of the main training list/CONFIG DB
# INPUT_PATH = sys.argv[1]
INPUT_PATH = "network_commands.txt"

# Create CONFIG DATABASE from input and get name
CONFIG_DB = pd.read_csv(INPUT_PATH, sep="\t", header=1, skiprows=0)
with open(INPUT_PATH) as f:
    CONFIG_NAME = f.readline().strip()

# Define how many cross-validation splits you want
CV_SPLITS = 4


# --------- Load the MAIN DATABASE
MAIN_DB = pd.read_csv("data/DB_order_parameters_POPC.csv", sep=",", header=0)


# --------- Create latent graph dictionary and load latent graphs
# FILES INPUT, of where the raw mol2 files and the processed graphs are
RAW_PATH = "2_KOS_mol2"
PROC_PATH = "3_latent_graphs"

# Create dictionary to get the mapping of the latent graph data to molecule name
dict_items = [[os.listdir(RAW_PATH)[i][:-5], i] for i in range(len(os.listdir(RAW_PATH)))]
latent_graph_dict = {key: value for (key, value) in dict_items}
# Append empty graph to dict
latent_graph_dict["0"] = -1

# Latent graph list (already scaled, if you used 12_make_graphs.py)
latent_graph_list = load_graph_data(PROC_PATH, latent_graph_dict)


# --------- Get EDGE and NODE features from the latent graph list
NO_EF_ONE = latent_graph_list[0].edge_attr.shape[1]
NO_NF_ONE = latent_graph_list[0].x.shape[1]


# --------- SETTING UP THE MAIN LOOP
for i in range(len(CONFIG_DB)):
    print("CURRENT CONFIG: "+str(i+1))
    # --- Get constants from CONFIG
    MODELNO = str(CONFIG_DB["MODELNO"].iloc[i])
    EPOCHS = int(CONFIG_DB["EPOCHS"].iloc[i])
    BATCH_SIZE = int(CONFIG_DB["BATCH_SIZE"].iloc[i])
    LEARNING_RATE = float(CONFIG_DB["LR"].iloc[i])
    NO_MP_ONE = int(CONFIG_DB["NO_MP_ONE"].iloc[i])
    NO_MP_TWO = int(CONFIG_DB["NO_MP_TWO"].iloc[i])

    # --- Set up the main folder
    if not os.path.exists(CONFIG_NAME):
        os.mkdir(CONFIG_NAME)

    # --- Setting up current config folder
    cur_CNAME = "CONFIG" + str(i) + "_MODEL_" + MODELNO + "_" + str(NO_MP_ONE) + "_" + str(NO_MP_TWO)
    if not os.path.exists(CONFIG_NAME+"/"+cur_CNAME):
        os.mkdir(CONFIG_NAME+"/"+cur_CNAME)

    # --- Perform cross validation split:
    for j in range(CV_SPLITS):
        if not os.path.exists(CONFIG_NAME+"/"+cur_CNAME+"/"+str(j+1)):
            os.mkdir(CONFIG_NAME+"/"+cur_CNAME+"/"+str(j+1))

    # Shuffle DB to get different results
    shuffled_DB = MAIN_DB.sample(frac=1).reset_index(drop=True)
    # Set up split indices in CV_SPLITS different lists
    split_indices = [list(k) for k in np.array_split(list(range(len(shuffled_DB))), CV_SPLITS)]

    # Get each combination of train-test-split e.g. [0,1,2,3,4], last element is test set
    split_combos = [list(k) for k in combinations(list(range(CV_SPLITS)), CV_SPLITS-1)]
    split_combos = [k+list(set(list(range(CV_SPLITS)))-set(k)) for k in split_combos]

    # --- Get all indices combined that will go into train/test sets
    train_indices = []
    test_indices = []
    for comb in split_combos:
        temp_train_idx = []
        temp_test_idx = split_indices[comb[-1]]
        for idx in comb[:-1]:
            temp_train_idx += split_indices[idx]
        train_indices.append(temp_train_idx)
        test_indices.append(temp_test_idx)

    # --------- SAVE THE NEW DATAFRAMES
    if not os.path.exists(CONFIG_NAME+"/"+cur_CNAME+"/"+str(1)+"/train_df.csv"):
        for j in range(CV_SPLITS):
            # --- Set up the new pandas dataframes
            train_df = MAIN_DB.loc[train_indices[j]]
            test_df = MAIN_DB.loc[test_indices[j]]

            # --- Save the new frames
            train_df.to_csv(CONFIG_NAME+"/"+cur_CNAME+"/"+str(j+1)+"/train_df.csv", sep=",", index=False)
            test_df.to_csv(CONFIG_NAME+"/"+cur_CNAME+"/"+str(j+1)+"/test_df.csv", sep=",", index=False)

    # --------- TRAINING CV LOOP
    for j in range(CV_SPLITS):
        print("CV: "+str(j+1)+"/"+str(CV_SPLITS))
        if not os.path.exists(CONFIG_NAME+"/"+cur_CNAME+"/"+str(j+1)+"/model_statedict"):
            # --- Load training and testing DF
            TRAIN_DF = pd.read_csv(CONFIG_NAME+"/"+cur_CNAME+"/"+str(j+1)+"/train_df.csv", sep=",", header=0)
            TEST_DF = pd.read_csv(CONFIG_NAME+"/"+cur_CNAME+"/"+str(j+1)+"/test_df.csv", sep=",", header=0)

            # --- Set up Graph Network
            if MODELNO == "A":
                model = GNN_A(BATCH_SIZE, NO_MP_ONE, NO_MP_TWO).to(device)
            if MODELNO == "B":
                model = GNN_B(BATCH_SIZE, NO_MP_ONE, NO_MP_TWO).to(device)
            if MODELNO == "C":
                model = GNN_C(BATCH_SIZE, NO_MP_ONE, NO_MP_TWO).to(device)
            if MODELNO == "V":
                model = GNN_V(BATCH_SIZE, NO_MP_ONE, NO_MP_TWO).to(device)
            # --- Optimizer and Criterion and LR Decay
            optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
            #You might want to decay the learning rate after some steps, then enable scheduler
            #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
            criterion = torch.nn.MSELoss()
            # --- Backward hook with gradient clipping (can be enabled)
            """clip_value = 0.1
            for p in model.parameters():
                p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))"""

            # --- Training Loop
            loss_list = []
            train_err_list = []
            test_err_list = []

            TIME = time.time()
            model.train()
            for epoch in range(EPOCHS):
                tloss = train(BATCH_SIZE, device, TRAIN_DF,
                              latent_graph_dict, latent_graph_list)
                train_err = evaluate(BATCH_SIZE, device, TRAIN_DF,
                                     latent_graph_dict, latent_graph_list)
                test_err = evaluate(BATCH_SIZE, device, TEST_DF,
                                    latent_graph_dict, latent_graph_list)

                #Take a scheduler step to decay the learning rate (can be enabled)
                #scheduler.step()

                # Check if something has failed, and Loss is NaN, if so, break, and restart with restarter
                if np.isnan(tloss)==True:
                    sys.exit()

                loss_list.append(tloss)
                train_err_list.append(train_err)
                test_err_list.append(test_err)

                print('Epoch: {:03d}, Loss: {:.5f}, Train Err: {:.5f}, Test Err: {:.5f}'.format(epoch+1, tloss,
                                                                                                train_err, test_err))
            print("\n")

            # --- Saving time that training needed
            dat = open(CONFIG_NAME+"/"+cur_CNAME+"/"+str(j+1)+"/time.txt","w")
            dat.write("TIME: "+str(time.time() - TIME)+" s\n")
            dat.write("or\n")
            dat.write("TIME: " + str((time.time() - TIME) / 60) + " m\n")
            dat.close()

            # --- Saving and plotting loss function et al
            loss_err_df = pd.DataFrame(np.array([list(range(EPOCHS)), loss_list,
                                                 train_err_list, test_err_list]).transpose(),
                                       columns=["Epoch", "Loss", "Train_Err", "Test_Err"])
            loss_err_df.to_csv(CONFIG_NAME+"/"+cur_CNAME+"/"+str(j+1)+"/LOSS_ERRORS.txt", sep="\t", index=False)

            plot_loss_error(CONFIG_NAME+"/"+cur_CNAME+"/"+str(j+1), EPOCHS, loss_list, train_err_list, test_err_list)


            # --------- PLOTTING
            TRAIN_MAE_CV = plotter(CONFIG_NAME+"/"+cur_CNAME+"/"+str(j+1), model, TRAIN_DF, latent_graph_dict,
                                   latent_graph_list, BATCH_SIZE, device, "TRAIN")
            TEST_MAE_CV = plotter(CONFIG_NAME + "/" + cur_CNAME + "/" + str(j + 1), model, TEST_DF, latent_graph_dict,
                                  latent_graph_list, BATCH_SIZE, device, "TEST")

            # --- Saving statistics
            if not os.path.exists(CONFIG_NAME+"/"+cur_CNAME+"/"+"TRAIN_MAE.txt"):
                pd.DataFrame(np.transpose(list(range(2, 17))), columns=["CARBON"]).to_csv(CONFIG_NAME+"/"+cur_CNAME+"/" +
                                                                                      "TRAIN_MAE.txt", sep="\t",
                                                                                      index=False)
            if not os.path.exists(CONFIG_NAME+"/"+cur_CNAME+"/"+"TEST_MAE.txt"):
                pd.DataFrame(np.transpose(list(range(2, 17))), columns=["CARBON"]).to_csv(CONFIG_NAME+"/"+cur_CNAME+"/" +
                                                                                      "TEST_MAE.txt", sep="\t",
                                                                                      index=False)

            STAT_TRAIN_DF = pd.read_csv(CONFIG_NAME+"/"+cur_CNAME+"/"+"TRAIN_MAE.txt", header=0, sep="\t")
            newaxis = ["CARBON"] + ["MAE_ERROR" + str(k + 1) for k in range(len(STAT_TRAIN_DF.columns))]
            STAT_TRAIN_DF = pd.concat([STAT_TRAIN_DF, TRAIN_MAE_CV["MAE_ERROR"]], axis=1).set_axis(newaxis, axis=1,
                                                                                                   inplace=False)
            STAT_TRAIN_DF.to_csv(CONFIG_NAME+"/"+cur_CNAME+"/"+"TRAIN_MAE.txt", sep="\t", index=False)

            STAT_TEST_DF = pd.read_csv(CONFIG_NAME + "/" + cur_CNAME + "/" + "TEST_MAE.txt", header=0, sep="\t")
            newaxis = ["CARBON"] + ["MAE_ERROR" + str(k + 1) for k in range(len(STAT_TEST_DF.columns))]
            STAT_TEST_DF = pd.concat([STAT_TEST_DF, TEST_MAE_CV["MAE_ERROR"]], axis=1).set_axis(newaxis, axis=1,
                                                                                                 inplace=False)
            STAT_TEST_DF.to_csv(CONFIG_NAME + "/" + cur_CNAME + "/" + "TEST_MAE.txt", sep="\t", index=False)

            # --------- SAVING MODEL AND EVERYTHING
            torch.save(model.state_dict(), CONFIG_NAME+"/"+cur_CNAME+"/"+str(j+1)+"/model_statedict")
