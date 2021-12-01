import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import os
from misc_modules.data_loader import data_loader
import numpy as np
import pandas as pd
import seaborn as sns


def plotter(path, model, database, latent_graph_dict, latent_graph_list, batch_size, device, name):

    # --- Making the directory where plots reside
    if not os.path.exists(path+"/"+name):
        os.mkdir(path+"/"+name)

    # --- Torch no grad for evaluation
    with torch.no_grad():
        batch_p1, batch_p2, batch_pm, batch_target, batch_Temperature = data_loader(database, latent_graph_dict,
                                                                                    latent_graph_list, batch_size,
                                                                                    shuffle=False)
        # Set up index to indicate which example is being plotted
        b_index = 0
        # Set up empty array for MAE calculation
        MAE_error = np.zeros(15)

        # Calculating each batch
        for btc in range(len(batch_p1)):
            p1 = batch_p1[btc].to(device)
            p2 = batch_p2[btc].to(device)
            pm = batch_pm[btc].to(device)
            targ = batch_target[btc].detach().cpu().numpy()
            Temperature = batch_Temperature[btc].to(device)

            out = model([p1, p2, pm, Temperature]).detach().cpu().numpy()

            for u in range(batch_size):
                # Make and save Dataframe that encompasses real and predicted values
                DF = pd.DataFrame(np.transpose([list(range(2,17)), out[u], targ[u]]),
                                  columns=["CARBON", "PREDICT", "REAL"])

                img_name = str(database.loc[b_index]["Papertag"]) + "_" + str(database.loc[b_index]["Molecule1"]) \
                           + "_" + str(database.loc[b_index]["Molecule2"]) + "_" \
                           + str(database.loc[b_index]["Membrane"]) + "_" + str(database.loc[b_index]["molp1"]) \
                           + "_" + str(database.loc[b_index]["molp2"]) + "_" + str(database.loc[b_index]["molpm"]) \
                           + "_" + str(database.loc[b_index]["Temperature"])

                DF.to_csv(path+"/"+name+"/"+img_name+".txt", sep="\t", index=False)

                # Adding to MAE error
                MAE_error += np.abs(DF["PREDICT"].values-DF["REAL"].values)

                # Plot the real and predicted results
                fig, axs = plt.subplots()
                axs.plot(DF["CARBON"].values, DF["PREDICT"].values, label="PREDICT")
                axs.plot(DF["CARBON"].values, DF["REAL"].values, label="REAL")
                axs.plot(DF["CARBON"].values, np.ones(15)*10, label=img_name)
                axs.set_ylabel("Order parameter $S$")
                axs.set_xlabel("Carbon position $n$")
                axs.legend(loc=3)
                axs.set_ylim([0, 0.35*10])
                axs.legend()
                fig.savefig(path+"/"+name+"/"+img_name+".png")
                plt.close("all")

                b_index += 1

        # Averaging the MAE error
        MAE_error /= (b_index+1)

        # Save MAE error
        MAE_DF = pd.DataFrame(np.transpose([list(range(2, 17)), list(MAE_error)]), columns=["CARBON", "MAE_ERROR"])
        MAE_DF.to_csv(path+"/MAE_ERROR_"+name+".txt", sep="\t", index=False)

    return MAE_DF

def plot_loss_error(path, num_epochs, loss_list, train_err_list, test_err_list):
    fig, axs = plt.subplots()
    axs.plot(range(1, num_epochs + 1), loss_list, label="Loss")
    axs.set_xlabel("Epoch")
    axs.legend(loc=3)
    axs.set_yscale('log')
    axs.legend()
    fig.savefig(path + "/LOSS.png")
    plt.close("all")

    fig, axs = plt.subplots()
    axs.plot(range(1, num_epochs + 1), train_err_list, label="Train Err")
    axs.plot(range(1, num_epochs + 1), test_err_list, label="Test Err")
    axs.set_xlabel("Epoch")
    axs.legend(loc=3)
    axs.set_yscale('log')
    axs.legend()
    fig.savefig(path + "/ERRORS.png")
    plt.close("all")

