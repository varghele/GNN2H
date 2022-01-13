# The G-2-R (Graph-to-order) neural network

Predicting smoothed 2H NMR order parameters of POPC in presence of up to two added molecules.  

This is the repository for the Paper *Predicting 2H NMR acyl chain order parameters with graph neural networks* by M. Fischer, B. Schwarze and H. A. Scheidt.

In the following, you will find the repository explanation, as well as a guide to create datasets and run the experiments yourself.  

## Files and directories in the repository

`123_make_graphs.py` - This is the first file to run. It takes as input the files in the directory `1_raw_mol2` and first rotates and translates them to the same coordinate system.  

`network_commands.txt`- In here, commands for `main_train_script.py` are stored.

`requirements.txt` - Requirements to create a Conda environment to run the script.

`residentsleeper.bat` - File to automatically run the main script on a loop.

### 1_raw_mol2

In this directory, there are the raw `.mol2` files of all of the molecules in the dataset. The files there each have a TAG as their name. The unequivocal identification of this tag can be found in `data\smiles.txt`, where you can find the SMILES associated with each tag. Also, the TAG and its IUPAC name is found in the supplementary information (once published, I will provide the link to it here).  

If you want to extend this dataset by your own molecules, simply generate a Sybyl `.mol2` file with an unused TAG by using OpenBabel using a Generator such as [this one](https://www.cheminfo.org/Chemistry/Cheminformatics/FormatConverter/index.html), and add it to `1_raw_mol2` as well as the TAG to `data\DB_order_parameters_POPC.csv`.

### data

In this directory, you will find the `smiles.txt`, in which the SMILES codes for each molecule is saved for unequivocal identification of the molecule.  

You will also find the raw data used for this experiment in `2H_data`in the form of .txt files. The files in there are simply for completion (and because not all data was used). The training will instead rely on the database `DB_order_parameters_POPC.csv`, which was generated from those files.  

In the files, the 2H NMR order parameters are saved as comma-separated values, where the first column is the number **n** of the order parameter (2...16), and the second column is the order parameter **S**.  

The names have the following form:  

`PAPERTAG_MOL1_MOL2_MEMB_P1_P2_PM_TEMP_wtXX.txt`,

where: 

**PAPERTAG** - is the reference tag assigned to the paper the data was acquired for. This can be found in the supplementary information.  
**MOL1** - TAG of the first molecule, **0** if no molecule was used  
**MOL2** - TAG of the second molecule, **0** if no molecule was used  
**MEMB** - TAG of the membrane, here only **POPC**  
**P1** - molecular ratio of first molecule (between 0 and 1)  
**P2** - molecular ratio of second molecule (between 0 and 1)  
**PM** - molecular ratio of membrane (between 0 and 1)  
**TEMP** - Temperature this measurement was recorded at in Kelvin  
**wtXX** - water content of the sample in weight% (wt01-wt99), or wtNULL if unknown

There is also the file `DB_order_parameters_POPC.csv`, which is the database file that is used in training.

### graph_modules

In here, you have the `.py` files of the four model architectures **A**, **B**, **C** and **V**, as well as a quick description file.  
Each file contains the following hardcoded parameters:  

**NO_GRAPH_FEATURES_ONE** - number of global features in first stage GN  
**NO_GRAPH_FEATURES_TWO** - number of global features in second stage GN  

**NO_EDGE_FEATURES_ONE** - number of edge features in first stage GN  
**NO_EDGE_FEATURES_TWO** - number of edge features in second stage GN  

**NO_NODE_FEATURES_ONE** - number of node features in first stage GN  
**NO_NODE_FEATURES_TWO=NO_GRAPH_FEATURES_ONE** - no. global features of first stage is == no. node features of second stage  

**ENCODING_NODE_1** - no. of units in final MLP layer that encodes node features in stage 1  
**ENCODING_EDGE_1** - no. of units in final MLP layer that encodes edge features in stage 1  
**ENCODING_EDGE_2=NO_EDGE_FEATURES_TWO** - no. of units in final MLP layer that encodes temperature and mol% information 

**HIDDEN_NODE_ONE,HIDDEN_NODE_TWO** - no. of hidden units per linear layer in node MLP stage 1/2  
**HIDDEN_EDGE_ONE,HIDDEN_EDGE_TWO** -   no. of hidden units per linear layer in edge MLP stage 1/2
**HIDDEN_GRAPH_ONE,HIDDEN_GRAPH_TWO** -   no. of hidden units per linear layer in global MLP stage 1/2

In each file, the Node, Edge and Global models for both the first and second stage graph network are defined, and then the graph neural network is constructed. The full graph net is a class with the name *GNN_FULL_CLASS*. You can import the architecture into another script by running (for example):

    from graph_modules.graph_net_V import GNN_FULL_CLASS

### misc_modules

There are four `.py` files in this directory: `data_loader.py`, `load_graph_data.py`, `plotter.py` and `train_and_evaluate.py`.  

`data_loader.py` is used to mini-batch data for parallel processing. This most likely needs to be modified, as the current version fails with numbers of datapoints that are not multiples of BATCH_SIZE

`load_graph_data.py` is used when graphs have been generated, and to load the graphs

`plotter.py` contains all plotting functions, that later plot errors, losses, and prediction results

`train_and_evaluate.py` contains functions for training and evaluation functions **NOTE:** Those are currently **not** used in the `main_train_script.py`, due to occuring errors. However they provide you a good starting point to write your own train functions.

## Datasets

As mentioned, data and the dataset are stored in `data`. The dataset to run `main_train_script.py` is `data\DB_order_parameters_POPC.csv`.

It is a comma-seperated .csv file with the columns:  
| Papertag | Molecule1 | Molecule2 | Membrane | molp1 | molp2 | molpm | Temperature | WTp | 2 | ... | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|  

where: 

**Papertag** - is the reference tag assigned to the paper the data was acquired for. This can be found in the supplementary information.  
**Molecule1** - TAG of the first molecule, **0** if no molecule was used  
**Molecule2** - TAG of the second molecule, **0** if no molecule was used  
**Membrane** - TAG of the membrane, here only **POPC**  
**molp1** - molecular ratio of first molecule (between 0 and 1)  
**molp2** - molecular ratio of second molecule (between 0 and 1)  
**molpm** - molecular ratio of membrane (between 0 and 1)  
**Temperature** - Temperature this measurement was recorded at in Kelvin  
**WTp** - water content of the sample in weight% (1-99), or NULL if unknown
**2...16** - order parameter value at methyl group carbon no. **n**

The information that belongs in the columns is the same as in the names of the files in `2H_data`. Additionally, the value of the order parameters from position 2 to 16 are in the columns 2 to 16.

## Requirements

The requirements to recreate the exact Conda environment as the researchers can be found in `requirements.txt`.  
You can recreate the environment by running:  

    conda create --name <envname> --file requirements.txt

The scripts were run on a **NVIDIA RTX 2080 Super** with **8GB** of memory. Scripts were written for **Python 3.8** and **CUDA 11.1**.  

With this hardware, a BATCH_SIZE of 4 to 8 could be achieved. If you run a different CUDA and Python version on your GPU, you most likely can't run the file as is.

You will by hand need to install the following modules for your configuration:  
- PyTorch
- PyTorch Geometric
- seaborn
- scikit-learn
- pandas
- matplotlib


# Running the script

## Pre-processing

Make sure, that you have the dataset `data\DB_order_parameters_POPC.csv`, and that all the molecules that are in there can also be found in `1_raw_mol2`.

Now, run `123_make_graphs.py`, this will rotate and translate the molecules stored in the `.mol2` files.

The resulting `.mol2` files are saved in `2_KOS_mol2`. From there, the graphs are generated from the molecules, and the resulting graphs are stored in `3_latent_graphs`.  

Each graph is stored in three files: `zzz_ea.pt`, `zzz_ei.pt` and `zzz_x.pt`, where the edge attributes, edge indices and node attributes of the graph are stored.


After that, you are done with pre-processing and can train your network.

## Set up your experiments

Edit `network_commands.txt` to add experiments you want to run. It has the following structure:  

**4_training_results**  
| CONFIG | MODELNO | EPOCHS | BATCH_SIZE | LR | NO_MP_ONE | NO_MP_TWO |
|---|---|---|---|---|---|---|
| 0 | B | 500 | 4 | 0.0001 | 2 | 2 |
| 1 | V | 500 | 4 | 0.0001 | 2 | 2 |

The first line of the file is the name of the directory, in which the results will be stored. In this case `4_training_results`.  

The following parameters have to be set:  
- **CONFIG** is the number of your experiment. It starts at 0.
- **MODELNO** selects the architecture. Select A,B,C or V
- **EPOCHS** select number of training epochs
- **BATCH_SIZE** size of batch. This many graphs will be handled simultaneously. 4 for 8GB GPU.
- **LR** learning rate for the AdamW optimizer
- **NO_MP_ONE** number of first stage message-passing steps
- **NO_MP_TWO** number of second stage message-passing steps

When the script finds the experiment in `4_training_results` (or the name of your folder) it skips performing the experiment.  

## Running your experiments

Now you can run `main_train_script.py` that will perform all of the experiments in `4_training_results`, unless their results are already in there.

It is currently hardcoded, that each experiment is performed **k=4** times for cross-validation.  

Sometimes, the script can crash or produce a NaN error, most likely due to weird GPU-HDD interactions. 

To keep everything running, run `residentsleeper.bat`. 

This will run `main_train_script.py` in perpetuity, restarting it after a crash. 

However, it also restarts the script once training has ended, and it closes automatically, so after training is finished it will continue in an endless loop of opening and closing windows. 

Then you can/have to kill it manually.

Good luck, don't die  
--V
