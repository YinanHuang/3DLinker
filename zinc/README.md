# Use your own dataset
We provide ZINC dataset as primary dataset. Follow the instructions below if you would like to use your own dataset instead.

\[Note\] Currently only atom types appeared in ZINC are allowed in any new dataset, such as Br, C, Cl, F, H, I, N, O, S. Besides, to get a good performance on a new dataset, 
it is typically better to train a new model since the data distribution may shift.

## Step 1: Prepare raw data
Suppose your raw data consists of pairs (fragment SMILES, molecule SMILES). First write your raw data into a txt file 
in the following format:

`fragments(SMILES) molecule(SMILES)`

For example, one line of the txt file could be

`COc1ccccc1[*:2].Fc1cccc([*:1])c1 COc1ccccc1CCC(=O)c1cccc(F)c1`

Then run the following to preprocess your raw data into triplet format (molecule, linker, fragment):

`python raw_preprocessing.py --data_path RAW_DATA_PATH --output_path SAVE_PATH --verbose`

This step also check if the atom types in your data are allowed (currently only atom types in ZINC are allowed).

## Step 2: Process to final dataset
Run the following to produce the final dataset for training/test:

`python prepare_data.py --data_path PREPROCESSED_DATA_PATH --dataset_name NAME_OF_YOUR_DATASET`

Here processed_data_path is the file obtained in step 1. This step will also compute 3D coordinates for molecules by RDKit.

After runing the script above, you will get a json file named "molecules_NAME_OF_YOUR_DATASET.json",
which can be used for training and test.  

## Step 3: Train/generation on your dataset
To train/test on your own dataset obtained in step 2, simply run the training/test script with the config file changed. Concretly, write the path to your new
dataset into values of key <train_file>/<valid_file> in the config json file "train_config.json"/"test_config.json". 