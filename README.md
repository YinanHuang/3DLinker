# 3DLinker: An E (3) Equivariant Variational Autoencoder for Molecular Linker Design
## About
This directory contains the code and resources of the following paper:
[_"3DLinker: An E(3) Equivariant Variational Autoencoder for Molecular Linker Design"_](https://arxiv.org/abs/2205.07309)
1. 3DLinker is a 3D graph variational auto-encoder that is equivariant to rigid transformations and reflections (E(3) group). It takes two molecular fragments as input and generates a "linker" (both with graphs and spatial coordinates) attaching these two fragments.
2. We thank the authors of [Deep generative models for 3D linker design](https://pubs.acs.org/doi/full/10.1021/acs.jcim.9b01120) for releasing their code. Our code is based on their source code release ([link](https://github.com/fimrie/DeLinker)).
3. Please feel free to contact Yinan Huang yinan8114@gmail.com if you have issue using the code.

## Overview of Model
We introduce 3DLinker, a variational auto-encoder, to address the simultaneous generation of graphs and spatial coordinates in molecular linker design. Our model leverages an important geometric inductive bias: equivariance w.r.t. E(3) transformations. See the concrete encoding and decoding (generation) process below.

![model](https://raw.githubusercontent.com/YinanHuang/3DLinker/master/3dlinker.png)

**Step 1.  Encode the fragments and ground-truth into equivariant node-level embeddings**

An equivariant GNN is applied to jointly embed the fragments and ground-truth into node-level embeddings, including both scalar-type and vector-type embeddings.  They are equivariant in the sense that scalars and vectors are 0-order and 1-order E(3) tensors respectively.

**Step2. Predict anchor nodes**

Use the node embeddings to predict the anchor nodes that the linker will attach the two fragments.

**Step3. Predict node type**

Use the node embeddings to predict the type of nodes in the linker.

**Step4. Predict edges and coordinates**

Following an auto-regressive policy, we sequentially predict the edges and coordinates of the selected node. The nodes are selected in a BFS manner. 

For more details, see Methodology of our paper. 

## Sub-directories
* \[generated_samples\] contains the generated molecules. Each generation will produce a .smi file (graphs info) and a .sdf file (coordinates info).  
* \[zinc\] contains preprocessed ZINC data. Tranining dataset is not included due to upload limit. See [Data](##Data) for downloading training dataset.
* \[check_points\] contains pytorch model checkpoints. "pretrained_model.pickle" is a provided checkpoint that can recover the experimental results in the paper.
* \[analysis\] contains evaluation code.
 
## Data
Only test dataset is included in this directory, which can be used for generation and evaluation. To train your own model, you can download the training dataset from [here](https://drive.google.com/drive/folders/1z4P_IDM5Zrc6Aju6qqwPvPQTd9lgZnXy).

## Code Usage
**Python Envirnoment**

The code is tested in Python 3.9 with Pytorch 1.11.

You can create a new conda environment using the provided yaml file:
`conda env create -f env.yml`

or manually install the following packages:
* Pytorch: install a proper version compatible with your platform (see [Pytorch versions](https://pytorch.org/get-started/previous-versions/))
* RDKit: `conda install -c rdkit rdkit`
* Docopt: `pip install docopt`
* Joblib: `pip install joblib`


**Generation**

To generate new molecules using pretrained model, run 
`python main.py --dataset zinc --config-file test_config.json --generation True --load_cpt ./check_points/pretrained_model.pickle`

The default setting is to generate 250 samples per test data, saved in directory "./generated_samples" as a smi file and a sdf file. The .smi file contains lines of fragments, ground-truth, generation. Look up "test_config.json" to see and modify the setting.  

**Evaluation**

To evaluate the generated molecules, enter the analysis directory and run
`python evaluate_generated_mols.py ZINC PATH_TO_GENERATED_MOLS ../zinc/smi_train.txt 1 True None ./wehi_pains.csv`

**Training**

To train your own model, first download trainingrun
`python main.py --dataset zinc --config-file train_config.json`

Change hyper-parameters like batch size in file train_config.json. More hyper-params can be found in main.py.
