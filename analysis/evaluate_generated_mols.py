#!/usr/bin/env python

# # Analysis script
# 
# Basic flow:
# - Load data
# - Check validity
# - Check 2D properties
# - Generate conformers
# - Check 3D similarity

# Imports

import numpy as np
import re

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolStandardize
from rdkit.Chem import rdMolAlign

from joblib import Parallel, delayed

# import rdkit_conf_parallel
# import calc_SC_RDKit
import frag_utils

import os, sys

# Setup

if len(sys.argv) != 8:
    print("Not provided all arguments")
    quit()

data_set = sys.argv[1] # Options: ZINC, CASF
gen_smi_file = sys.argv[2] # Path to generated molecules
train_set_path = sys.argv[3] # Path to training set
n_cores = int(sys.argv[4]) # Number of cores to use
verbose = bool(sys.argv[5]) # Output results
if sys.argv[6] == "None":
    restrict = None
else:
    restrict = int(sys.argv[6]) # Set to None if don't want to restrict
pains_smarts_loc = sys.argv[7] # Path to PAINS SMARTS

if verbose:
    print("##### Start Settings #####")
    print("Data set:", data_set)
    print("Generated smiles file:", gen_smi_file)
    print("Training set:", train_set_path)
    print("Number of cores:", n_cores)
    print("Verbose:", verbose)
    print("Restrict data:", restrict)
    print("PAINS SMARTS location:", pains_smarts_loc)
    print("#####  End Settings  #####")



# Prepare data

# Load molecules
# FORMAT: (Starting fragments (SMILES), Original molecule (SMILES), Generated molecule (SMILES))
generated_smiles = frag_utils.read_triples_file(gen_smi_file)

if restrict is not None and int(restrict) > 0:
        generated_smiles = generated_smiles[:restrict]

if verbose:
    print("Number of generated SMILES: %d" % len(generated_smiles))


in_mols = [smi[1] for smi in generated_smiles]
frag_mols = [smi[0] for smi in generated_smiles]
gen_mols = []
# drop invalid generations
for smi in generated_smiles:
    try:
        gen_mols.append(Chem.CanonSmiles(smi[2]))
    except:
        gen_mols.append("*")

# Remove dummy atoms from starting points
clean_frags = Parallel(n_jobs=n_cores)(delayed(frag_utils.remove_dummys)(smi) for smi in frag_mols)


# Check valid
results = []
val_idx = []
i = 0
for in_mol, frag_mol, gen_mol, clean_frag in zip(in_mols, frag_mols, gen_mols, clean_frags):
    if Chem.MolFromSmiles(gen_mol) == None:
        continue
        # gen_mols is chemically valid
    try:
        Chem.SanitizeMol(Chem.MolFromSmiles(gen_mol), sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    except:
        print('Chemical Invalid.')
        continue
    # gen_mols should contain both fragments
    if len(Chem.MolFromSmiles(gen_mol).GetSubstructMatch(Chem.MolFromSmiles(clean_frag))) == Chem.MolFromSmiles(
            clean_frag).GetNumAtoms():
        results.append([in_mol, frag_mol, gen_mol, clean_frag])
        val_idx.append(i)
    i += 1

if verbose:
    print("Number of generated SMILES: \t%d" % len(generated_smiles))
    print("Number of valid SMILES: \t%d" % len(results))
    print("%% Valid: \t\t\t%.2f%%" % (len(results)/len(generated_smiles)*100))


# Determine linkers of generated molecules
linkers = Parallel(n_jobs=n_cores)(delayed(frag_utils.get_linker)(Chem.MolFromSmiles(m[2]), Chem.MolFromSmiles(m[3]), m[1]) for m in results)

# Standardise linkers
for i, linker in enumerate(linkers):
    if linker == "":
        continue
    try:
        linker_canon = Chem.MolFromSmiles(re.sub('[0-9]+\*', '*', linker))
        Chem.rdmolops.RemoveStereochemistry(linker_canon)
        linkers[i] = MolStandardize.canonicalize_tautomer_smiles(Chem.MolToSmiles(linker_canon))
    except:
        continue
    
# Update results
for i in range(len(results)):
    results[i].append(linkers[i])


# Prepare training set database

# Load ZINC training set
linkers_train = []

with open(train_set_path, 'r') as f:
    for line in f:
        toks = line.strip().split(' ')
        linkers_train.append(toks[1])
        
if verbose:
    print("Number of training examples: %d" % len(linkers_train))


# Prepare unique set of linkers

# Remove stereochemistry
linkers_train_nostereo = []
for smi in list(set(linkers_train)):
    mol = Chem.MolFromSmiles(smi)
    Chem.RemoveStereochemistry(mol)
    linkers_train_nostereo.append(Chem.MolToSmiles(Chem.RemoveHs(mol)))
    
# Standardise / canonicalise training set linkers
linkers_train_nostereo = {smi.replace(':1', '').replace(':2', '') for smi in set(linkers_train_nostereo)}
linkers_train_canon = []
for smi in list(linkers_train_nostereo):
    linkers_train_canon.append(MolStandardize.canonicalize_tautomer_smiles(smi))

# Remove duplicates
linkers_train_canon_unique = list(set(linkers_train_canon))

if verbose:
    print("Number of unique linkers: %d" % len(linkers_train_canon_unique))

# 2D analysis

# Create dictionary of results
results_dict = {}
for res in results:
    if res[0]+'.'+res[1] in results_dict: # Unique identifier - starting fragments and original molecule
        results_dict[res[0]+'.'+res[1]].append(tuple(res))
    else:
        results_dict[res[0]+'.'+res[1]] = [tuple(res)]

# Check number of unique molecules
if verbose:
    print("Unique molecules: %.2f%%" % (frag_utils.unique(results_dict.values())*100))

# Check novelty of generated molecules
count_novel = 0
for res in results:
    if res[4] in linkers_train_canon_unique:
        continue
    else:
        count_novel +=1
        
if verbose:
    print("Novel linkers: %.2f%%" % (count_novel/len(results)*100))

# Check proportion recovered
# Create dictionary of results
results_dict_with_idx = {}
for i, res in enumerate(results):
    if res[0]+'.'+res[1] in results_dict_with_idx: # Unique identifier - starting fragments and original molecule
        results_dict_with_idx[res[0]+'.'+res[1]].append([res, val_idx[i]])
    else:
        results_dict_with_idx[res[0]+'.'+res[1]] = [[res, val_idx[i]]]
recovered, rec_idx = frag_utils.check_recovered_original_mol_with_idx(list(results_dict_with_idx.values()))
if verbose:
    print("Recovered: %.2f%%" % (sum(recovered)/len(results_dict.values())*100))

# Check if molecules pass 2D filters 
filters_2d = frag_utils.calc_filters_2d_dataset(results, pains_smarts_loc=pains_smarts_loc, n_cores=n_cores)

results_filt = []
for res, filt in zip(results, filters_2d):
    if filt[0] and filt[1] and filt[2]:
        results_filt.append(res)
        
if verbose:
    print("Pass all 2D filters: \t\t%.2f%%" % (len(results_filt)/len(results)*100))
    print("Valid and pass all 2D filters: \t%.2f%%" % (len(results_filt)/len(generated_smiles)*100))
    print("Pass synthetic accessibility (SA) filter: \t%.2f%%" % (len([f for f in filters_2d if f[0]])/len(filters_2d)*100))
    print("Pass ring aromaticity filter: \t\t\t%.2f%%" % (len([f for f in filters_2d if f[1]])/len(filters_2d)*100))
    print("Pass SA and ring filters: \t\t\t%.2f%%" % (len([f for f in filters_2d if f[0] and f[1]])/len(filters_2d)*100))
    print("Pass PAINS filters: \t\t\t\t%.2f%%" % (len([f for f in filters_2d if f[2]])/len(filters_2d)*100))


# estimate rmsd
from rdkit.Geometry import Point3D
def write_3d_pos(mol, pos):
    success = AllChem.Compute2DCoords(mol, 0)
    if success == -1:
        print('3D positions fail to write')
        exit(1)
    conf = mol.GetConformer()
    for i in range(conf.GetNumAtoms()):
        x, y, z = pos[0, i, 0], pos[0, i, 1], pos[0, i, 2]
        x, y, z = x.astype('double'), y.astype('double'), z.astype('double')
        conf.SetAtomPosition(i, Point3D(x, y, z))
    return mol


# get recovered generated mols
from copy import deepcopy
sdf_file = deepcopy(gen_smi_file[:-3]) + 'sdf'
gen_3d_mols = Chem.SDMolSupplier(sdf_file)
gen_3d_mols_rec = []
for i in rec_idx:
    gen_3d_mols_rec.append(gen_3d_mols[i])


import json
# load ground truth mols
val_data_path = '../zinc/molecules_zinc_test_canonical.json'
with open(val_data_path, 'r') as f:
    in_mols_val = json.load(f)
in_mols_dic = {}
for mol in in_mols_val:
    temp = Chem.MolFromSmiles(mol['smiles_out'])
    Chem.RemoveStereochemistry(temp)
    # G1 = topology_from_rdkit(temp)
    # G2 = topology_from_adj(mol['node_features_out'], mol['graph_out'])
    mol_sdf = Chem.MolFromSmiles(mol['smiles_out'])
    mol_sdf = write_3d_pos(mol_sdf, np.array(mol['positions_out']).reshape([1, len(mol['positions_out']), 3]))
    in_mols_dic[Chem.MolToSmiles(temp)] = mol_sdf

in_3d_mols = []
# find referenced ground-truth
for mol in gen_3d_mols_rec:
    smi = Chem.MolToSmiles(mol)
    temp = Chem.MolFromSmiles(smi)
    Chem.RemoveStereochemistry(temp)
    smi = Chem.MolToSmiles(temp)
    in_3d_mols.append(in_mols_dic[smi])

# compute rmsd
def find_exit(mol, num_frag):
    neighbors = []
    for atom_idx in range(num_frag, mol.GetNumAtoms()):
        N = mol.GetAtoms()[atom_idx].GetNeighbors()
        for n in N:
            if n.GetIdx() < num_frag:
                neighbors.append(n.GetIdx())
    # assert len(neighbors) == 2
    return neighbors

from networkx.algorithms import isomorphism
rmsd = []
mappings = []
index = []
exits = []
for i in range(len(gen_3d_mols_rec)):
    mol1 = gen_3d_mols_rec[i]
    mol2 = in_3d_mols[i]
    pos1 = mol1.GetConformer().GetPositions()
    pos2 = mol2.GetConformer().GetPositions()
    # check if they are 3d recovered
    G1 = frag_utils.topology_from_rdkit(mol1)
    G2 = frag_utils.topology_from_rdkit(mol2)
    GM = isomorphism.GraphMatcher(G1, G2)
    num_frag = Chem.MolFromSmiles(frag_mols[rec_idx[i]]).GetNumAtoms() - 2
    flag = GM.is_isomorphic()
    exits = find_exit(mol1, num_frag)
    if flag and len(exits) == 2 and flag: # check if isomorphic and if exit nodes are correctly aligned
        error = Chem.rdMolAlign.GetBestRMS(mol1, mol2)
        num_linker = mol2.GetNumAtoms() - Chem.MolFromSmiles(frag_mols[rec_idx[i]]).GetNumAtoms() + 2
        num_atoms = mol1.GetNumAtoms()
        error *= np.sqrt(num_atoms / num_linker) # only count rmsd on linker
        rmsd.append(error)

rmsd = np.array(rmsd)
print('Aveage RMSD is %f' % np.mean(np.array(rmsd)))