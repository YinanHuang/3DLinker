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

import rdkit_conf_parallel
import calc_SC_RDKit
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
gen_mols = [smi[2] for smi in generated_smiles]

# Remove dummy atoms from starting points
clean_frags = Parallel(n_jobs=n_cores)(delayed(frag_utils.remove_dummys)(smi) for smi in frag_mols)


# Check valid
results = []
for in_mol, frag_mol, gen_mol, clean_frag in zip(in_mols, frag_mols, gen_mols, clean_frags):
    if len(Chem.MolFromSmiles(gen_mol).GetSubstructMatch(Chem.MolFromSmiles(clean_frag)))>0:
        results.append([in_mol, frag_mol, gen_mol, clean_frag])

if verbose:
    print("Number of generated SMILES: \t%d" % len(generated_smiles))
    print("Number of valid SMILES: \t%d" % len(results))
    print("%% Valid: \t\t\t%.2f%%" % (len(results)/len(generated_smiles)*100))


# Determine linkers of generated molecules
linkers = Parallel(n_jobs=n_cores)(delayed(frag_utils.get_linker)(Chem.MolFromSmiles(m[2]), Chem.MolFromSmiles(m[3]), m[1])                                    for m in results)

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
recovered = frag_utils.check_recovered_original_mol(list(results_dict.values()))
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


