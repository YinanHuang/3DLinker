#!/usr/bin/env/python
"""
Usage:
    prepare_data.py [options]

Options:
    -h --help                Show this screen
    --data_path FILE         Path to data file containing fragments and reference molecules
    --dataset_name NAME      Name of dataset (for use in output file naming)
    --test_mode              To prepare the data for DeLinker in test mode
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from docopt import docopt
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import rdFMCS
import json
import numpy as np
from utils import bond_dict, dataset_info, need_kekulize, to_graph_mol, graph_to_adj_mat, compute_3d_coors, compute_3d_coors_multiple
import utils
from align_utils import align_mol_to_frags

dataset = 'zinc'

def read_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    num_lines = len(lines)
    data = []
    for i, line in enumerate(lines):
        toks = line.strip().split(' ')
        if len(toks) == 3:
            smi_frags, abs_dist, angle = toks
            smi_mol = smi_frags
            smi_linker = ''
        elif len(toks) == 5:
            smi_mol, smi_linker, smi_frags, abs_dist, angle = toks
        else:
            print("Incorrect input format. Please check the README for useage.")
            exit()
        data.append({'smi_mol': smi_mol, 'smi_linker': smi_linker, 
                     'smi_frags': smi_frags,
                     'abs_dist': [abs_dist,angle]})
        if i % 2000 == 0:
            print('Finished reading: %d / %d' % (i, num_lines), end='\r')
    print('Finished reading: %d / %d' % (num_lines, num_lines))
    return data

def preprocess(raw_data, dataset, name, test=False, sdf_file=None):
    print('Parsing smiles as graphs.')
    processed_data =[]
    total = len(raw_data)
    for i, (smi_mol, smi_frags, smi_link, abs_dist) in enumerate([(mol['smi_mol'], mol['smi_frags'], 
                                                                   mol['smi_linker'], mol['abs_dist']) for mol in raw_data]):
        if test:
            smi_mol = smi_frags
            smi_link = ''
        (mol_out, mol_in), nodes_to_keep, exit_points, re_idx = align_mol_to_frags(smi_mol, smi_link, smi_frags)
        if mol_out == []:
            continue
        nodes_in, edges_in = to_graph_mol(mol_in, dataset)
        nodes_out, edges_out = to_graph_mol(mol_out, dataset)
        if min(len(edges_in), len(edges_out)) <= 0:
            continue
        # generate 3d coordinates of mols
        pos_out, _ = compute_3d_coors_multiple(mol_out)
        pos_in = pos_out[:len(nodes_in)]
        pos_out, pos_in = pos_out.tolist(), pos_in.tolist()
        processed_data.append({
                'graph_in': edges_in,
                'graph_out': edges_out, 
                'node_features_in': nodes_in,
                'node_features_out': nodes_out, 
                'smiles_out': smi_mol,
                'smiles_in': smi_frags,
                'v_to_keep': nodes_to_keep,
                'exit_points': exit_points,
                'abs_dist': abs_dist,
                'positions_out': pos_out,
                'positions_in': pos_in
            })
        if i % 500 == 0:
            print('Processed: %d / %d' % (i, total), end='\r')
    print('Processed: %d / %d' % (total, total))
    print('Saving data')
    with open('molecules_%s.json' % name, 'w') as f:
        json.dump(processed_data, f)
    print('Length raw data: \t%d' % total)
    print('Length processed data: \t%d' % len(processed_data))
          

if __name__ == "__main__":
    # Parse args
    args = docopt(__doc__)
    if args.get('--data_path') and args.get('--dataset_name'):
        data_paths = [args.get('--data_path')]
        names = [args.get('--dataset_name')]
    else:
        # data_paths = ['data_zinc_final_train.txt', 'data_zinc_final_valid.txt', 'data_zinc_final_test.txt', 'data_zinc_final_test.txt', 'data_casf_final.txt']
        # names = ['zinc_train', 'zinc_valid', 'zinc_test', 'zinc_test_mode2', 'casf_test']
        data_paths = ['mols_test.txt']
        names = ['mols_test']
    test_mode = args.get('--test_mode')

    for data_path, name in zip(data_paths, names):
        print("Preparing: %d", name)
        raw_data = read_file(data_path)
        preprocess(raw_data, dataset, name, test_mode)
