#!/usr/bin/env/python
"""
Usage:
    main.py [options]

Options:
    -h --help                Show this screen
    --dataset NAME           Dataset name: zinc (or qm9, cep)
    --config-file FILE       Hyperparameter configuration file path (in JSON format)
    --config CONFIG          Hyperparameter configuration dictionary (in JSON format)
    --log_dir NAME           log dir name
    --data_dir NAME          data dir name
    --restore FILE           File to restore weights from.
    --freeze-graph-model     Freeze weights of graph model components
    --restrict_data INT      Limit data
    --generation BOOL        Train or test
    --load_cpt STR           Path to load checkpoint
"""

from docopt import docopt
from collections import defaultdict, deque
import numpy as np
import torch
from torch import nn
import sys, traceback
import json
import os
from rdkit.Chem import QED
from model import ChemModel
from utils import *
import pickle
import random
from rdkit import Chem
from copy import deepcopy
import os
import time
from data_augmentation import *


class Linker(ChemModel):
    def __init__(self, args):
        super().__init__(args)

    @classmethod
    def default_params(cls):
        params = dict(super().default_params())
        params.update({
            'task_sample_ratios': {},
            'use_edge_bias': True,  # whether use edge bias in gnn

            'clamp_gradient_norm': 0.25,
            'out_layer_dropout_keep_prob': 1.0,

            'tie_fwd_bkwd': True,
            'random_seed': 0,  # fixed for reproducibility

            'batch_size': 8,
            'prior_learning_rate': 0.05,
            'stop_criterion': 1,
            'num_epochs': 10,
            'epoch_to_generate': 10,
            'number_of_generation_per_valid': 250,
            'maximum_distance': 50,
            "use_argmax_generation": False,  # use random sampling or argmax during generation
            'residual_connection_on': True,  # whether residual connection is on
            'residual_connections': {  # For iteration i, specify list of layers whose output is added as an input
                2: [0],
                4: [0, 2],
                6: [0, 2, 4],
                8: [0, 2, 4, 6],
                10: [0, 2, 4, 6, 8],
                12: [0, 2, 4, 6, 8, 10],
                14: [0, 2, 4, 6, 8, 10, 12],
            },
            'num_timesteps': 5,  # gnn propagation step
            'hidden_size': 28,
            'idx_size': 4,
            'vector_size': 12,
            'max_num_nodes': 48,
            'encoding_size': 12,
            'encoding_vec_size': 12, 
            'kl_trade_off_lambda': 0.6,  # kl tradeoff
            'pos_trade_off_lambda': 1,
            'learning_rate': 0.0006,
            'graph_state_dropout_keep_prob': 1,
            'compensate_num': 0,  # how many atoms to be added during generation
            'try_different_starting': True,
            "num_different_starting": 1,

            'generation': False,  # only generate
            'use_graph': True,  # use gnn
            "label_one_hot": True,  # one hot label or not
            "multi_bfs_path": False,  # whether sample several BFS paths for each molecule
            "bfs_path_count": 30,
            "path_random_order": False,  # False: canonical order, True: random order
            "sample_transition": False,  # whether to use transition sampling
            'edge_weight_dropout_keep_prob': 1,
            'check_overlap_edge': False,
            "truncate_distance": 10,
            "output_name": '',
            "check_point_path": 'check_points',
            'if_save_check_point': True,
            'save_params_file': False,
            'use_cuda': True,
            'decoder': 'VGNN',
            'accumulation_steps': 1,
            'num_rbf': 12,
            'max_correlation_length': 5, 
            'lr_decay': 1,
            'if_generate_pos': True,
            'if_generate_exit': True, 
            'noise_free': False, 
            'prior_variance': 1, 
            'if_update_pos': True,
            'reverse_augmentation': False,
        })

        return params

    # Data preprocessing and chunking into minibatches:
    def process_raw_graphs(self, raw_data, is_training_data, file_name, bucket_sizes=None):
        if bucket_sizes is None:
            bucket_sizes = dataset_info(self.params["dataset"])["bucket_sizes"]
        # incremental_results, raw_data = self.calculate_incremental_results(raw_data, bucket_sizes, file_name,
        #                                                                   is_training_data)
        incremental_results, raw_data = self.calculate_incremental_results(raw_data, bucket_sizes, file_name,
                                                                           is_training_data)
        bucketed = defaultdict(list)
        x_dim = len(raw_data[0]["node_features_out"][0])

        for d, incremental_result_1 in zip(raw_data, incremental_results[1]):
            # choose a bucket
            chosen_bucket_idx = np.argmax(bucket_sizes > max(max([v for e in d['graph_in'] for v in [e[0], e[2]]]),
                                                             max([v for e in d['graph_out'] for v in [e[0], e[2]]])))
            chosen_bucket_size = bucket_sizes[chosen_bucket_idx]
            # total number of nodes in this data point out
            n_active_nodes_in = len(d["node_features_in"])
            n_active_nodes_out = len(d["node_features_out"])
            bucketed[chosen_bucket_idx].append({
                #'adj_mat_in': graph_to_adj_mat(d['graph_in'], chosen_bucket_size, self.num_edge_types,
                #                               self.params['tie_fwd_bkwd']),
                #'adj_mat_out': graph_to_adj_mat(d['graph_out'], chosen_bucket_size, self.num_edge_types,
                #                                self.params['tie_fwd_bkwd']),
                'adj_mat_in': deepcopy(d['graph_in']),
                'adj_mat_out': deepcopy(d['graph_out']),
                'v_to_keep': node_keep_to_dense(d['v_to_keep'], chosen_bucket_size),
                'exit_points': d['exit_points'],
                'abs_dist': d['abs_dist'],
                'it_num': 0,
                'incre_adj_mat_out': incremental_result_1[0],
                'distance_to_others_out': incremental_result_1[1],
                'overlapped_edge_features_out': incremental_result_1[8],
                'node_sequence_out': incremental_result_1[2],
                'edge_type_masks_out': incremental_result_1[3],
                'edge_type_labels_out': incremental_result_1[4],
                'edge_masks_out': incremental_result_1[6],
                'edge_labels_out': incremental_result_1[7],
                'local_stop_out': incremental_result_1[5],
                'number_iteration_out': len(incremental_result_1[5]),
                'init_in': d["node_features_in"] + [[0 for _ in range(x_dim)] for __ in
                                                    range(chosen_bucket_size - n_active_nodes_in)],
                'init_out': d["node_features_out"] + [[0 for _ in range(x_dim)] for __ in
                                                      range(chosen_bucket_size - n_active_nodes_out)],
                'mask_in': [1. for _ in range(n_active_nodes_in)] + [0. for _ in
                                                                     range(chosen_bucket_size - n_active_nodes_in)],
                'mask_out': [1. for _ in range(n_active_nodes_out)] + [0. for _ in
                                                                       range(chosen_bucket_size - n_active_nodes_out)],
                'smiles_in': d['smiles_in'],
                'smiles_out': d['smiles_out'],
                'positions_out': positions_padding(d['positions_out'], chosen_bucket_size).astype("float16"),
                'positions_in': positions_padding(d['positions_in'], chosen_bucket_size).astype("float16")

            })

        if is_training_data:
            for (bucket_idx, bucket) in bucketed.items():
                np.random.shuffle(bucket)

        bucket_at_step = [[bucket_idx for _ in range(len(bucket_data) // self.params['batch_size'])]
                          for bucket_idx, bucket_data in bucketed.items()]
        bucket_at_step = [x for y in bucket_at_step for x in y]

        return bucketed, bucket_sizes, bucket_at_step


    def calculate_incremental_results(self, raw_data, bucket_sizes, file_name, is_training_data):
        incremental_results = [[], []]
        # Copy the raw_data if more than 1 BFS path is added
        # new_raw_data = []
        for idx, d in enumerate(raw_data):
            out_direc = "out"
            res_idx = 1

            # Use canonical order or random order here. canonical order starts from index 0. random order starts from random nodes
            if not self.params["path_random_order"]:
                # Use several different starting index if using multi BFS path
                if self.params["multi_bfs_path"]:
                    list_of_starting_idx = list(range(self.params["bfs_path_count"]))
                else:
                    list_of_starting_idx = [0]  # the index 0
            else:
                # Get the node length for this output molecule
                node_length = len(d["node_features_" + out_direc])
                if self.params["multi_bfs_path"]:
                    list_of_starting_idx = np.random.choice(node_length, self.params["bfs_path_count"],
                                                            replace=True)  # randomly choose several
                else:
                    list_of_starting_idx = [random.choice(list(range(node_length)))]  # randomly choose one
            for list_idx, starting_idx in enumerate(list_of_starting_idx):
                # Choose a bucket
                chosen_bucket_idx = np.argmax(bucket_sizes > max(max([v for e in d['graph_out']
                                                                      for v in [e[0], e[2]]]),
                                                                 max([v for e in d['graph_in']
                                                                      for v in [e[0], e[2]]])))
                chosen_bucket_size = bucket_sizes[chosen_bucket_idx]

                nodes_no_master = d['node_features_' + out_direc]
                edges_no_master = d['graph_' + out_direc]
                incremental_adj_mat, distance_to_others, node_sequence, edge_type_masks, edge_type_labels, local_stop, edge_masks, edge_labels, overlapped_edge_features = \
                    construct_incremental_graph_preselected(self.params['dataset'], edges_no_master, chosen_bucket_size,
                                                            len(nodes_no_master), d['v_to_keep'], d['exit_points'],
                                                            nodes_no_master, self.params, is_training_data,
                                                            initial_idx=starting_idx)
                if self.params["sample_transition"] and list_idx > 0:
                    incremental_results[res_idx][-1] = [x + y for x, y in zip(incremental_results[res_idx][-1],
                                                                              [incremental_adj_mat, distance_to_others,
                                                                               node_sequence, edge_type_masks,
                                                                               edge_type_labels, local_stop, edge_masks,
                                                                               edge_labels, overlapped_edge_features])]
                else:
                    incremental_results[res_idx].append(
                        [incremental_adj_mat, distance_to_others, node_sequence, edge_type_masks,
                         edge_type_labels, local_stop, edge_masks, edge_labels, overlapped_edge_features])
                    if self.params['reverse_augmentation']:
                        incremental_adj_mat, distance_to_others, node_sequence, edge_type_masks, edge_type_labels, local_stop, edge_masks, edge_labels, overlapped_edge_features = \
                            construct_incremental_graph_preselected(self.params['dataset'], edges_no_master,
                                                                    chosen_bucket_size,
                                                                    len(nodes_no_master), d['v_to_keep'],
                                                                    d['exit_points'],
                                                                    nodes_no_master, self.params, is_training_data,
                                                                    initial_idx=starting_idx, reverse=True)
                        incremental_results[res_idx].append(
                            [incremental_adj_mat, distance_to_others, node_sequence, edge_type_masks,
                             edge_type_labels, local_stop, edge_masks, edge_labels, overlapped_edge_features])
                    # Copy the raw_data here
                    # new_raw_data.append(d)
            # Progress
            if idx % 100 == 0:
                print('\r'+'finish calculating %d incremental matrices' % idx, end='')
        print('\n')

        if self.params['reverse_augmentation']:
            raw_data_aug = []
            for data in raw_data:
                data_reverse = data.copy()
                data_reverse['exit_points'] = sorted(data_reverse['exit_points'], reverse=True)
                raw_data_aug.extend((data, data_reverse))
        else:
            raw_data_aug = raw_data
        return incremental_results, raw_data_aug # , new_raw_data
    def make_minibatch_iterator(self, data, is_training):
        (bucketed, bucket_sizes, bucket_at_step) = data
        batch_dataset = []
        if is_training:
            np.random.shuffle(bucket_at_step)
            for _, bucketed_data in bucketed.items():
                np.random.shuffle(bucketed_data)
        bucket_counters = defaultdict(int)
        for step in range(len(bucket_at_step)):
            bucket = bucket_at_step[step]
            start_idx = bucket_counters[bucket] * self.params['batch_size']
            end_idx = (bucket_counters[bucket] + 1) * self.params['batch_size']
            elements = bucketed[bucket][start_idx:end_idx]
            batch_data = self.make_batch(elements, bucket_sizes[bucket])
            batch_dataset.append(batch_data)
            bucket_counters[bucket] += 1
        return batch_dataset

    def make_batch(self, elements, maximum_vertice_num):
        # get maximum number of iterations in this batch. used to control while_loop
        max_iteration_num = -1
        for d in elements:
            max_iteration_num = max(d['number_iteration_out'], max_iteration_num)
        batch_data = {'adj_mat_in': [], 'adj_mat_out': [], 'v_to_keep': [], 'exit_points': [], 'abs_dist': [],
                      'it_num': [], 'init_in': [], 'init_out': [],
                      'edge_type_masks_out': [], 'edge_type_labels_out': [], 'edge_masks_out': [],
                      'edge_labels_out': [],
                      'node_mask_in': [], 'node_mask_out': [], 'task_masks': [], 'node_sequence_out': [],
                      'iteration_mask_out': [], 'local_stop_out': [], 'incre_adj_mat_out': [],
                      'distance_to_others_out': [], 'max_iteration_num': max_iteration_num,
                      'overlapped_edge_features_out': [], 'positions_out': [], 'positions_in': [], 'smiles_in': [], 'smiles_out': []}

        for d in elements:
            batch_data['adj_mat_in'].append(d['adj_mat_in'])
            batch_data['adj_mat_out'].append(d['adj_mat_out'])
            # batch_data['adj_mat_in'].append(graph_to_adj_mat(d['adj_mat_in'], len(d['init_in']) , self.num_edge_types,
            #                                   self.params['tie_fwd_bkwd']))
            # batch_data['adj_mat_out'].append(graph_to_adj_mat(d['adj_mat_out'], len(d['init_in']), self.num_edge_types,
            #                                                 self.params['tie_fwd_bkwd']))
            batch_data['v_to_keep'].append(node_keep_to_dense(d['v_to_keep'], maximum_vertice_num))
            batch_data['exit_points'].append(d['exit_points'])
            batch_data['abs_dist'].append(d['abs_dist'])
            batch_data['it_num'] = [0]
            batch_data['init_in'].append(d['init_in'])
            batch_data['init_out'].append(d['init_out'])
            batch_data['node_mask_in'].append(d['mask_in'])
            batch_data['node_mask_out'].append(d['mask_out'])
            batch_data['positions_in'].append(d['positions_in'])
            batch_data['positions_out'].append(d['positions_out'])
            batch_data['smiles_in'].append(d['smiles_in'])
            batch_data['smiles_out'].append(d['smiles_out'])

            for direc in ['_out']:
                # sparse to dense for saving memory
                # incre_adj_mat = incre_adj_mat_to_dense(d['incre_adj_mat' + direc], self.num_edge_types,
                 #                                      maximum_vertice_num)
                incre_adj_mat = adj_list_padding(d['incre_adj_mat'+direc], max_iteration_num, d['number_iteration'+direc])
                # incre_adj_mat = d['incre_adj_mat'+direc]
                distance_to_others = distance_to_others_dense(d['distance_to_others' + direc], maximum_vertice_num)
                overlapped_edge_features = overlapped_edge_features_to_dense(d['overlapped_edge_features' + direc],
                                                                             maximum_vertice_num)
                node_sequence = node_sequence_to_dense(d['node_sequence' + direc], maximum_vertice_num)
                edge_type_masks = edge_type_masks_to_dense(d['edge_type_masks' + direc], maximum_vertice_num,
                                                           self.num_edge_types)
                edge_type_labels = edge_type_labels_to_dense(d['edge_type_labels' + direc], maximum_vertice_num,
                                                             self.num_edge_types)
                edge_masks = edge_masks_to_dense(d['edge_masks' + direc], maximum_vertice_num)
                edge_labels = edge_labels_to_dense(d['edge_labels' + direc], maximum_vertice_num)


                # batch_data['incre_adj_mat' + direc].append(incre_adj_mat +
                 #                                          [np.zeros((self.num_edge_types, maximum_vertice_num,
                 #                                                     maximum_vertice_num))
                 #                                           for _ in
                 #                                          range(max_iteration_num - d['number_iteration' + direc])])
                batch_data['incre_adj_mat' + direc].append(incre_adj_mat)
                batch_data['distance_to_others' + direc].append(distance_to_others +
                                                                [np.zeros((maximum_vertice_num))
                                                                 for _ in range(
                                                                    max_iteration_num - d['number_iteration' + direc])])
                batch_data['overlapped_edge_features' + direc].append(overlapped_edge_features +
                                                                      [np.zeros((maximum_vertice_num))
                                                                       for _ in range(max_iteration_num - d[
                                                                          'number_iteration' + direc])])
                batch_data['node_sequence' + direc].append(node_sequence +
                                                           [np.zeros((maximum_vertice_num))
                                                            for _ in
                                                            range(max_iteration_num - d['number_iteration' + direc])])
                batch_data['edge_type_masks' + direc].append(edge_type_masks +
                                                             [np.zeros((self.num_edge_types, maximum_vertice_num))
                                                              for _ in
                                                              range(max_iteration_num - d['number_iteration' + direc])])
                batch_data['edge_masks' + direc].append(edge_masks +
                                                        [np.zeros((maximum_vertice_num))
                                                         for _ in
                                                         range(max_iteration_num - d['number_iteration' + direc])])
                batch_data['edge_type_labels' + direc].append(edge_type_labels +
                                                              [np.zeros((self.num_edge_types, maximum_vertice_num))
                                                               for _ in range(
                                                                  max_iteration_num - d['number_iteration' + direc])])
                batch_data['edge_labels' + direc].append(edge_labels +
                                                         [np.zeros((maximum_vertice_num))
                                                          for _ in
                                                          range(max_iteration_num - d['number_iteration' + direc])])
                batch_data['iteration_mask' + direc].append([1 for _ in range(d['number_iteration' + direc])] +
                                                            [0 for _ in
                                                             range(max_iteration_num - d['number_iteration' + direc])])
                batch_data['local_stop' + direc].append([int(s) for s in d['local_stop' + direc]] +
                                                        [0 for _ in
                                                         range(max_iteration_num - d['number_iteration' + direc])])

        return batch_data


    def load_current_batch_as_tensor(self, batch_data):
        num_nodes = len(batch_data['init_in'][0])
        num_graphs = len(batch_data['init_in'])
        # initial_representations_in = batch_data['init_in']
        initial_representations_in = torch.tensor(self.pad_annotations(batch_data['init_in']))
        # initial_representations_out = batch_data['init_out']
        initial_representations_out = torch.tensor(self.pad_annotations(batch_data['init_out']))
        self.data['initial_node_representation_in'] = initial_representations_in
        self.data['initial_node_representation_out'] = initial_representations_out
        self.data['node_symbols_out'] = torch.tensor(batch_data['init_out'], dtype=torch.float32)
        self.data['node_symbols_in'] = torch.tensor(batch_data['init_in'], dtype=torch.float32)
        self.data['node_mask_in'] = torch.tensor(batch_data['node_mask_in'])
        self.data['node_mask_out'] = torch.tensor(batch_data['node_mask_out'])
        self.ops['graph_state_mask_in'] = torch.unsqueeze(self.data['node_mask_in'], 2)
        self.ops['graph_state_mask_out'] = torch.unsqueeze(self.data['node_mask_out'], 2)
        self.data['num_graphs'] = torch.tensor(num_graphs)
        self.data['adjacency_matrix_in'] = torch.tensor(np.array([graph_to_adj_mat(batch_data['adj_mat_in'][i], num_nodes , self.num_edge_types,
                                               self.params['tie_fwd_bkwd']) for i in range(num_graphs)]), dtype=torch.float32)
        self.data['adjacency_matrix_out'] = torch.tensor(
            np.array([graph_to_adj_mat(batch_data['adj_mat_out'][i], num_nodes, self.num_edge_types,
                                      self.params['tie_fwd_bkwd']) for i in range(num_graphs)]), dtype=torch.float32)
        # self.data['adjacency_matrix_in'] = torch.tensor(np.array(batch_data['adj_mat_in']), dtype=torch.float32)
        # self.data['adjacency_matrix_out'] = torch.tensor(np.array(batch_data['adj_mat_out']), dtype=torch.float32)
        self.data['num_vertices'] = torch.tensor(self.data['adjacency_matrix_in'].size(2))
        self.data['z_prior_h'] = (0 if self.params['noise_free'] else 1) * torch.normal(0, 1, [self.params['batch_size'], self.data['num_vertices'], self.params['encoding_size']])
        self.data['z_prior_h_in'] = (0 if self.params['noise_free'] else 1) * torch.normal(0, 1, [self.params['batch_size'], self.data['num_vertices'],
                                                            self.params['encoding_size']])
        self.data['z_prior_v_in'] = (0 if self.params['noise_free'] else 1) * torch.normal(0, 1, [self.params['batch_size'],
                                                              self.data['num_vertices'], self.params['encoding_vec_size'], 3])
        self.data['z_prior_v'] = (0 if self.params['noise_free'] else 1) * torch.normal(0, 1, [self.params['batch_size'],
                                                              self.data['num_vertices'], self.params['encoding_vec_size'], 3])
        # self.data['z_prior_h'] = torch.normal(0, 1,[self.params['batch_size'], self.data['num_vertices'],
        #                                                                               self.params['encoding_size']])
        # self.data['z_prior_h_in'] = torch.normal(0, 1, [self.params['batch_size'], self.data['num_vertices'], self.params['encoding_size']])
        # self.data['z_prior_v_in'] = torch.normal(0, 1, [self.params['batch_size'], self.data['num_vertices'], self.params['encoding_size'], 3])

        # self.data['pos_noise'] = torch.normal(0, 1, [self.params['batch_size'], self.data['num_vertices'], 3])
        self.data['is_generative'] = False
        self.data['iteration_mask_out'] = torch.tensor(batch_data['iteration_mask_out'])
        self.data['max_iteration_num'] = torch.tensor(batch_data['max_iteration_num'])
        self.data['latent_node_symbols_in'] = initial_representations_in
        self.data['latent_node_symbols_out'] = initial_representations_out
        incre_adj_mat_out = [incre_adj_list_to_adj_mat(self.data['adjacency_matrix_in'][i].numpy(), batch_data['incre_adj_mat_out'][i],
                                                       self.num_edge_types) for i in range(num_graphs)]
        # self.data['incre_adj_mat_out'] = torch.tensor(np.array(batch_data['incre_adj_mat_out']), dtype=torch.float32)
        self.data['incre_adj_mat_out'] = torch.tensor(np.array(incre_adj_mat_out), dtype=torch.float32)
        self.data['distance_to_others_out'] = torch.tensor(np.array(batch_data['distance_to_others_out']), dtype=torch.int32)
        self.data['overlapped_edge_features_out'] = torch.tensor(np.array(batch_data['overlapped_edge_features_out']), dtype=torch.int32)
        self.data['node_sequence_out'] = torch.tensor(np.array(batch_data['node_sequence_out']), dtype=torch.float32)
        self.data['edge_type_masks_out'] = torch.tensor(np.array(batch_data['edge_type_masks_out']), dtype=torch.float32)
        self.data['edge_type_labels_out'] = torch.tensor(np.array(batch_data['edge_type_labels_out']), dtype=torch.float32)
        self.data['edge_masks_out'] = torch.tensor(np.array(batch_data['edge_masks_out']), dtype=torch.float32)
        self.data['edge_labels_out'] = torch.tensor(np.array(batch_data['edge_labels_out']), dtype=torch.float32)
        self.data['local_stop_out'] = torch.tensor(batch_data['local_stop_out'], dtype=torch.float32)
        self.data['abs_dist'] = torch.tensor(str2float(batch_data['abs_dist']), dtype=torch.float32)
        self.data['it_num'] = torch.tensor(batch_data['it_num'], dtype=torch.int32)
        self.data['positions_out'] = torch.tensor(np.array(batch_data['positions_out']), dtype=torch.float32)
        self.data['positions_in'] = torch.tensor(np.array(batch_data['positions_in']), dtype=torch.float32)
        self.data['exit_points'] = torch.tensor(np.array(batch_data['exit_points']), dtype=torch.float32)
        # transfer to designated device
        for key in self.data.keys():
            if torch.is_tensor(self.data[key]):
                self.data[key] = self.data[key].to(self.device)
        self.ops['graph_state_mask_in'] = self.ops['graph_state_mask_in'].to(self.device)
        self.ops['graph_state_mask_out'] = self.ops['graph_state_mask_out'].to(self.device)


    def make_model(self):
        node_dim = self.params['hidden_size']
        out_dim = self.params['encoding_size']
        out_vec_dim = self.params['encoding_vec_size']
        idx_dim = self.params['idx_size']
        h_dim = node_dim + idx_dim
        expanded_h_dim = h_dim + node_dim + 1 # plus a node type embedding and a focus bit
        v_dim = self.params['vector_size']

        # Dict for all pure weights
        self.weights = nn.ParameterDict()
        # Dict for all submodules/units
        self.units = nn.ModuleDict()

        # weights for embedding
        self.units['node_embedding'] = nn.Embedding(self.params['num_symbols'], node_dim)
        self.units['distance_embedding_in'] = nn.Embedding(self.params['maximum_distance'], expanded_h_dim)
        self.units['overlapped_edge_weight_in'] = nn.Embedding(2, expanded_h_dim)
        self.units['idx_embedding'] = nn.Embedding(self.params['max_num_nodes'], idx_dim)
        # weights for GNN of encoder and decoder
        for scope in ['_encoder', '_decoder']:
            if scope == '_encoder':
                new_h_dim = h_dim
            elif scope == '_decoder':
                new_h_dim = expanded_h_dim
            for iter_idx in range(self.params['num_timesteps']):
                weights_suffix = scope + str(iter_idx)
                # input size
                input_size = new_h_dim * (1 + length(self.params['residual_connections'].get(iter_idx)))  
                input_vec_size = v_dim * (1 + length(self.params['residual_connections'].get(iter_idx)))
                
                # pairwise size for message computation
                pair_h_dim = new_h_dim + 1
                # weights for edge convolution (message computation)
                self.weights['wave_num'+weights_suffix] = nn.Parameter(torch.rand(1))
                self.weights['weights_rbf_0'+weights_suffix] = nn.Parameter(glorot_init([self.params['num_rbf'], new_h_dim]))
                self.weights['weights_rbf_1' + weights_suffix] = nn.Parameter(
                        glorot_init([self.params['num_rbf'], v_dim]))
                self.weights['weights_rbf_2' + weights_suffix] = nn.Parameter(
                        glorot_init([self.params['num_rbf'], v_dim]))
                self.weights['biases_rbf_0'+weights_suffix] = nn.Parameter(torch.zeros([1, new_h_dim], dtype=torch.float32))
                self.weights['biases_rbf_1'+weights_suffix] = nn.Parameter(torch.zeros([1, v_dim], dtype=torch.float32))
                self.weights['biases_rbf_2'+weights_suffix] = nn.Parameter(torch.zeros([1, v_dim], dtype=torch.float32))
                self.weights['combine_h_v_weight'+weights_suffix] = nn.Parameter(glorot_init([v_dim, v_dim]))
                for edge_type in range(self.num_edge_types):
                    #self.weights['edge_s_hidden_weights'+weights_suffix] = nn.Parameter(glorot_init([
                    #    self.num_edge_types, new_h_dim+v_dim, new_h_dim+v_dim]))
                    self.weights['edge_s_biases'+weights_suffix] = nn.Parameter(torch.zeros([self.num_edge_types, 1,
                                                                                             new_h_dim], dtype=torch.float32))
                    self.weights['edge_s_output_weights'+weights_suffix] = nn.Parameter(glorot_init([self.num_edge_types,
                                                                                                     new_h_dim+v_dim, new_h_dim]))
                    #self.weights['edge_v0_hidden_weights' + weights_suffix] = nn.Parameter(glorot_init([
                    #    self.num_edge_types, new_h_dim, v_dim]))
                    self.weights['edge_v0_biases' + weights_suffix] = nn.Parameter(torch.zeros([self.num_edge_types, 1,
                                                                                               v_dim],
                                                                                              dtype=torch.float32))
                    self.weights['edge_v0_output_weights' + weights_suffix] = nn.Parameter(
                            glorot_init([self.num_edge_types,
                                     new_h_dim, v_dim]))
                     #self.weights['edge_v1_hidden_weights' + weights_suffix] = nn.Parameter(glorot_init([
                    #        self.num_edge_types, new_h_dim, v_dim]))
                    self.weights['edge_v1_biases' + weights_suffix] = nn.Parameter(torch.zeros([self.num_edge_types, 1,
                                                                                                v_dim],
                                                                                               dtype=torch.float32))
                    self.weights['edge_v1_output_weights' + weights_suffix] = nn.Parameter(
                            glorot_init([self.num_edge_types,
                                     new_h_dim, v_dim]))
                    #self.weights['edge_v_hidden_weights'+weights_suffix] = nn.Parameter(glorot_init([self.num_edge_types,
                    #                                                                                 v_dim, v_dim]))
                    self.weights['edge_v_nonlinear_Q'+weights_suffix] = nn.Parameter(glorot_init([self.num_edge_types,
                                                                                                  v_dim, v_dim]))
                    self.weights['edge_v_nonlinear_K'+weights_suffix] = nn.Parameter(glorot_init([self.num_edge_types,
                                                                                                    v_dim, v_dim]))
                    #self.weights['edge_v_output_weights'+weights_suffix] = nn.Parameter(glorot_init([self.num_edge_types,
                    #                                                                          v_dim, v_dim]))
                # weights for aggregation
                # self.weights['aggregate_v_hidden'+weights_suffix] = nn.Parameter(glorot_init([2 * v_dim, 2 * v_dim]))
                self.weights['aggregate_Q_weights' + weights_suffix] = nn.Parameter(glorot_init([input_vec_size + v_dim, input_vec_size + v_dim]))
                self.weights['aggregate_K_weights' + weights_suffix] = nn.Parameter(glorot_init([input_vec_size + v_dim, input_vec_size + v_dim]))
                self.weights['aggregate_v_output'+weights_suffix] = nn.Parameter(glorot_init([input_vec_size + v_dim, v_dim]))
                # messages and node representation combination function
                self.units['node_gru' + weights_suffix] = torch.nn.GRUCell(input_size, new_h_dim)
                # self.units['node_vec_gru' + weights_suffix] = GRUCell_vec(v_dim, input_vec_size)


        # weights for computing mean and log variance
        self.weights['mean_h_weights_out'] = nn.Parameter(glorot_init([h_dim, out_dim]))
        self.weights['mean_h_biases_out'] = nn.Parameter(torch.zeros([1, out_dim]))
        self.weights['variance_h_weights_out'] = nn.Parameter(glorot_init([h_dim, out_dim]))
        self.weights['variance_h_biases_out'] = nn.Parameter(torch.zeros([1, out_dim]))
        self.weights['mean_h_weights'] = nn.Parameter(glorot_init([h_dim, h_dim]))
        self.weights['mean_h_biases'] = nn.Parameter(torch.zeros([1, h_dim]))
        # self.weights['variance_h_weights'] = nn.Parameter(glorot_init([h_dim, h_dim]))
        # self.weights['variance_h_biases'] = nn.Parameter(torch.zeros([1, h_dim]))
        self.weights['mean_h_hidden_out_all'] = nn.Parameter(glorot_init([h_dim, h_dim]))
        self.weights['mean_h_weights_out_all'] = nn.Parameter(glorot_init([h_dim, out_dim]))
        self.weights['mean_h_biases_out_all'] = nn.Parameter(torch.zeros([1, h_dim]))
        self.weights['variance_h_weights_out_all'] = nn.Parameter(glorot_init([h_dim, out_dim]))
        self.weights['variance_h_hidden_out_all'] = nn.Parameter(glorot_init([h_dim, h_dim]))
        self.weights['variance_h_biases_out_all'] = nn.Parameter(torch.zeros([1, h_dim]))
        self.weights['mean_v_weights_out'] = nn.Parameter(glorot_init([v_dim, out_vec_dim]))
        self.weights['mean_v_weights_out_all'] = nn.Parameter(glorot_init([v_dim, out_vec_dim]))
        #self.weights['mean_v_biases_out'] = nn.Parameter(glorot_init([1, v_dim]))
        self.weights['variance_v_weights_out'] = nn.Parameter(glorot_init([h_dim, out_vec_dim]))
        self.weights['variance_v_biases_out'] = nn.Parameter(torch.zeros([1, out_vec_dim]))
        #self.weights['mean_v_biases_out'] = nn.Parameter(glorot_init([1, v_dim]))
        self.weights['variance_v_weights_out_all'] = nn.Parameter(glorot_init([h_dim, out_vec_dim]))
        self.weights['variance_v_biases_out_all'] = nn.Parameter(torch.zeros([1, out_vec_dim]))
        self.weights['mean_v_weights_in'] = nn.Parameter(glorot_init([v_dim, v_dim]))
        ##self.weights['mean_v_biases_in'] = nn.Parameter(glorot_init([1, v_dim]))

        # weights for combining (sampling) means and log variances
        self.weights['mean_h_combine_weights_in'] = nn.Parameter(glorot_init([out_dim, h_dim]))
        self.weights['mean_h_all_combine_weights_in'] = nn.Parameter(glorot_init([out_dim, h_dim]))
        self.weights['mean_v_all_combine_weights_in'] = nn.Parameter(glorot_init([out_vec_dim, v_dim]))
        self.weights['mean_v_combine_weights_in'] = nn.Parameter(glorot_init([out_vec_dim, v_dim]))
        self.weights['atten_h_weights_c_in'] = nn.Parameter(glorot_init([h_dim, h_dim]))
        self.weights['atten_h_weights_y_in'] = nn.Parameter(glorot_init([h_dim, h_dim]))
        self.weights['atten_v_weights_querys'] = nn.Parameter(glorot_init([v_dim, v_dim]))
        self.weights['atten_v_weights_keys'] = nn.Parameter(glorot_init([v_dim, v_dim]))

        # weights for exit points
        if self.params['if_generate_exit']:
            d_dim = 0
            self.weights['exit_points_vec_weights'] = nn.Parameter(glorot_init([v_dim, v_dim]))
            self.weights['exit_points_hidden_weights'] = nn.Parameter(glorot_init([v_dim + h_dim + d_dim, v_dim + h_dim + d_dim]))
            self.weights['exit_points_biases'] = nn.Parameter(torch.zeros([1, v_dim + h_dim + d_dim], dtype=torch.float32))
            self.weights['exit_points_output_weights'] = nn.Parameter(glorot_init([v_dim + h_dim + d_dim, 1]))
            self.weights['exit_points_conditional_hidden_weights'] = nn.Parameter(
                glorot_init([2 * (v_dim + h_dim), 2 * (v_dim + h_dim)]))
            self.weights['exit_points_conditional_biases'] = nn.Parameter(
                torch.zeros([1, 2 * (v_dim + h_dim)], dtype=torch.float32))
            self.weights['exit_points_conditional_output_weights'] = nn.Parameter(glorot_init([2 * (v_dim + h_dim), 1]))

        # record the total number of features
        feature_dimension = 6 * expanded_h_dim + 2 * v_dim
        self.params["feature_dimension"] = feature_dimension
        # weights for edge logits
        self.weights['combine_edge_v_weight'] = nn.Parameter(glorot_init([v_dim, v_dim]))
        self.weights['edge_iteration_in'] = nn.Parameter(glorot_init([feature_dimension+1, feature_dimension+1]))
        self.weights['edge_iteration_biases_in'] = nn.Parameter(torch.zeros([1, feature_dimension+1], dtype=torch.float32))
        self.weights['edge_iteration_output_in'] = nn.Parameter(glorot_init([feature_dimension+1, 1]))
        # weights for stop nodes
        self.weights['stop_node_in'] = nn.Parameter(glorot_init([1, expanded_h_dim]))
        self.weights['stop_node_vec_in'] = nn.Parameter(glorot_init([1, v_dim, 3]))

        # atten weights for positions prediction
        if self.params['if_generate_pos']:
            cat_dim = 2 * expanded_h_dim + v_dim
            self.weights['A1_matrix'] = nn.Parameter(glorot_init([v_dim, v_dim]))
            self.weights['A2_matrix'] = nn.Parameter(glorot_init([v_dim, v_dim]))
            self.weights['scores1_hidden'] = nn.Parameter(glorot_init([cat_dim, cat_dim]))
            self.weights['scores1_biases'] = nn.Parameter(torch.zeros([1, cat_dim], dtype=torch.float32))
            self.weights['scores1_output'] = nn.Parameter(glorot_init([cat_dim, 1]))
            self.weights['scores2_hidden'] = nn.Parameter(glorot_init([cat_dim, cat_dim]))
            self.weights['scores2_biases'] = nn.Parameter(torch.zeros([1, cat_dim], dtype=torch.float32))
            self.weights['scores2_output'] = nn.Parameter(glorot_init([cat_dim, 1]))
            # self.weights['self_interaction_hidden'] = nn.Parameter(glorot_init([2 * v_dim, 2 * v_dim]))
            self.weights['self_interaction_Q'] = nn.Parameter(glorot_init([2 * v_dim, 2 * v_dim]))
            self.weights['self_interaction_K'] = nn.Parameter(glorot_init([2 * v_dim, 2 * v_dim]))
            self.weights['self_interaction_output'] = nn.Parameter(glorot_init([2 * v_dim, 2 * v_dim]))
            # self.weights['cross_interaction_hidden'] = nn.Parameter(glorot_init([2 * v_dim, 2 * v_dim]))
            self.weights['cross_interaction_Q'] = nn.Parameter(glorot_init([2 * v_dim, 2 * v_dim]))
            self.weights['cross_interaction_K'] = nn.Parameter(glorot_init([2 * v_dim, 2 * v_dim]))
            self.weights['cross_interaction_output'] = nn.Parameter(glorot_init([2 * v_dim, 1]))

            # weights for updating positions
            if self.params['if_update_pos']:
                self.weights['A1_matrix_update'] = nn.Parameter(glorot_init([v_dim, v_dim]))
                self.weights['A2_matrix_update'] = nn.Parameter(glorot_init([v_dim, v_dim]))
                self.weights['scores1_hidden_update'] = nn.Parameter(glorot_init([cat_dim, cat_dim]))
                self.weights['scores1_biases_update'] = nn.Parameter(torch.zeros([1, cat_dim], dtype=torch.float32))
                self.weights['scores1_output_update'] = nn.Parameter(glorot_init([cat_dim, 1]))
                self.weights['scores2_hidden_update'] = nn.Parameter(glorot_init([cat_dim, cat_dim]))
                self.weights['scores2_biases_update'] = nn.Parameter(torch.zeros([1, cat_dim], dtype=torch.float32))
                self.weights['scores2_output_update'] = nn.Parameter(glorot_init([cat_dim, 1]))
                # self.weights['self_interaction_hidden'] = nn.Parameter(glorot_init([2 * v_dim, 2 * v_dim]))
                self.weights['self_interaction_Q_update'] = nn.Parameter(glorot_init([2 * v_dim, 2 * v_dim]))
                self.weights['self_interaction_K_update'] = nn.Parameter(glorot_init([2 * v_dim, 2 * v_dim]))
                self.weights['self_interaction_output_update'] = nn.Parameter(glorot_init([2 * v_dim, 2 * v_dim]))
                # self.weights['cross_interaction_hidden'] = nn.Parameter(glorot_init([2 * v_dim, 2 * v_dim]))
                self.weights['cross_interaction_Q_update'] = nn.Parameter(glorot_init([2 * v_dim, 2 * v_dim]))
                self.weights['cross_interaction_K_update'] = nn.Parameter(glorot_init([2 * v_dim, 2 * v_dim]))
                self.weights['cross_interaction_output_update'] = nn.Parameter(glorot_init([2 * v_dim, 1]))
        #self.weights['key_matrix'] = nn.Parameter(glorot_init([expanded_h_dim, expanded_h_dim]))
        #self.weights['query_matrix'] = nn.Parameter(glorot_init([expanded_h_dim, expanded_h_dim]))
        #self.weights['atten_hidden_weights'] = nn.Parameter(glorot_init([2*expanded_h_dim, 2*expanded_h_dim]))
        #self.weights['atten_hidden_biases'] = nn.Parameter(torch.zeros([1, 2*expanded_h_dim], dtype=torch.float32))
        #self.weights['atten_output_weights'] = nn.Parameter(glorot_init([2*expanded_h_dim, 1]))
        #self.weights['atten_pred_x'] = nn.Parameter(glorot_init([expanded_h_dim, expanded_h_dim]))

        # weights for edge type logits
        for i in range(self.num_edge_types):
            self.weights['edge_type_%d_in' % i] = nn.Parameter(glorot_init([feature_dimension+1,feature_dimension+1]))
            self.weights['edge_type_biases_%d_in' % i] = nn.Parameter(torch.zeros([1, feature_dimension+1]))
            self.weights['edge_type_output_%d_in' % i] = nn.Parameter(glorot_init([feature_dimension+1, 1]))

        # weights for node symbol logits
        self.weights['node_symbol_weights_in'] = nn.Parameter(glorot_init([h_dim+1, h_dim+1]))
        self.weights['node_symbol_biases_in'] = nn.Parameter(torch.zeros([1, h_dim+1]))
        self.weights['node_symbol_hidden_in'] = nn.Parameter(glorot_init([h_dim+1, self.params['num_symbols']]))
        self.weights['node_combine_weights_in'] = nn.Parameter(glorot_init([h_dim+1, h_dim+1]))
        self.weights['node_atten_weights_c_in'] = nn.Parameter(glorot_init([h_dim+1, h_dim+1]))
        self.weights['node_atten_weights_y_in'] = nn.Parameter(glorot_init([h_dim+1, h_dim+1]))

        # weights for prior distribution
        # self.weights['prior_logvariance'] = nn.Parameter(torch.zeros([1, 3]))

        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.params['learning_rate'])
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.params['lr_decay'])

    def get_node_embedding_state(self, one_hot_state, source):
        node_symbols = torch.argmax(one_hot_state, dim=2)
        if source:
            return self.units['node_embedding'](node_symbols) * self.ops['graph_state_mask_in']
        else:
            return self.units['node_embedding'](node_symbols) * self.ops['graph_state_mask_out']

    def get_idx_embedding(self, one_hot_idx, source):
        node_idx = torch.argmax(one_hot_idx, dim=2)
        if source:
            return self.units['idx_embedding'](node_idx) * self.ops['graph_state_mask_in']
        else:
            return self.units['idx_embedding'](node_idx) * self.ops['graph_state_mask_out']

    def compute_final_node_representations_EGNN(self, h, x0, adj, scope_name, fixed_coor=False):
        x = x0.clone()
        v = self.data['num_vertices']
        if scope_name == '_encoder':  # encoding: h_dim (repr)
            h_dim = self.params['hidden_size'] + self.params['idx_size']
        else:  # decoding: h_dim (repr) + h_dim (embedding) + 1 (focus bit)
            h_dim = self.params['hidden_size'] + self.params['hidden_size'] + 1 + self.params['idx_size']

        # record all hidden states at each iteration
        h = torch.reshape(h, [-1, h_dim])
        all_hidden_states = [h]
        for iter_idx in range(self.params['num_timesteps']):
            weights_suffix = scope_name + str(iter_idx)
            # compute the (h_i, h_j) pair features. WARNING: this approach will consume a large amount of memory
            _, h_sender = pairwise_construct(h.reshape([-1, v, h_dim]), v)
            # H = torch.cat((h_receiver, h_sender),
            # dim=3)  # (batch_size, v, v, h_dim), H_{n, i, j} = (h_i, h_j) for n-th data
            # compute distance matrix
            x_self, x_others = pairwise_construct(x, v)
            x_delta = torch.add(x_self, -x_others)  # displacement vectors
            dist = torch.linalg.norm(x_delta, dim=3)
            dist = torch.unsqueeze(dist, dim=3)
            # # final features (h_j, ||x_i-x_j||)
            H = torch.cat((h_sender, dist), dim=3)
            # free useless variables
            del h_sender, x_self, x_others
            for edge_type in range(self.num_edge_types):
                # the message passed from this vertex to others
                m = fully_connected(H, self.weights['edge_hidden_weights' + weights_suffix][edge_type],
                                    self.weights['edge_biases' + weights_suffix][edge_type],
                                    self.weights['edge_output_weights' + weights_suffix][edge_type])
                m = torch.nn.SiLU()(m)
                # collect the messages from other vertices to each vertex
                if edge_type == 0:
                    message = m
                    # acts = torch.einsum('bijk, bij->bik', m, adj[:, edge_type]) # contraction of index j
                    acts = torch.sum(torch.unsqueeze(adj[:, edge_type], dim=3) * m, dim=2)
                else:
                    message += m
                    # acts += torch.einsum('bijk, bij->bik', m, adj[:, edge_type])
                    acts += torch.sum(torch.unsqueeze(adj[:, edge_type], dim=3) * m, dim=2)
            # all messages collected for each node
            del m
            acts = torch.reshape(acts, [-1, h_dim])

            # add residual connection
            layer_residual_connections = self.params['residual_connections'].get(iter_idx)
            if layer_residual_connections is None:
                layer_residual_states = []
            else:
                layer_residual_states = [all_hidden_states[residual_layer_idx]
                                         for residual_layer_idx in layer_residual_connections]
            # concat current hidden states with residual states
            acts = torch.cat([acts] + layer_residual_states, dim=1)

            # feed message inputs and hidden states to combine function
            h = self.units['node_gru' + weights_suffix](acts, h)
            # record the new hidden states
            all_hidden_states.append(h)
            del acts
            # update coordinates
            combine_weights = fully_connected(message, self.weights['coor_hidden_weights' + weights_suffix],
                                              self.weights['coor_biases' + weights_suffix],
                                              self.weights['coor_output_weights' + weights_suffix])
            combine_weights = torch.tanh(combine_weights)
            combine_weights = torch.reshape(combine_weights, [-1, v, v]) * \
                              torch.reshape(self.ops['graph_state_mask_out'], [-1, 1, v])
            if not fixed_coor:
                # x += self.weights['coor_coef' + weights_suffix] * torch.einsum('bijk, bij->bik', x_delta, combine_weights) # classic EGNN
                x += torch.einsum('bijk, bij, beij->bik', x_delta / (dist + 1), combine_weights, adj)  # normalized EGNN

        last_h = torch.reshape(all_hidden_states[-1], [-1, v, h_dim])
        return last_h, x

    def compute_final_node_representations_VGNN(self, h, v, x, adj, scope_name):
        # h: hidden scalars, x: 3d positions, v: hidden vectors
        #x = v0.clone()
        num_atoms = self.data['num_vertices']
        v_dim = self.params['vector_size']
        if scope_name == '_encoder':  # encoding: h_dim (repr)
            h_dim = self.params['hidden_size'] + self.params['idx_size']
        else:  # decoding: h_dim (repr) + h_dim (embedding) + 1 (focus bit)
            h_dim = self.params['hidden_size'] + self.params['hidden_size'] + 1 + self.params['idx_size']

        # record all hidden states at each iteration
        # h = torch.reshape(h, [-1, h_dim])
        all_hidden_states = [h]
        all_hidden_vec = []
        for iter_idx in range(self.params['num_timesteps']):
            weights_suffix = scope_name + str(iter_idx)
            # compute distance matrix
            x_self, x_others = pairwise_construct(x, num_atoms)
            x_delta = torch.add(x_self, -x_others)  # displacement vectors
            dist = torch.linalg.norm(x_delta, dim=3)
            dist = torch.unsqueeze(dist, dim=3)
            # dist += SMALL_NUMBER
            # free useless variables
            del x_self, x_others

            # compute message
            m_h, m_v = self.compute_message(h, v, x_delta, dist, adj, weights_suffix)
            # compute update
            # add residual connection
            layer_residual_connections = self.params['residual_connections'].get(iter_idx)
            if layer_residual_connections is None:
                layer_residual_states = []
                layer_residual_vec = []
            else:
                layer_residual_states = [all_hidden_states[residual_layer_idx]
                                         for residual_layer_idx in layer_residual_connections]
                layer_residual_vec = [all_hidden_vec[residual_layer_idx]
                                         for residual_layer_idx in layer_residual_connections]
            # concat current hidden states with residual states
            m_h = torch.cat([m_h] + layer_residual_states, dim=2)
            m_v = torch.cat([m_v] + layer_residual_vec, dim=2)
            h, v = self.compute_update(h, v, m_h, m_v, weights_suffix)
            all_hidden_states.append(h)
            all_hidden_vec.append(v)
        return h, v

    def compute_message(self, h, v, x_delta, dist, adj, weights_suffix):
        h_dim = h.size(-1)
        # compute three rbf kernels
        w_rbfs = []
        v_norm = torch.norm(torch.einsum('bnci, cd->bndi', v, self.weights['combine_h_v_weight'+weights_suffix]), dim=3) + SMALL_NUMBER
        rbf = torch.linspace(0, self.params['max_correlation_length'], self.params['num_rbf'],
                             device=self.device)
        rbf = torch.square(torch.tile(dist, [1, 1, 1, self.params['num_rbf']]) - rbf) * self.weights[
            'wave_num' + weights_suffix]
        rbf = torch.exp(- rbf)
        for i in range(3):
            w_rbf = torch.einsum('nh, bijn->bijh', self.weights['weights_rbf_'+ str(i) +weights_suffix], rbf) + self.weights['biases_rbf_'+str(i)+weights_suffix]
            w_rbfs.append(w_rbf)
        m = 0
        m_v = 0
        for edge_type in range(self.num_edge_types):
            #phi_h_s = fully_connected(h, self.weights['edge_s_hidden_weights' + weights_suffix][edge_type],
            #                        self.weights['edge_s_biases' + weights_suffix][edge_type],
            #                        self.weights['edge_s_output_weights' + weights_suffix][edge_type])
            phi_h_s = scalar_neuron(torch.cat([h, v_norm], dim=2), self.weights['edge_s_output_weights'+weights_suffix][edge_type],
                    self.weights['edge_s_biases'+weights_suffix][edge_type])
            #phi_h_s = fully_connected(torch.cat([h, v_norm], dim=2), self.weights['edge_s_hidden_weights' + weights_suffix][edge_type],
            #                        self.weights['edge_s_biases' + weights_suffix][edge_type],
            #                        self.weights['edge_s_output_weights' + weights_suffix][edge_type])
            #phi_h_v0 = fully_connected(h, self.weights['edge_v0_hidden_weights' + weights_suffix][edge_type],
            #                        self.weights['edge_v0_biases' + weights_suffix][edge_type],
            #                       self.weights['edge_v0_output_weights' + weights_suffix][edge_type])
            phi_h_v0 = scalar_neuron(h, self.weights['edge_v0_output_weights'+weights_suffix][edge_type],
                    self.weights['edge_v0_biases'+weights_suffix][edge_type])
            phi_h_v1 = scalar_neuron(h, self.weights['edge_v1_output_weights' + weights_suffix][edge_type],
                                     self.weights['edge_v1_biases' + weights_suffix][edge_type])
            #phi_h_v1 = fully_connected(h, self.weights['edge_v1_hidden_weights' + weights_suffix][edge_type],
            #                        self.weights['edge_v1_biases' + weights_suffix][edge_type],
            #                        self.weights['edge_v1_output_weights' + weights_suffix][edge_type])
            #phi_v_v = fully_connected_vec(v,
             #                         self.weights['edge_v_nonlinear_Q'+weights_suffix][edge_type],
             #                         self.weights['edge_v_nonlinear_K' + weights_suffix][edge_type],
             #                         self.weights['edge_v_output_weights'+weights_suffix][edge_type])
            #phi_v_v = vector_neuron_leaky(v, self.weights['edge_v_nonlinear_Q'+weights_suffix][edge_type],
            #                          self.weights['edge_v_nonlinear_K' + weights_suffix][edge_type])
            phi_v_v = vector_neuron(v, self.weights['edge_v_nonlinear_Q' + weights_suffix][edge_type],
                                          self.weights['edge_v_nonlinear_K' + weights_suffix][edge_type])
            # phi_v_norm = torch.unsqueeze(torch.norm(w_v, dim=3), dim=3) + SMALL_NUMBER
            
            m += torch.einsum('bjh, bijh, bij->bijh', phi_h_s, w_rbfs[0], adj[:, edge_type])
            m_v += torch.einsum('bjvk, bjv, bijv, bij->bijvk', phi_v_v, phi_h_v0, w_rbfs[1], adj[:, edge_type]) + \
                    torch.einsum('bijk, bjv, bijv, bij->bijvk', x_delta, phi_h_v1, w_rbfs[2], adj[:, edge_type])
            # m += torch.einsum('bjh, bij->bijh', phi_h_s, adj[:, edge_type])
            #m_v += torch.einsum('bjvk, bjv, bijv, bij->bijvk', phi_v_v, phi_h_v0, w_rbfs[1], adj[:, edge_type]) + torch.einsum('bijk, bjv, bijv, bij->bijvk', x_delta, phi_h_v1, w_rbfs[2], adj[:, edge_type])
            #if edge_type == 0:
                # m = torch.einsum('bjh, bijh, bij->bijh', phi_h_s, w_rbfs[0], adj[:, edge_type])
            #    m = torch.einsum('bjh, bij->bijh', phi_h_s, adj[:, edge_type])
            #    m_v = torch.einsum('bjvk, bjv, bijv, bij->bijvk', phi_v_v, phi_h_v0, w_rbfs[1], adj[:, edge_type]) + \
                    #        torch.einsum('bijk, bjv, bijv, bij->bijvk', x_delta, phi_h_v1, w_rbfs[2], adj[:, edge_type])
            #else:
                # m += torch.einsum('bjh, bijh, bij->bijh', phi_h_s, w_rbfs[0], adj[:, edge_type])
             #   m += torch.einsum('bjh, bij->bijh', phi_h_s, adj[:, edge_type])
              #  m_v += torch.einsum('bjvk, bjv, bijv, bij->bijvk', phi_v_v, phi_h_v0, w_rbfs[1], adj[:, edge_type]) + \
                      #       torch.einsum('bijk, bjv, bijv, bij->bijvk', x_delta, phi_h_v1, w_rbfs[2],
                       #            adj[:, edge_type])

        # aggregate messages
        m = torch.sum(m, dim=2)
        m_v = torch.sum(m_v, dim=2)
        return m, m_v

 
    def compute_update(self, h, v, m_h, m_v, weights_suffix):
        h_shape = h.size()
        h = h.reshape([-1, h.size(-1)])
        m_h = m_h.reshape([-1, m_h.size(-1)])
        h_new = self.units['node_gru'+weights_suffix](m_h, h)
        # m_v = torch.zeros_like(m_v, device=self.device)
        # v_new = 0.5 * v + 0.5 * vector_linear(torch.cat([m_v2, m_v], dim=-2), self.weights['aggregate_v_output'+weights_suffix])
        v_new = fully_connected_vec(torch.cat([v, m_v], dim=-2),
                self.weights['aggregate_Q_weights'+weights_suffix],
                              self.weights['aggregate_K_weights'+weights_suffix],
                                                    self.weights['aggregate_v_output'+weights_suffix])
        # v_new = self.units['node_vec_gru' + weights_suffix](v, m_v) 
        #v_new = 0.8 * vector_linear(v, self.weights['self_aggregate_Q_weights' + weights_suffix]) + \
        #        0.2 * vector_linear(m_v, self.weights['cross_aggregate_Q_weights' + weights_suffix])
        # v_new = v
        h_new = h_new.reshape(h_shape)
        return h_new, v_new

    def compute_update2(self, h, v, m_h, m_v, weights_suffix):
        num_atoms = h.size(1)
        h_dim = h.size(2)
        v_dim = v.size(2)
        h = h.reshape([-1, h_dim])
        m_h = m_h.reshape([-1, h_dim])
        v = v.reshape([-1, v_dim, 3])
        m_v = m_v.reshape([-1, v_dim, 3])
        V = torch.einsum('vu, nvj->nuj', self.weights['V_matrix'+weights_suffix], m_v)
        U = torch.einsum('vu, nvj->nuj', self.weights['U_matrix'+weights_suffix], m_v)
        #Q = torch.einsum('vu, nvj->nuj', self.weights['Q_matrix' + weights_suffix], m_v)
        #K = torch.einsum('vu, nvj->nuj', self.weights['K_matrix' + weights_suffix], m_v)
        V_norm = torch.linalg.norm(V, dim=2)
        #K_norm = torch.linalg.norm(K, dim=2)
        m_ex = torch.cat([m_h, V_norm], dim=1)
        a_hh = fully_connected(m_ex, self.weights['aggregate_hh_hidden'+weights_suffix],
                               self.weights['aggregate_hh_biases'+weights_suffix],
                               self.weights['aggregate_hh_output'+weights_suffix])
        a_hv = fully_connected(m_ex, self.weights['aggregate_hv_hidden'+weights_suffix],
                               self.weights['aggregate_hv_biases'+weights_suffix],
                               self.weights['aggregate_hv_output'+weights_suffix])
        a_vv = fully_connected(m_ex, self.weights['aggregate_vv_hidden'+weights_suffix],
                               self.weights['aggregate_vv_biases'+weights_suffix],
                               self.weights['aggregate_vv_output'+weights_suffix])
        inner_product = torch.einsum('nvi, nvi->nv', U, V)
        inner_product = torch.matmul(inner_product, self.weights['inner_product_weights'+weights_suffix])
        h_new = self.units['node_update_gru'+weights_suffix](a_hh + a_hv * inner_product,
                                                             h)
        # v_new = torch.cat([v, torch.unsqueeze(a_vv, dim=2) * U], dim=1)
        temp = torch.einsum('nvi, nvi->nv', U, V) / torch.square(V_norm+SMALL_NUMBER)
        temp = torch.unsqueeze(temp * (temp < 0), dim=2)
        v_new = torch.cat([v, torch.unsqueeze(a_vv, dim=2) * (U - temp * V)], dim=1)
        #v_new = fully_connected_vec(v_new, self.weights['vec_update_hidden_weights'+weights_suffix],
                                    #self.weights['vec_update_nonlinear_Q'+weights_suffix],
                                    #self.weights['vec_update_nonlinear_K'+weights_suffix],
                                    #self.weights['vec_update_output_weights'+weights_suffix])
        v_new = torch.einsum('nvi, vu->nui', v_new, self.weights['vec_update_weights'+weights_suffix])
        return h_new.reshape([-1, num_atoms, h_dim]), v_new.reshape([-1, num_atoms, v_dim, 3])


    def compute_mean_and_logvariance(self):
        v = self.data['num_vertices']
        h_dim = self.params['hidden_size'] + self.params['idx_size']
        out_dim = self.params['encoding_size']
        out_vec_dim = self.params['encoding_vec_size']
        v_dim = self.params['vector_size']

        # Full molecule's encoding: average of all node's representations
        avg_last_h_out = torch.sum(self.ops['final_node_representations_out'] * self.ops['graph_state_mask_out'], dim=1) / \
                                                                                torch.sum(self.ops['graph_state_mask_out'], dim=1)
        mean_h_out = torch.matmul(avg_last_h_out, self.weights['mean_h_weights_out']) + self.weights['mean_h_biases_out']
        logvariance_h_out = torch.matmul(avg_last_h_out, self.weights['variance_h_weights_out']) + self.weights['variance_h_biases_out']
        mean_h_out_ex = torch.reshape(torch.tile(torch.unsqueeze(mean_h_out, 1), [1, v, 1]), [-1, out_dim])
        logvariance_h_out_ex = torch.reshape(torch.tile(torch.unsqueeze(logvariance_h_out, 1), [1, v, 1]), [-1, out_dim])

        # Node's encoding of unlinked fragments
        reshaped_last_h = torch.reshape(self.ops['final_node_representations_in'], [-1, h_dim])
        mean_h = torch.matmul(reshaped_last_h, self.weights['mean_h_weights']) + self.weights['mean_h_biases']

        # Node's encoding of full molecule
        reshaped_last_h_out = torch.reshape(self.ops['final_node_representations_out'], [-1, h_dim])
        # mean_h_out_all = torch.matmul(reshaped_last_h_out, self.weights['mean_h_weights_out_all']) + \
        #                  self.weights['mean_h_biases_out_all']
        mean_h_out_all = fully_connected(reshaped_last_h_out, self.weights['mean_h_hidden_out_all'],
                                         self.weights['mean_h_biases_out_all'], self.weights['mean_h_weights_out_all'])
        # logvariance_h_out_all = torch.matmul(reshaped_last_h_out, self.weights['variance_h_weights_out_all']) + self.weights[
        #    'variance_h_biases_out_all']
        logvariance_h_out_all = fully_connected(reshaped_last_h_out, self.weights['variance_h_hidden_out_all'],
                                                self.weights['variance_h_biases_out_all'],
                                                self.weights['variance_h_weights_out_all'])
        
        
        # Node's vector encodings
        reshaped_last_v_out = torch.reshape(self.ops['final_node_vec_representations_out'], [-1, v_dim, 3])
        reshaped_last_v_in = torch.reshape(self.ops['final_node_vec_representations_in'], [-1, v_dim, 3])
        mean_v_out_all = torch.einsum('nvi, vu->nui', reshaped_last_v_out, self.weights['mean_v_weights_out_all'])
        #logvariance_v_out = torch.norm(torch.einsum('nvi, vu->nui', reshaped_last_v_out, self.weights['variance_v_weights_out']), dim=2) + \
        #    self.weights['variance_v_biases_out']
        logvariance_v_out_all = torch.matmul(reshaped_last_h_out, self.weights['variance_v_weights_out_all']) + \
                                self.weights['variance_v_biases_out_all']
        logvariance_v_out_all = torch.tile(torch.unsqueeze(logvariance_v_out_all, dim=2), [1, 1, 3])
        mean_v_in = torch.einsum('nvi, vu->nui', reshaped_last_v_in, self.weights['mean_v_weights_in'])

        
        # fully molecule's vector representation
        avg_last_v_out = torch.sum(self.ops['final_node_vec_representations_out'] * torch.unsqueeze(self.ops['graph_state_mask_out'], dim=-2), dim=1)
        mean_v_out = torch.einsum('bci, cd->bdi', avg_last_v_out, self.weights['mean_v_weights_out'])
        logvariance_v_out = torch.matmul(avg_last_h_out, self.weights['variance_v_weights_out']) + \
                                self.weights['variance_v_biases_out']
        logvariance_v_out = torch.tile(torch.unsqueeze(logvariance_v_out, dim=-1), [1, 1, 3])
        logvariance_v_out_ex = torch.reshape(torch.tile(torch.unsqueeze(logvariance_v_out, dim=1), [1, v, 1, 1]), [-1, out_vec_dim, 3])
        mean_v_out_ex = torch.reshape(torch.tile(torch.unsqueeze(mean_v_out, dim=1), [1, v, 1, 1]), [-1, out_vec_dim, 3])
        # return: mean_h is the fragments' encodings; mean_h_out is the avg encoding; mean_h_out_all is the full
        return mean_h, mean_h_out_ex, logvariance_h_out_ex, mean_h_out_all, logvariance_h_out_all, mean_v_in, \
               mean_v_out_all, logvariance_v_out_all, mean_v_out_ex, logvariance_v_out_ex


    def sample_with_mean_and_logvariance(self):
        v = self.data['num_vertices']
        h_dim = self.params['hidden_size'] + self.params['idx_size']
        out_dim = self.params['encoding_size']
        out_vec_dim = self.params['encoding_vec_size']
        v_dim = self.params['vector_size']
        # Sample from N(0,1)
        z_prior_h = torch.reshape(self.data['z_prior_h'], [-1, out_dim])
        z_prior_h_in = torch.reshape(self.data['z_prior_h_in'], [-1, out_dim])
        z_prior_v = torch.reshape(self.data['z_prior_v'], [-1, out_vec_dim, 3])
        z_prior_v_in = torch.reshape(self.data['z_prior_v_in'], [-1, out_vec_dim, 3])#  * torch.sqrt(torch.exp(self.weights['prior_logvariance']))
        # Train: sample from N(u, sigma). Genearation: sample from N(0,1)
        if self.data['is_generative']:
            z_sampled_h = z_prior_h
            z_linker_sampled_h = z_prior_h_in
            z_linker_sampled_v = z_prior_v_in
            z_sampled_v = z_prior_v
        else:
            z_sampled_h = self.ops['mean_h_out'] + torch.multiply(torch.sqrt(torch.exp(self.ops['logvariance_h_out'])), z_prior_h)
            z_linker_sampled_h = self.ops['mean_h_out_all'] + torch.multiply(torch.sqrt(torch.exp(
                self.ops['logvariance_h_out_all'])), z_prior_h_in)
            z_linker_sampled_v = self.ops['mean_v_out_all'] + torch.multiply(torch.sqrt(torch.exp(
                self.ops['logvariance_v_out_all'])), z_prior_v_in)
            # z_linker_sampled_v = self.ops['mean_v_out_all'] + torch.sqrt(torch.tensor(self.params['prior_variance'], device=self.device)) * torch.multiply(torch.sqrt(torch.exp(
                #self.ops['logvariance_v_out_all'])), z_prior_v_in)
            z_sampled_v = self.ops['mean_v_out'] + torch.multiply(torch.sqrt(torch.exp(
                self.ops['logvariance_v_out'])), z_prior_v)
            # z_sampled_v = self.ops['mean_v_out'] + torch.sqrt(torch.tensor(self.params['prior_variance'], device=self.device)) * torch.multiply(torch.sqrt(torch.exp(
               # self.ops['logvariance_v_out'])), z_prior_v)
            # z_linker_sampled_h = z_prior_h_in
        z_sampled_h = torch.reshape(torch.matmul(z_sampled_h, self.weights['mean_h_combine_weights_in']), [-1, v, h_dim])
        z_linker_sampled_h = torch.reshape(torch.matmul(z_linker_sampled_h, self.weights['mean_h_all_combine_weights_in']), [-1, v, h_dim])
        z_linker_sampled_v = torch.reshape(torch.einsum('nci, cd->ndi', z_linker_sampled_v,
                                                        self.weights['mean_v_all_combine_weights_in']), [-1, v, v_dim, 3])
        z_sampled_v = torch.reshape(torch.einsum('nci, cd->ndi', z_sampled_v,
                                                        self.weights['mean_v_combine_weights_in']), [-1, v, v_dim, 3])
        # linkers encodings sampling
        mean_h = torch.reshape(self.ops['mean_h'], [-1, v, h_dim])
        mean_h = mean_h * self.ops['graph_state_mask_in']
        inverted_mask = torch.ones(self.ops['graph_state_mask_in'].size(), device=self.device) - self.ops['graph_state_mask_in']
        updated_vals = z_linker_sampled_h * inverted_mask
        mean_h = torch.reshape(mean_h + updated_vals, [-1, h_dim])

        # Combine fragments encodings with full molecule encodings
        # Attention mechanism over in-mol encodings to determine combination with z_sampled_h
        atten_masks_c = torch.tile(torch.unsqueeze(self.ops['graph_state_mask_out'], 2), [1, 1, v, 1]) * LARGE_NUMBER - LARGE_NUMBER
        atten_masks_yi = torch.tile(torch.unsqueeze(self.ops['graph_state_mask_out'], 1), [1, v, 1, 1]) * LARGE_NUMBER - LARGE_NUMBER
        atten_masks = atten_masks_yi + atten_masks_c

        atten_c = torch.tile(torch.unsqueeze(torch.reshape(mean_h, [-1, v, h_dim]), 2), [1, 1, v, 1])
        atten_yi = torch.tile(torch.unsqueeze(torch.reshape(mean_h, [-1, v, h_dim]), 1), [1, v, 1, 1])
        atten_c = torch.reshape(torch.matmul(torch.reshape(atten_c, [-1, h_dim]),
                                             self.weights['atten_h_weights_c_in']), [-1, v, v, h_dim])
        atten_yi = torch.reshape(torch.matmul(torch.reshape(atten_yi, [-1, h_dim]),
                                             self.weights['atten_h_weights_y_in']), [-1, v, v, h_dim])
        atten_mi = torch.nn.Sigmoid()(torch.add(atten_c, atten_yi) + atten_masks)
        atten_mi = torch.sum(atten_mi, 2) / torch.tile(torch.unsqueeze(torch.sum(
            self.ops['graph_state_mask_out'], 1), 1), [1, v, 1])

        mean_h_sampled = torch.reshape(mean_h, [-1, v, h_dim]) * self.ops['graph_state_mask_out'] # + atten_mi * z_sampled_h

        # sample vector encodings
        mean_v = torch.reshape(self.ops['mean_v_in'], [-1, v, v_dim, 3])
        mean_v = mean_v * torch.unsqueeze(self.ops['graph_state_mask_in'], dim=3)
        updated_vals = z_linker_sampled_v * torch.unsqueeze(inverted_mask, dim=3)
        mean_v = torch.reshape(mean_v + updated_vals, [-1, v, v_dim, 3])

        # compute attention
        Q = vector_linear(mean_v, self.weights['atten_v_weights_querys'])
        K = vector_linear(mean_v, self.weights['atten_v_weights_keys'])
        # V = vector_linear(mean_v, self.weights['atten_v_weights_values'])
        atten = torch.einsum('bvci, bwci->bvw', Q, K)
        atten = torch.nn.Softmax(dim=-2)(torch.unsqueeze(atten, dim=-1) + atten_masks)
        mean_v_sampled = mean_v * torch.unsqueeze(self.ops['graph_state_mask_out'], dim=-1) # + 0 * torch.einsum('bvwj, bwci->bvci', atten, z_sampled_v)

        return mean_h_sampled, mean_v_sampled

    def predict_positions(self, h, h_all, v, v_all, x, mask):
        # reshape
        # h = h.reshape()
        num_nodes = h.size(-2)
        # h_mean = torch.tile(torch.unsqueeze(torch.mean(h, dim=-2), dim=-2), [num_nodes, 1])
        A1 = vector_linear(v_all, self.weights['A1_matrix'])
        # A1_norm = torch.norm(A1, dim=-1)
        A2 = vector_linear(v, self.weights['A2_matrix'])
        # A2_norm = torch.norm(A2, dim=-1)
        inner_product = torch.sum(A1 * A2, dim=-1)
        # inner_product = inner_product / torch.
        temp = torch.cat([h, h_all, inner_product], dim=2)
        # predict mass center (E(3) equivariant)
        # x_mean = torch.einsum('bni, bn->bi', x, mask) / (torch.unsqueeze(torch.sum(mask, dim=1), dim=1) + SMALL_NUMBER) # mass center of the current graph
        x_mean = torch.sum(self.data['positions_in'], dim=1) / torch.sum(self.ops['graph_state_mask_in'],
                                                                                   dim=1)
        scores_1 = fully_connected(temp, self.weights['scores1_hidden'], self.weights['scores1_biases'],
                                 self.weights['scores1_output'])
        scores_1 = scores_1.reshape([-1, num_nodes])
        coefs_1 = torch.tanh(scores_1) * mask / torch.unsqueeze(torch.sum(mask, dim=-1), dim=-1)
        # probs_1 = torch.nn.Softmax(dim=1)(scores_1)
        x_center = x_mean + torch.torch.einsum('bn, bni->bi', coefs_1, x - torch.unsqueeze(x_mean, 1))
        # x_center = torch.torch.einsum('bn, bni->bi', coefs_1, x)
        # x_center = x_mean + torch.torch.einsum('bn, bni, bn->bi', coefs_1, x - torch.unsqueeze(x_mean, dim=-2), mask)
        # predict displacement (O(3) equivariant)
        scores_2 = fully_connected(temp, self.weights['scores2_hidden'], self.weights['scores2_biases'],
                                   self.weights['scores2_output'])
        scores_2 = scores_2.reshape([-1, num_nodes])
        coefs_2 = torch.tanh(scores_2) * mask / torch.unsqueeze(torch.sum(mask, dim=-1), dim=-1)
        # probs_2 = torch.nn.Softmax(dim=1)(scores_2)
        v_ex = torch.cat([v, v_all], dim=-2)
        v_ex = fully_connected_vec(v_ex, self.weights['self_interaction_Q'],
                            self.weights['self_interaction_K'], self.weights['self_interaction_output'])
        v_ex = torch.einsum('bn, bnci->bci', coefs_2, v_ex)
        x_displacement = fully_connected_vec(v_ex, self.weights['cross_interaction_Q'],
                                             self.weights['cross_interaction_K'],
                                             self.weights['cross_interaction_output'])
        x_displacement = x_displacement.reshape([-1, 3])
        # x_displacement = fully_connected_vec(v[:, 0], self.weights['cross_interaction_Q'],
        #                    self.weights['cross_interaction_K'], self.weights['cross_interaction_output'])
        # x_displacement = x_displacement.reshape([-1, 3])
        return x_center + x_displacement
   
    def update_positions(self, h_all, v_all, x, mask):
        num_atoms = h_all.size(1)
        # h_mean = torch.tile(torch.unsqueeze(torch.unsqueeze(torch.mean(h_all, dim=-2), dim=-2), dim=-2), [num_nodes, 1])
        x_self, x_others = pairwise_construct(x, num_atoms)
        x_delta = torch.add(x_self, -x_others)
        # x_delta = x_delta / (torch.unsqueeze(torch.linalg.norm(x_delta, dim=-1), dim=-1) + SMALL_NUMBER )
        del x_self, x_others
        h_self, h_others = torch.tile(torch.unsqueeze(h_all, dim=2), [1, 1, num_atoms, 1]), \
                           torch.tile(torch.unsqueeze(h_all, dim=1), [1, num_atoms, 1, 1])
        v_self, v_others = torch.tile(torch.unsqueeze(v_all, dim=2), [1, 1, num_atoms, 1, 1]), \
                           torch.tile(torch.unsqueeze(v_all, dim=1), [1, num_atoms, 1, 1, 1])
        A1 = vector_linear(v_self, self.weights['A1_matrix_update'])
        # A1_norm = torch.norm(A1, dim=-1)
        A2 = vector_linear(v_others, self.weights['A2_matrix_update'])
        # A2_norm = torch.norm(A2, dim=-1)
        # A1_norm = torch.tile(torch.unsqueeze(A1_norm, dim=2), [1, 1, num_atoms, 1])
        # A2_norm = torch.tile(torch.unsqueeze(A2_norm, dim=1), [1, num_atoms, 1, 1])
        inner_product = torch.sum(A1 * A2, dim=-1)
        temp = torch.cat([h_self, h_others, inner_product], dim=-1)
        scores_1 = fully_connected(temp, self.weights['scores1_hidden_update'], self.weights['scores1_biases_update'],
                                   self.weights['scores1_output_update'])
        scores_1 = scores_1.reshape([-1, num_atoms, num_atoms])
        coefs_1 = torch.einsum('bnm, bm->bnm', torch.tanh(scores_1), mask) / (torch.unsqueeze(torch.unsqueeze(torch.sum(mask, dim=-1), dim=-1), dim=-1) + SMALL_NUMBER)
        update_1 = torch.torch.einsum('bnm, bnmi->bni', coefs_1, x_delta)

        scores_2 = fully_connected(temp, self.weights['scores2_hidden_update'], self.weights['scores2_biases_update'],
                                   self.weights['scores2_output_update'])
        scores_2 = scores_2.reshape([-1, num_atoms, num_atoms])
        coefs_2 = torch.einsum('bnm, bm->bnm', torch.tanh(scores_2), mask) / (torch.unsqueeze(torch.unsqueeze(torch.sum(mask, dim=-1), dim=-1), dim=-1) + SMALL_NUMBER)
        v_ex = torch.cat([v_self, v_others], dim=-2)
        v_ex = fully_connected_vec(v_ex, self.weights['self_interaction_Q_update'],
                                   self.weights['self_interaction_K_update'], self.weights['self_interaction_output_update'])
        v_ex = torch.einsum('bnm, bnmci->bnci', coefs_2, v_ex)
        update_2 = fully_connected_vec(v_ex, self.weights['cross_interaction_Q_update'],
                                             self.weights['cross_interaction_K_update'],
                                             self.weights['cross_interaction_output_update'])
        update_2 = torch.reshape(update_2, [-1, num_atoms, 3])
        return update_1 + update_2


    def compute_loss(self):
        v = self.data['num_vertices']
        h_dim = self.params['hidden_size'] + self.params['idx_size']
        out_dim = self.params['encoding_size']
        out_vec_dim = self.params['encoding_vec_size']
        v_dim = self.params['vector_size']
        kl_trade_off_lambda = self.params['kl_trade_off_lambda']
        # kl_trade_off_lambda = torch.min(kl_trade_off_lambda * self.data['iter_step'] / 5,
        #                                torch.tensor(kl_trade_off_lambda, device=self.device))
        pos_trade_off_lambda = self.params['pos_trade_off_lambda']
        in_direc = 'in'
        out_direc = 'out'

        # compute exit points

        if self.params['if_generate_exit']:
            two_frags_mask = find_two_frags_with_idx(torch.sum(self.data['adjacency_matrix_in'], dim=1).cpu().detach().numpy(), self.data['exit_points'], 
                                                 self.device)
            exit_points_probs_1, exit_points_probs_2 = self.compute_exit_points_probs(two_frags_mask)

            exit_points_loss = self.compute_exit_points_loss(exit_points_probs_1 + exit_points_probs_2)
            self.ops['mean_exit_points_loss'] = torch.mean(exit_points_loss)
        # compute logit matrices: update self.cross_entropy_losses and self.node_symbol_logits
        # self.ops['node_predicted_coors'] = torch.clone(self.data['positions_in'])
        else:
            self.ops['mean_exit_points_loss'] = 0
       
        # compute logit matrices: update self.cross_entropy_losses and self.node_symbol_logits
        self.compute_logit_matrices()
        # self.predict_positions_ad_hoc()

        # Edge + edge's type prediction loss
        self.ops['edge_loss_' + in_direc] = torch.sum(self.ops['cross_entropy_losses_' + in_direc] * self.data['iteration_mask_' + out_direc], dim=1)

        # Kl loss
        frags_mask = self.ops['graph_state_mask_out'] - self.ops['graph_state_mask_in']
        kl_loss_in = 1 + self.ops['logvariance_h_out_all'] - torch.square(self.ops['mean_h_out_all']) - torch.exp(self.ops['logvariance_h_out_all'])
        kl_loss_in = torch.reshape(kl_loss_in, [-1, v, out_dim]) * frags_mask
        kl_loss_noise = 1 + self.ops['logvariance_h_out'] - torch.square(self.ops['mean_h_out']) - torch.exp(self.ops['logvariance_h_out'])
        kl_loss_noise = torch.reshape(kl_loss_noise, [-1, v, out_dim]) * self.ops['graph_state_mask_'+out_direc]
        prior_variance = torch.tensor(1, device=self.device)
        kl_loss_vec_in = 1 + self.ops['logvariance_v_out_all'] - torch.log(prior_variance) -\
                         torch.square(self.ops['mean_v_out_all']) / prior_variance - \
                         torch.exp(self.ops['logvariance_v_out_all']) / prior_variance
        kl_loss_vec_noise = 1 + self.ops['logvariance_v_out'] - torch.log(prior_variance) - \
                         torch.square(self.ops['mean_v_out']) / prior_variance - \
                         torch.exp(self.ops['logvariance_v_out']) / prior_variance
        #kl_loss_vec_in = 1 + self.ops['logvariance_v_out']- \
        #                 torch.square(self.ops['mean_v_out']) - \
        #                 torch.exp(self.ops['logvariance_v_out'])
        kl_loss_vec_in = torch.reshape(kl_loss_vec_in, [-1, v, out_vec_dim, 3]) * torch.unsqueeze(frags_mask, dim=3)
        kl_loss_vec_noise = torch.reshape(kl_loss_vec_noise, [-1, v, out_vec_dim, 3]) * torch.unsqueeze(self.ops['graph_state_mask_' + out_direc], dim=-1)
        self.ops['kl_loss_'+in_direc] = - 0.5 * torch.sum(kl_loss_in, [1, 2]) - 0 * torch.sum(kl_loss_noise, [1, 2]) -\
                                        0.5 * torch.sum(kl_loss_vec_in, [1, 2, 3]) - 0 * torch.sum(kl_loss_vec_noise, [1, 2, 3])

        # Node symbol loss
        self.ops['node_symbol_prob_'+in_direc] = self.compute_node_symbol_prob(self.ops['node_symbol_logits_'+in_direc])
        self.ops['node_symbol_loss_'+in_direc] = - torch.sum((torch.log(self.ops['node_symbol_prob_'+in_direc] + SMALL_NUMBER) *
                                                             self.data['node_symbols_'+out_direc]) * frags_mask,
                                                             dim=[1, 2])

        # Node positions loss

        pos_loss = self.compute_pos_loss(self.ops['node_predicted_coors'], self.data['positions_out'], frags_mask)
        # self.ops['mean_node_coors_loss'] = torch.mean(torch.sqrt(torch.sum(self.ops['node_pos_losses_'+in_direc], dim=1) / torch.sum(frags_mask, dim=[1, 2])))
        self.ops['mean_node_coors_loss'] = torch.mean(torch.sqrt(pos_loss))
        # self.ops['mean_node_coors_loss'] = torch.mean(torch.sum(self.ops['node_pos_losses_' + in_direc], dim=1) / torch.sum(frags_mask, dim=[1, 2]))
        # self.ops['mean_node_square_coors_loss'] = torch.mean(torch.sum(self.ops['node_pos_losses_'+in_direc], dim=1) / torch.sum(frags_mask, dim=[1, 2]))
        self.ops['mean_node_square_coors_loss'] = torch.mean(torch.log(pos_loss))
        # self.ops['mean_node_coors_loss'] = torch.mean(torch.sqrt(torch.sum(self.ops['node_pos_losses_'+in_direc], dim=1) / torch.sum(frags_mask, dim=[1, 2])))
        # self.ops['mean_node_coors_loss'] = torch.mean(torch.sum(self.ops['node_pos_losses_' + in_direc], dim=1) / torch.sum(frags_mask, dim=[1, 2]))
        # self.ops['mean_node_square_coors_loss'] = torch.log(torch.mean(torch.sum(self.ops['node_pos_losses_'+in_direc], dim=1) / torch.sum(frags_mask, dim=[1, 2])))

        # overall loss
        self.ops['mean_edge_loss_'+in_direc] = torch.mean(self.ops['edge_loss_'+in_direc]) + entropy(self.data['edge_labels_out'])
        self.ops['mean_node_symbol_loss_'+in_direc] = torch.mean(self.ops['node_symbol_loss_'+in_direc])
        self.ops['mean_kl_loss_'+in_direc] = torch.mean(self.ops['kl_loss_'+in_direc])
        self.ops['total_loss'] = self.ops['mean_edge_loss_'+in_direc] + self.ops['mean_node_symbol_loss_'+in_direc] + \
               kl_trade_off_lambda * self.ops['mean_kl_loss_'+in_direc] + \
                                 pos_trade_off_lambda * self.ops['mean_node_square_coors_loss'] + self.ops['mean_exit_points_loss']
    

    def compute_exit_points_probs(self, two_frags_mask):
        v = vector_linear(self.ops['z_sampled_v_in'], self.weights['exit_points_vec_weights'])
        v_norm = torch.norm(v, dim=-1)
        num_graphs = v.size(0)
        num_atoms = v.size(1)
        # get valences mask
        valences = [get_initial_valence(torch.argmax(self.data['node_symbols_in'], dim=-1).cpu().detach().numpy()[b],
                                        self.params['dataset']) for b in range(num_graphs)]
        valences = torch.tensor(valences, device=self.device)
        temp = torch.sum(self.data['adjacency_matrix_in'], dim=-1)
        temp = torch.unsqueeze(torch.unsqueeze(torch.tensor([1, 2, 3], device=self.device), dim=0), dim=-1) * temp
        valences = torch.unsqueeze(valences - torch.sum(temp, dim=1), dim=-1)
        valences[valences != 0] = 1
        # first fragment
        features = torch.cat([self.ops['z_sampled_h_in'], v_norm], dim=2)
        scores = fully_connected(features, self.weights['exit_points_hidden_weights'], self.weights['exit_points_biases'],
                                 self.weights['exit_points_output_weights'])
        scores_1 = scores + (two_frags_mask[:, 0] * LARGE_NUMBER - LARGE_NUMBER) + (valences * LARGE_NUMBER - LARGE_NUMBER)
        # scores_2 = scores + (two_frags_mask[:, 1] * LARGE_NUMBER - LARGE_NUMBER)
        probs_1 = torch.nn.Softmax(dim=1)(scores_1)
        # probs_2 = torch.nn.Softmax(dim=1)(scores_2)
        # second fragment
        features_of_frag1 = torch.tile(torch.unsqueeze(features[torch.arange(num_graphs), self.data['exit_points'][:, 0].type(torch.long)], dim=1), [1, num_atoms, 1])
        # features_of_frag2 = torch.tile(
        #    torch.unsqueeze(features[torch.arange(num_graphs), self.data['exit_points'][:, 1].type(torch.long)], dim=1),
        #    [1, num_atoms, 1])
        features_of_frag12 = features_of_frag1 * two_frags_mask[:, 1] # + features_of_frag2 * two_frags_mask[:, 0]
        del features_of_frag1
        features_of_frag_given_another = torch.cat([features, features_of_frag12], dim=-1)
        del features, features_of_frag12
        scores = fully_connected(features_of_frag_given_another, self.weights['exit_points_conditional_hidden_weights'],
                                 self.weights['exit_points_conditional_biases'],
                                 self.weights['exit_points_conditional_output_weights'])
        # scores_1 = scores + (two_frags_mask[:, 0] * LARGE_NUMBER - LARGE_NUMBER)
        scores_2 = scores + (two_frags_mask[:, 1] * LARGE_NUMBER - LARGE_NUMBER) + (valences * LARGE_NUMBER - LARGE_NUMBER)
        # probs_1 = probs_1 * torch.nn.Softmax(dim=1)(scores_1)
        probs_2 = torch.nn.Softmax(dim=1)(scores_2)
        return probs_1, probs_2
    
    def compute_exit_points_loss(self, probs):
        # exit_points = self.data['exit_points'].cpu().detach().numpy().astype(int)
        exit_mask = generate_exit_mask(self.data['exit_points'], probs.size(1))
        return - torch.sum(exit_mask * torch.log(probs + SMALL_NUMBER), dim=[1, 2])

    def sample_exit_points(self, two_frags_mask):
        v = vector_linear(self.ops['z_sampled_v_in'], self.weights['exit_points_vec_weights'])
        v_norm = torch.norm(v, dim=-1)
        num_graphs = v.size(0)
        num_atoms = v.size(1)
        # first fragment
        features = torch.cat([self.ops['z_sampled_h_in'], v_norm], dim=2)
        scores = fully_connected(features, self.weights['exit_points_hidden_weights'],
                                 self.weights['exit_points_biases'],
                                 self.weights['exit_points_output_weights'])
        scores_1 = scores + (two_frags_mask[:, 0] * LARGE_NUMBER - LARGE_NUMBER)
        probs_1 = torch.nn.Softmax(dim=1)(scores_1)
        exit_point_1 = torch.argmax(probs_1.reshape([-1, num_atoms]), dim=1)
        # second fragment
        features_of_frag1 = torch.tile(
            torch.unsqueeze(features[torch.arange(num_graphs), exit_point_1], dim=1),
            [1, num_atoms, 1])
        features_of_frag12 = features_of_frag1 * two_frags_mask[:, 1]
        del features_of_frag1
        features_of_frag_given_another = torch.cat([features, features_of_frag12], dim=-1)
        del features, features_of_frag12
        scores = fully_connected(features_of_frag_given_another, self.weights['exit_points_conditional_hidden_weights'],
                                 self.weights['exit_points_conditional_biases'],
                                 self.weights['exit_points_conditional_output_weights'])
        scores_2 = scores + (two_frags_mask[:, 1] * LARGE_NUMBER - LARGE_NUMBER)
        probs_2 = torch.nn.Softmax(dim=1)(scores_2)
        exit_point_2 = torch.argmax(probs_2, dim=1)
        return {exit_point_1.cpu().detach().item(), exit_point_2.cpu().detach().item()}
   
    def compute_pos_loss(self, x_pred, x_truth, frags_mask):
        return torch.sum(torch.square(torch.linalg.norm((x_truth - x_pred) * frags_mask, dim=-1)), dim=1) / \
                 torch.sum(frags_mask, dim=[1, 2])
   
    def compute_logit_matrices(self):
        v = self.data['num_vertices']
        batch_size = self.data['initial_node_representation_out'].size()[0]
        h_dim = self.params['hidden_size']

        in_direc = 'in'
        out_direc = 'out'

        # Initial state: embedding
        latent_node_state = self.get_node_embedding_state(self.data['latent_node_symbols_'+out_direc], source=False)
        # Concat z_smapled with node symbols
        filtered_z_sampled = torch.cat([self.ops['z_sampled_h_'+in_direc],
                                        latent_node_state], dim=2)
        self.ops['initial_repre_for_decoder_' + in_direc] = filtered_z_sampled
        # self.ops['initial_coor_for_decoder_' + in_direc] = torch.clone(self.data['positions_in'])
        # self.ops['initial_coor_for_decoder_'+in_direc] = torch.clone(self.data['positions_out'])
        # compute the cross entropy losses at each step
        cross_entropy_losses = torch.zeros(dtype=torch.float32, size=[batch_size, self.data['max_iteration_num']], device=self.device)
        edge_predictions = torch.zeros(dtype=torch.float32, size=[batch_size, self.data['max_iteration_num'], v + 1], device=self.device) # additional one for stop node
        edge_type_predictions = torch.zeros(dtype=torch.float32, size=[batch_size, self.data['max_iteration_num'], self.num_edge_types ,v], device=self.device)
        pos_rmsd_losses = torch.zeros(dtype=torch.float32, size=[batch_size, self.data['max_iteration_num']], device=self.device)
        for idx in range(self.data['max_iteration_num']):
            cross_entropy_losses[:, idx], edge_predictions[:, idx],\
            edge_type_predictions[:, idx], pos_rmsd_losses[:, idx] = self.generate_cross_entropy(idx)
        # Record the predictions for generation
        self.ops['edge_predictions_'+in_direc] = edge_predictions
        self.ops['edge_type_predictions'] = edge_type_predictions

        # Final cross entropy losses
        self.ops['cross_entropy_losses_'+in_direc] = cross_entropy_losses

        # node symbol predictions with attention mechanism
        self.ops['node_symbol_logits_'+in_direc] = self.compute_node_symbol_logit()

        # final rmsd position loss
        self.ops['node_pos_losses_'+in_direc] = pos_rmsd_losses

    def compute_node_symbol_logit(self):
        # compute node symbol logits based on z_sampled_in
        v = self.data['num_vertices']
        batch_size = self.data['initial_node_representation_out'].size()[0]
        h_dim = self.params['hidden_size'] + self.params['idx_size']
        in_direc = 'in'
        out_direc = 'out'

        # dist = torch.tile(torch.reshape(self.data['abs_dist'], [-1, 1, 2]), [1, v, 1])
        num_atoms = torch.unsqueeze(torch.tile(torch.sum(
            self.data['node_mask_' + out_direc] - self.data['node_mask_' + in_direc],
            dim=1, keepdim=True), [1, v]), 2)
        pos_info = num_atoms
        z_sampled = torch.cat([self.ops['z_sampled_h_' + in_direc], pos_info], dim=2)

        atten_masks_c = torch.tile(torch.unsqueeze(
            self.ops['graph_state_mask_' + out_direc] - self.ops['graph_state_mask_' + in_direc], 2),
            [1, 1, v, 1]) * LARGE_NUMBER - LARGE_NUMBER
        atten_masks_yi = torch.tile(torch.unsqueeze(
            self.ops['graph_state_mask_' + out_direc] - self.ops['graph_state_mask_' + in_direc], 1),
            [1, v, 1, 1]) * LARGE_NUMBER - LARGE_NUMBER
        atten_masks = atten_masks_yi + atten_masks_c
        atten_c = torch.tile(torch.unsqueeze(z_sampled, 2), [1, 1, v, 1])
        atten_yi = torch.tile(torch.unsqueeze(z_sampled, 1), [1, v, 1, 1])
        atten_c = torch.reshape(torch.matmul(torch.reshape(atten_c, [-1, h_dim + 1]),
                                             self.weights['node_atten_weights_c_' + in_direc]), [-1, v, v, h_dim + 1])
        atten_yi = torch.reshape(torch.matmul(torch.reshape(atten_yi, [-1, h_dim + 1]),
                                              self.weights['node_atten_weights_y_' + in_direc]), [-1, v, v, h_dim + 1])
        atten_mi = torch.nn.Sigmoid()(torch.add(atten_c, atten_yi) + atten_masks)
        atten_mi = torch.sum(atten_mi, 2) / torch.tile(torch.unsqueeze(torch.sum(
            self.ops['graph_state_mask_' + out_direc], 1), 1), [1, v, 1])

        z_sampled = z_sampled * self.ops['graph_state_mask_' + in_direc] + \
                    atten_mi * torch.reshape(torch.matmul(torch.reshape(z_sampled, [-1, h_dim + 1]),
                                                          self.weights['node_combine_weights_' + in_direc]),
                                             [-1, v, h_dim + 1])
        # logits for node symbols
        # logit = torch.reshape(torch.matmul(torch.reshape(z_sampled, [-1, h_dim + 3]), self.weights['node_symbol_weights_' + in_direc])
                                                                  # + self.weights['node_symbol_biases_' + in_direc],
                                                                   #[-1, v, self.params['num_symbols']])
        logit = fully_connected(torch.reshape(z_sampled, [-1, h_dim + 1]), self.weights['node_symbol_weights_'+in_direc],
                                self.weights['node_symbol_biases_'+in_direc], self.weights['node_symbol_hidden_'+in_direc])
        logit = torch.reshape(logit, [-1, v, self.params['num_symbols']])
        return logit


    def compute_node_symbol_prob(self, node_symbol_logits):
        return torch.nn.Softmax(dim=2)(node_symbol_logits)

    def decoder(self, idx, if_generation=False):
        # compute the representations of nodes for prediction based on init_repr_decoder
        # and existing graph at iteration idx
        direc = 'out'
        direc_r = 'in'
        v = self.data['num_vertices']
        h_dim = self.params['hidden_size']
        expand_h_dim = h_dim + h_dim + self.params['idx_size'] + 1
        batch_size = self.data['initial_node_representation_out'].size(0)
        # Use latent representation as decoder's input
        filtered_z_sampled = self.ops['initial_repre_for_decoder_' + direc_r]
        v_sampled = self.ops['z_sampled_v_' + direc_r]
        # x_sampled = torch.clone(self.ops['initial_coor_for_decoder_'+direc_r])

        if self.params['if_generate_pos']:
            if not if_generation and not self.params['if_update_pos']:
                x_sampled = self.data['positions_out']
            else:
                x_sampled = self.ops['node_predicted_coors']
        else:
            x_sampled = self.data['positions_out']
        # data needed in this iteration
        incre_adj_mat = self.data['incre_adj_mat_' + direc][:, idx]
        distance_to_others = self.data['distance_to_others_' + direc][:, idx]
        overlapped_edge_features = self.data['overlapped_edge_features_' + direc][:, idx]
        node_sequence = self.data['node_sequence_' + direc][:, idx]
        node_sequence = torch.unsqueeze(node_sequence, dim=2)
        # concat the hidden states with the node in focus
        filtered_z_sampled = torch.cat([filtered_z_sampled, node_sequence], dim=2)
        # Decoder GNN
        if self.params['use_graph']:
            if self.params['decoder'] == 'EGNN':
                new_filtered_z_sampled, new_x_sampled = self.compute_final_node_representations_EGNN(filtered_z_sampled
                                                                    , x_sampled, incre_adj_mat, '_decoder')
            else:
                new_filtered_z_sampled, new_x_sampled = self.compute_final_node_representations_VGNN(filtered_z_sampled,
                                                            v_sampled, x_sampled, incre_adj_mat, '_decoder')
        else:
            new_filtered_z_sampled, new_x_sampled = filtered_z_sampled, x_sampled
        # Filter nonexist nodes
        new_filtered_z_sampled = new_filtered_z_sampled * self.ops['graph_state_mask_' + direc]
        # Take out the node in focus
        node_in_focus = torch.sum(node_sequence * new_filtered_z_sampled, dim=1)
        # concat vector features
        node_vec_in_focus = torch.einsum('bnj, bnci->bci', node_sequence, new_x_sampled)
        v_norm = torch.norm(torch.einsum('bnci, cd->bndi', new_x_sampled, self.weights['combine_edge_v_weight']), dim=3)
        v_norm_in_focus = torch.norm(torch.einsum('bci, cd->bdi', node_vec_in_focus, self.weights['combine_edge_v_weight']), dim=2)
        v_norm_in_focus_ex = torch.tile(torch.unsqueeze(v_norm_in_focus, dim=1), [1, v, 1])
        # edge pair representation
        edge_repr = torch.cat([torch.tile(torch.unsqueeze(node_in_focus, 1), [1, v, 1]), new_filtered_z_sampled,
                               v_norm_in_focus_ex, v_norm], dim=2)
        # combine edge representation with local and global representation
        local_graph_repr_before_expansion = torch.sum(new_filtered_z_sampled, dim=1) #/ \
                                            #torch.unsqueeze(torch.sum(self.data['node_mask_' + direc], dim=1), dim=1)
        local_graph_repr = torch.unsqueeze(local_graph_repr_before_expansion, dim=1)
        local_graph_repr = torch.tile(local_graph_repr, [1, v, 1])
        global_graph_repr_before_expandsion = torch.sum(filtered_z_sampled, dim=1) #/ \
                                              #torch.unsqueeze(torch.sum(self.data['node_mask_' + direc], dim=1), dim=1)
        global_graph_repr = torch.unsqueeze(global_graph_repr_before_expandsion, dim=1)
        global_graph_repr = torch.tile(global_graph_repr, [1, v, 1])
        # distance representation
        distance_repr = self.units['distance_embedding_' + direc_r](distance_to_others)
        # overlapped edge feature representation
        overlapped_edge_repr = self.units['overlapped_edge_weight_' + direc_r](overlapped_edge_features)
        # concat and reshape
        combined_edge_repr = torch.cat([edge_repr, local_graph_repr, global_graph_repr,
                                        distance_repr, overlapped_edge_repr], dim=2)
        combined_edge_repr = torch.reshape(combined_edge_repr,
                                           [-1, self.params['feature_dimension']])
        # Add 3d structural info (dist, ang) and iteration number
        # dist = torch.reshape(torch.tile(torch.reshape(self.data['abs_dist'], [-1, 1, 2]), [1, v, 1]), [-1, 2])
        it_num = (idx + self.data['it_num']).type(torch.float32)
        it_num = torch.reshape(it_num, [1, 1])
        it_num = torch.tile(it_num, [combined_edge_repr.size(0), 1])
        # pos_info = torch.cat([dist, it_num], dim=1)
        pos_info = it_num
        combined_edge_repr = torch.cat([combined_edge_repr, pos_info], dim=1)
        return combined_edge_repr, node_in_focus, v_norm_in_focus, local_graph_repr_before_expansion, \
               global_graph_repr_before_expandsion, new_filtered_z_sampled, new_x_sampled

    def compute_edge_logits(self, combined_edge_repr, node_in_focus, v_norm_in_focus, local_graph_repr_before_expansion,
                            global_graph_repr_before_expandsion, idx):
        # compute edge logits based on combined_edge_repr and existing graph at iteration idx
        direc = 'out'
        direc_r = 'in'
        v = self.data['num_vertices']
        h_dim = self.params['hidden_size'] + self.params['idx_size']
        batch_size = self.data['initial_node_representation_out'].size()[0]
        edge_masks = self.data['edge_masks_' + direc][:, idx]
        edge_masks = edge_masks * LARGE_NUMBER - LARGE_NUMBER
        #node_sequence = self.data['node_sequence_' + direc][:, idx]
        #node_sequence = torch.unsqueeze(node_sequence, dim=2)

        # Compute edge prediction logits
        edge_logits = fully_connected(combined_edge_repr, self.weights['edge_iteration_' + direc_r],
                                      self.weights['edge_iteration_biases_' + direc_r],
                                      self.weights['edge_iteration_output_' + direc_r])
        edge_logits = torch.reshape(edge_logits, [-1, v])
        # filter invalid nodes
        edge_logits = edge_logits + edge_masks
        # Calculate whether it will stop at this step
        # Prepare the data
        expanded_stop_node = torch.tile(self.weights['stop_node_' + direc_r], [batch_size, 1])

        distance_to_stop_node = self.units['distance_embedding_' + direc_r](torch.tile(torch.tensor([0], device=self.device),
                                                                                       [batch_size]))
        overlap_edge_stop_node = self.units['overlapped_edge_weight_' + direc_r](
            torch.tile(torch.tensor([0], device=self.device), [batch_size]))
        # concat vec features
        v_norm_stop = torch.norm(torch.einsum('bci, cd->bdi', self.weights['stop_node_vec_'+direc_r], self.weights['combine_edge_v_weight']), dim=2)
        v_norm_stop = torch.tile(v_norm_stop, [batch_size, 1])
        combined_stop_node_repr = torch.cat([node_in_focus, expanded_stop_node, v_norm_in_focus, v_norm_stop, local_graph_repr_before_expansion,
                                             global_graph_repr_before_expandsion, distance_to_stop_node,
                                             overlap_edge_stop_node], dim=1)
        # dist = self.data['abs_dist']
        it_num = (idx + self.data['it_num']).type(torch.float32)
        it_num = torch.reshape(it_num, [1, 1])
        it_num = torch.tile(it_num, [combined_stop_node_repr.size(0), 1])
        pos_info = it_num
        combined_stop_node_repr = torch.cat([combined_stop_node_repr, pos_info], dim=1)
        # logits for stop node
        stop_logits = fully_connected(combined_stop_node_repr, self.weights['edge_iteration_' + direc_r],
                                      self.weights['edge_iteration_biases_' + direc_r],
                                      self.weights['edge_iteration_output_' + direc_r])
        edge_logits = torch.cat([edge_logits, stop_logits], dim=1)
        return edge_logits

    def compute_edge_type_logits(self, combined_edge_repr, idx):
        direc = 'out'
        direc_r = 'in'
        v = self.data['num_vertices']
        # make invalid locations to be very small before using softmax function
        edge_type_masks = self.data['edge_type_masks_' + direc][:, idx]
        edge_type_masks = edge_type_masks * LARGE_NUMBER - LARGE_NUMBER
        # Compute edge type logits
        edge_type_logits = []
        for i in range(self.num_edge_types):
            edge_type_logit = fully_connected(combined_edge_repr,
                                              self.weights['edge_type_%d_%s' % (i, direc_r)],
                                              self.weights['edge_type_biases_%d_%s' % (i, direc_r)],
                                              self.weights['edge_type_output_%d_%s' % (i, direc_r)])
            edge_type_logits.append(torch.reshape(edge_type_logit, [-1, 1, v]))
        edge_type_logits = torch.cat(edge_type_logits, dim=1)
        # filter invalid items
        edge_type_logits = edge_type_logits + edge_type_masks
        return edge_type_logits.type(torch.float32)



    def generate_cross_entropy(self, idx):
        direc = "out"
        direc_r = "in"
        # decoder: transform the initial representation to codings for prediction
        combined_edge_repr, node_in_focus, v_norm_in_focus, local_graph_repr_before_expansion, \
        global_graph_repr_before_expandsion, z_sampled, v_sampled = self.decoder(idx)

        # compute edge probability for existing graph
        edge_logits = self.compute_edge_logits(combined_edge_repr, node_in_focus, v_norm_in_focus,
                                               local_graph_repr_before_expansion,
                                               global_graph_repr_before_expandsion, idx)
        edge_probs = compute_edge_probs(edge_logits)

        # compute edge type probability for existing graph
        edge_type_logits = self.compute_edge_type_logits(combined_edge_repr, idx)
        edge_type_probs = compute_edge_type_probs(edge_type_logits)

        # ground truth
        edge_type_labels = self.data['edge_type_labels_'+direc][:, idx]
        edge_labels = self.data['edge_labels_'+direc][:, idx]
        local_stop = self.data['local_stop_'+direc][:, idx]

        # edge labels
        pos_loss = 0
        edge_labels = torch.cat([edge_labels, torch.unsqueeze(local_stop, 1)], dim=1)
        # softmax for edge
        edge_loss = - torch.sum(torch.log(edge_probs + SMALL_NUMBER) * edge_labels, dim=1)
        # softmax for edge type
        edge_type_loss = - edge_type_labels * torch.log(edge_type_probs + SMALL_NUMBER)
        edge_type_loss = torch.sum(edge_type_loss, dim=[1, 2])
        if self.params['if_generate_pos']:
            # position prediction
            # stage 1: initialize positions of new added node
            adj_mat = self.data['incre_adj_mat_out'][:, idx]
            current_mask = torch.sum(adj_mat, dim=[1, 2]) != 0
            current_mask = current_mask.type(torch.float32)
            if self.params['if_update_pos']:
                current_pos = self.ops['node_predicted_coors']
            else:
                current_pos = self.data['positions_out'].clone()
            # init_pos = self.data['positions_out'].clone()
            num_atoms = self.data['num_vertices']
            num_graphs = adj_mat.size(0)
            node_to_link = torch.argmax(edge_labels, dim=1)
            node_to_link[node_to_link == num_atoms] = 0
            self.ops['node_predicted_coors'][torch.arange(num_graphs), node_to_link] += self.initialize_positions(z_sampled, v_sampled, edge_labels, current_pos,
                                                                                                                  current_mask, idx)

            # stage 2: update all nodes' positions in linker when encounter stop nodes
            if self.params['if_update_pos']:
                update = self.update_positions(z_sampled, v_sampled, current_pos,
                                               current_mask - self.ops['graph_state_mask_in'].reshape([-1, num_atoms]))
                #update = self.update_positions(z_sampled, v_sampled, current_pos,
                #                               current_mask)
                node_to_link2 = torch.argmax(edge_labels, dim=1)
                linker_mask = torch.unsqueeze(current_mask, dim=-1) - self.ops['graph_state_mask_in']
                flag = torch.unsqueeze(node_to_link2 == num_atoms, 1)
                self.ops['node_predicted_coors'] += update * linker_mask * torch.unsqueeze(flag, dim=1)

        # total loss
        iteration_loss = edge_loss + edge_type_loss
        return iteration_loss, edge_probs, edge_type_probs, pos_loss

    def initialize_positions(self, z_sampled, v_sampled, edge_labels, current_pos, current_mask, idx):
        num_atoms = self.data['num_vertices']
        num_graphs = z_sampled.size(0)
        node_to_link = torch.argmax(edge_labels, dim=1)
        node_to_link[node_to_link == num_atoms] = 0
        link_index = torch.tile(torch.unsqueeze(torch.unsqueeze(node_to_link, dim=1), dim=2),
                                [1, num_atoms, z_sampled.size(-1)])
        h_ex = z_sampled.gather(1, link_index)
        link_index = torch.tile(torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(node_to_link, dim=1), dim=2), dim=3),
                                [1, num_atoms, v_sampled.size(-2), v_sampled.size(-1)])
        v_ex = v_sampled.gather(1, link_index)
        # x_pred, x_center, precisions = self.predict_positions_vote(h_ex, z_sampled, v_ex, v_sampled, current_pos, current_mask)
        x_pred = self.predict_positions(h_ex, z_sampled, v_ex, v_sampled, current_pos,
                                        current_mask)
        node_to_link2 = torch.argmax(edge_labels, dim=1)
        flag = (current_mask.gather(1, torch.unsqueeze(node_to_link, dim=1)) == 0) * torch.unsqueeze(
            node_to_link2 != num_atoms, 1) * \
               torch.unsqueeze((edge_labels != 0).any(1), dim=1)
        return flag * x_pred
 

    def optimize_step(self, if_step_and_zero_grad):
        # print('Total loss is %f' % self.ops['total_loss'])
        self.ops['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.params['clamp_gradient_norm'])
        if if_step_and_zero_grad:
            self.optimizer.step()
            self.optimizer.zero_grad()

    def set_zero_grad(self):
        self.optimizer.zero_grad()



    def generate_graph_with_state(self, random_normal_states_h, random_normal_states_in_h,
                                  random_normal_states_v, random_normal_states_in_v, num_vertices,
                                  generated_all_smiles, generate_all_mols, elements, count, correct_exit):
        # Load initial fragments data info
        self.load_dynamic_data_info(elements, None, None, num_vertices,
                                    None, None, None, None, None,
                                    random_normal_states_h, random_normal_states_in_h, random_normal_states_v,
                                    random_normal_states_in_v, 0)
        # encode the fragments: obtain encoding z_sampled_in; sample from N(0, 1)
        self.encoder()

        # sample exit points
        exit_truth = {self.data['exit_points'][:, i].item() for i in range(2)}
        if self.params['if_generate_exit']:
            two_frags_mask = find_two_frags_with_idx(
                torch.sum(self.data['adjacency_matrix_in'], dim=1).cpu().detach().numpy(), [[0, 1]],
                self.device)
            exit_points = self.sample_exit_points(two_frags_mask)
            if exit_points == exit_truth:
                correct_exit.append(1)
        else:
            exit_points = exit_truth


        # Get predicted node probs (symbol and keep)
        self.ops['node_symbol_logits_in'] = self.compute_node_symbol_logit()
        predicted_node_symbol_prob = self.compute_node_symbol_prob(self.ops['node_symbol_logits_in'])
        # Node numbers for each graph
        real_length = get_graph_length([elements['mask_out']])[0]
        # Sample node symbols
        sampled_node_symbol = \
        self.sample_node_symbol(predicted_node_symbol_prob.cpu().detach().numpy(), [real_length], self.params['dataset'])[0]
        # Sample vertices to keep
        sampled_node_keep = elements['v_to_keep']
        for node, keep in enumerate(sampled_node_keep):
            if keep == 1:
                sampled_node_symbol[node] = np.argmax(elements['init_in'][node])
        # Maximum Valences for each node
        valences = get_initial_valence(sampled_node_symbol, self.params['dataset'])
        # Randomly pick the starting point or use zero
        if not self.params["path_random_order"]:
            # Try different starting points
            if self.params["try_different_starting"]:
                starting_point = random.sample(range(real_length),
                                               min(self.params["num_different_starting"], real_length))
            else:
                starting_point = [0]
        else:
            if self.params["try_different_starting"]:
                starting_point = random.sample(range(real_length),
                                               min(self.params["num_different_starting"], real_length))
            else:
                starting_point = [random.choice(list(range(real_length)))]  # randomly choose one
        # Record all molecules from different starting points
        all_mol = []
        for idx in starting_point:
            # Generate a new molecule
            new_mol, total_log_prob = self.search_and_generate_molecule(idx, np.copy(valences),
                                                                        sampled_node_symbol, sampled_node_keep,
                                                                        real_length, random_normal_states_h,
                                                                        random_normal_states_in_h, random_normal_states_v,
                                                                        random_normal_states_in_v, elements, num_vertices,
                                                                        exit_points)
            # If multiple select best only by total_log_prob
            if self.params['dataset'] == 'zinc' and new_mol is not None:
                all_mol.append((0, total_log_prob, new_mol))
        # Select one out
        best_mol = select_best(all_mol)
        # Nothing generated
        if best_mol is None:
            return list(exit_points)
        # Record generated molecule
        generated_all_smiles.append(elements['smiles_in'] + " " + elements['smiles_out'] +
                                    " " + Chem.CanonSmiles(Chem.MolToSmiles(best_mol)))
        generate_all_mols.append(best_mol)
        # dump('%s_generated_smiles_%s' % (self.run_id, self.params['dataset']), generated_all_smiles)
        # Progress
        if count % 100 == 0:
            print('Generated mols %d' % count)

        return list(exit_points)


    def generate_new_graphs(self, data):
        (bucketed, bucket_sizes, bucket_at_step) = data
        bucket_counters = defaultdict(int)
        # all generated smiles
        generated_all_smiles = []
        generated_all_mols = []
        # counter
        count = 0
        correct_exit = []
        sampled_exits = []
        for step in range(len(bucket_at_step)):
            bucket = bucket_at_step[step]
            # data index
            start_idx = bucket_counters[bucket] * self.params['batch_size']
            end_idx = (bucket_counters[bucket] + 1) * self.params['batch_size']
            # batch data
            elements_batch = bucketed[bucket][start_idx:end_idx]
            for elements in elements_batch:
                # Allow control over number of additional atoms during generation
                maximum_length = self.compensate_node_length(elements, bucket_sizes[bucket])
                # Generate multiple outputs per mol in valid/test set
                for _ in range(self.params['number_of_generation_per_valid']):
                    # initial state
                    random_normal_states_h = generate_std_normal(1, maximum_length,
                                                               self.params['encoding_size'])
                    random_normal_states_in_h = generate_std_normal(1, maximum_length,
                                                                  self.params['encoding_size'])
                    random_normal_states_v = torch.normal(0, 1, [1, maximum_length, self.params['encoding_vec_size'], 3])
                    random_normal_states_in_v = torch.normal(0, 1, [1, maximum_length, self.params['encoding_vec_size'], 3])
                    with torch.no_grad():
                        sampled_exit = self.generate_graph_with_state(random_normal_states_h, random_normal_states_in_h,
                                                   random_normal_states_v, random_normal_states_in_v,
                                            maximum_length, generated_all_smiles, generated_all_mols, elements, count,
                                                       correct_exit)
                    count += 1
                sampled_exits.append(sampled_exit)
            bucket_counters[bucket] += 1
        # Terminate when loop finished
        print('Generation done')
        # Save smi output in non-pickle format
        print('Number of generated SMILES: %d' % len(generated_all_smiles))
        if self.params['output_name'] != '':
            file_name = os.path.join(self.params['output_name'],
                                     '%s_generated_smiles_%s.smi' % (self.run_id, self.params['dataset']))
            file_name_sdf = os.path.join(self.params['output_name'],
                                     '%s_generated_smiles_%s.sdf' % (self.run_id, self.params['dataset']))
        else:
            file_name = '%s_generated_smiles_%s.smi' % (self.run_id, self.params['dataset'])
            file_name_sdf = '%s_generated_smiles_%s.sdf' % (self.run_id, self.params['dataset'])
        with open(file_name, 'w') as out_file:
            for line in generated_all_smiles:
                out_file.write(line + '\n')

        # save exit nodes
        with open("anchor_nodes", "wb") as fp:  # Pickling
            pickle.dump(sampled_exits, fp)


        # Save mol output
        w = Chem.SDWriter(file_name_sdf)
        for m in generated_all_mols:
            w.write(m)

        # print successful exit predictions
        print('Percentage of correct exit: %f' % (np.sum(np.array(correct_exit)) / count))
    def compensate_node_length(self, elements, bucket_size):
        maximum_length = bucket_size + self.params['compensate_num']
        real_length = get_graph_length([elements['mask_in']])[0]
        real_length_out = get_graph_length([elements['mask_out']])[0] + self.params['compensate_num']
        elements['mask_out'] = [1] * real_length_out + [0] * (maximum_length - real_length_out)
        elements['init_out'] = np.zeros([maximum_length, self.params['num_symbols']])
        elements['adj_mat_out'] = []
        elements['mask_in'] = [1] * real_length + [0] * (maximum_length - real_length)
        elements['init_in'] = np.pad(elements['init_in'],
                                     pad_width=[[0, self.params['compensate_num']], [0, 0]],
                                     mode='constant')
        elements['v_to_keep'] = np.pad(elements['v_to_keep'], pad_width=[[0, self.params['compensate_num']]],
                                       mode='constant')
        elements['positions_out'] = np.pad(elements['positions_out'], [[0, self.params['compensate_num']],
                                                                       [0, 0]], mode='constant')
        elements['positions_in'] = np.pad(elements['positions_in'], [[0, self.params['compensate_num']],
                                                                       [0, 0]], mode='constant')
        return maximum_length

    def load_dynamic_data_info(self, elements, latent_node_symbol, incre_adj_mat, num_vertices,
                              distance_to_others, overlapped_edge_dense, node_sequence, edge_type_masks,
                              edge_masks, random_normal_states_h, random_normal_states_in_h,
                               random_normal_states_v, random_normal_states_in_v, iteration_num):
        if incre_adj_mat is None:
            incre_adj_mat = np.zeros([1, 1, self.num_edge_types, 1, 1])
            distance_to_others = np.zeros([1, 1, 1])
            overlapped_edge_dense = np.zeros([1, 1, 1])
            node_sequence = np.zeros([1, 1, 1])
            edge_type_masks = np.zeros([1, 1, self.num_edge_types, 1])
            edge_masks = np.zeros([1, 1, 1])
            latent_node_symbol = np.zeros([1, 1, self.params['num_symbols']])
        self.data['z_prior_h'] = random_normal_states_h
        self.data['z_prior_h_in'] = random_normal_states_in_h
        self.data['z_prior_v_in'] = random_normal_states_in_v
        self.data['z_prior_v'] = random_normal_states_v
        self.data['incre_adj_mat_out'] = torch.tensor(incre_adj_mat, dtype=torch.float32)
        self.data['num_vertices'] = torch.tensor(num_vertices)
        self.data['initial_node_representation_in'] = torch.tensor(self.pad_annotations([elements['init_in']]))
        self.data['initial_node_representation_out'] = torch.tensor(self.pad_annotations([elements['init_out']]))
        self.data['node_symbols_out'] = torch.tensor(np.array([elements['init_out']]))
        self.data['node_symbols_in'] = torch.tensor(np.array([elements['init_in']]))
        self.data['latent_node_symbols_in'] = torch.tensor(self.pad_annotations(latent_node_symbol))
        self.data['latent_node_symbols_out'] = torch.tensor(self.pad_annotations(latent_node_symbol))
        # self.data['adjacency_matrix_in'] = torch.tensor([elements['adj_mat_in']], dtype=torch.float32)
        # self.data['adjacency_matrix_out'] = torch.tensor([elements['adj_mat_out']], dtype=torch.float32)
        self.data['adjacency_matrix_in'] = torch.tensor(np.array([graph_to_adj_mat(elements['adj_mat_in'], num_vertices,
        self.num_edge_types, self.params['tie_fwd_bkwd'])]), dtype=torch.float32)
        self.data['adjacency_matrix_out'] = torch.tensor(np.array([graph_to_adj_mat(elements['adj_mat_out'], num_vertices,
            self.num_edge_types, self.params['tie_fwd_bkwd'])]), dtype=torch.float32)
        self.data['vertices_to_keep'] = torch.tensor([elements['v_to_keep']])
        self.data['exit_points'] = torch.tensor([elements['exit_points']])
        self.data['abs_dist'] = torch.tensor(str2float([elements['abs_dist']]))
        self.data['it_num'] = torch.tensor([iteration_num])
        self.data['node_mask_in'] = torch.tensor([elements['mask_in']])
        self.data['node_mask_out'] = torch.tensor([elements['mask_out']])
        self.data['graph_state_keep_prob'] = torch.tensor(1)
        self.data['edge_weight_dropout_keep_prob'] = torch.tensor(1)
        self.data['iteration_mask_out'] = torch.tensor([[1]])
        self.data['is_generative'] = True
        self.data['out_layer_dropout_keep_prob'] = torch.tensor(1)
        self.data['distance_to_others_out'] = torch.tensor(distance_to_others)
        self.data['overlapped_edge_features_out'] = torch.tensor(overlapped_edge_dense)
        self.data['max_iteration_num'] = torch.tensor(1)
        self.data['node_sequence_out'] = torch.tensor(node_sequence, dtype=torch.float32)
        self.data['edge_type_masks_out'] = torch.tensor(edge_type_masks)
        self.data['edge_masks_out'] = torch.tensor(edge_masks)
        self.ops['graph_state_mask_in'] = torch.unsqueeze(self.data['node_mask_in'], 2)
        self.ops['graph_state_mask_out'] = torch.unsqueeze(self.data['node_mask_out'], 2)
        self.data['positions_out'] = torch.reshape(torch.tensor(elements['positions_out'], dtype=torch.float32),
                                                     [-1, num_vertices, 3])
        self.data['positions_in'] = torch.reshape(torch.tensor(elements['positions_in'], dtype=torch.float32),
                                                  [-1, num_vertices, 3])
        if self.params['use_cuda']:
            for key in self.data.keys():
                if torch.is_tensor(self.data[key]):
                    self.data[key] = self.data[key].to(self.device)
            self.ops['graph_state_mask_in'] = self.ops['graph_state_mask_in'].to(self.device)
            self.ops['graph_state_mask_out'] = self.ops['graph_state_mask_out'].to(self.device)


    def pad_annotations(self, annotations):
        return np.pad(annotations,
                       pad_width=[[0, 0], [0, 0], [0, self.params['hidden_size'] - self.params["num_symbols"]]],
                       mode='constant')


    def search_and_generate_molecule(self, initial_idx, valences, sampled_node_symbol,
                                     sampled_node_keep, real_n_vertices, random_normal_states_h,
                                     random_normal_states_in_h, random_normal_states_v, random_normal_states_in_v,
                                     elements, max_n_vertices, exit_points):
        # New molecule
        new_mol = Chem.MolFromSmiles('')
        new_mol = Chem.rdchem.RWMol(new_mol)

        # Add atoms
        add_atoms(new_mol, sampled_node_symbol, self.params['dataset'])
        # Initialize queue
        queue = deque([])

        # color 0: have not visited; color 1: in the queue; color 2: searched already
        color = [0] * max_n_vertices
        # Empty adj list at the beginning
        incre_adj_list = defaultdict(list)

        count_bonds = 0
        # Add edges between vertices to keep
        for src, bond, des in elements['adj_mat_in']:
            if sampled_node_keep[src] and sampled_node_keep[des]:
                incre_adj_list[src].append((des, bond))
                incre_adj_list[des].append((src, bond))
                valences[src] -= (bond + 1)
                valences[des] -= (bond + 1)
                # add bond to mol
                new_mol.AddBond(int(src), int(des), number_to_bond[bond])
                count_bonds += 1
        # Add exit nodes to queue and update colors of fragment nodes
        for v, keep in enumerate(sampled_node_keep[0:real_n_vertices]):
            if keep == 1: # nodes in fragments
                if v in exit_points:
                    queue.append(v)
                    color[v] = 1
                else:
                    # Mask out nodes that are not exit points
                    valences[v] = 0
                    color[v] = 2
        # Record the log probability at each step
        total_log_prob = 0
        # Add initial_idx to queue if no nodes kept
        if len(queue) == 0:
            queue.append(initial_idx)
            color[initial_idx] = 1

        iteration_num = 0

        while len(queue) > 0:
            node_in_focus = queue.popleft()
            # iterate until the stop node is selected
            while True:
                edge_type_mask_sparse, edge_mask_sparse = generate_mask(valences, incre_adj_list, color, real_n_vertices,
                              node_in_focus, self.params['check_overlap_edge'], new_mol)
                edge_type_mask = edge_type_masks_to_dense([edge_type_mask_sparse], max_n_vertices,
                                                          self.num_edge_types)
                edge_mask = edge_masks_to_dense([edge_mask_sparse], max_n_vertices)
                node_sequence = node_sequence_to_dense([node_in_focus], max_n_vertices)
                distance_to_others_sparse = bfs_distance(node_in_focus, incre_adj_list)
                distance_to_others = distance_to_others_dense([distance_to_others_sparse], max_n_vertices)
                overlapped_edge_sparse = get_overlapped_edge_feature(edge_mask_sparse, color, new_mol)
                overlapped_edge_dense = overlapped_edge_features_to_dense([overlapped_edge_sparse], max_n_vertices)
                incre_adj_mat = incre_adj_mat_to_dense([incre_adj_list], self.num_edge_types, max_n_vertices)
                sampled_node_symbol_one_hot = self.node_symbol_one_hot(sampled_node_symbol, real_n_vertices, max_n_vertices)

                # load current graph data info
                self.load_dynamic_data_info(elements, [sampled_node_symbol_one_hot], [incre_adj_mat], max_n_vertices,
                                            [distance_to_others], [overlapped_edge_dense], [node_sequence],
                                            [edge_type_mask], [edge_mask], random_normal_states_h,
                                            random_normal_states_in_h, random_normal_states_v, random_normal_states_in_v,
                                            iteration_num)

                # prepare initial repre for decoder
                latent_node_state = self.get_node_embedding_state(self.data['latent_node_symbols_out'], source=False)
                # Concat z_smapled with node symbols
                filtered_z_sampled = torch.cat([self.ops['z_sampled_h_in'],
                                                latent_node_state], dim=2)
                self.ops['initial_repre_for_decoder_in'] = filtered_z_sampled

                # decoder: compute the representation for the current existing graph
                combined_edge_repr, node_in_focus_repr, v_norm_in_focus, local_graph_repr_before_expansion, \
                global_graph_repr_before_expandsion, z_sampled, v_sampled = self.decoder(0, if_generation=True) # 0 since we only look at the current iteration
                # compute edge logits and probs
                edge_logits = self.compute_edge_logits(combined_edge_repr, node_in_focus_repr, v_norm_in_focus,
                                                       local_graph_repr_before_expansion,
                                                       global_graph_repr_before_expandsion, 0)
                edge_probs = compute_edge_probs(edge_logits)
                # compute edge type logits and probs
                edge_type_logits = self.compute_edge_type_logits(combined_edge_repr, 0)
                edge_type_probs = compute_edge_type_probs(edge_type_logits)
                # increment number of iterations
                iteration_num += 1
                # select an edge
                if not self.params['use_argmax_generation']:
                    neighbor = np.random.choice(np.arange(max_n_vertices + 1), p=edge_probs[0].cpu().detach().numpy())
                else:
                    neighbor = torch.argmax(edge_probs[0]).item()
                # update log prob
                total_log_prob += torch.log(edge_probs[0, neighbor] + SMALL_NUMBER)

                if self.params['if_generate_pos']:
                    # position prediction
                    idx = 0
                    adj_mat = self.data['incre_adj_mat_out'][:, idx]
                    current_mask = torch.sum(adj_mat, dim=[1, 2]) != 0
                    current_mask = current_mask.type(torch.float32)
                    current_pos = self.ops['node_predicted_coors'].clone()
                    num_atoms = self.data['num_vertices']
                    num_graphs = adj_mat.size(0)
                    if neighbor != max_n_vertices:
                        # stage 1: initialize positions of new added node
                        # init_pos = self.data['positions_out'].clone()
                        edge_labels = torch.zeros([1, max_n_vertices], device=self.device)
                        edge_labels[0, neighbor] = 1
                        node_to_link = torch.tensor(neighbor, device=self.device)
                        self.ops['node_predicted_coors'][
                            torch.arange(num_graphs), node_to_link] += self.initialize_positions(z_sampled, v_sampled,
                                                                                             edge_labels, current_pos,
                                                                                                 current_mask, idx)

                    # stage 2: update all nodes' positions in linker when encounter stop nodes
                    if self.params['if_update_pos'] and neighbor == max_n_vertices:
                        update = self.update_positions(z_sampled, v_sampled, current_pos,
                                                       current_mask - self.ops['graph_state_mask_in'].reshape(
                                                           [-1, num_atoms]))
                        # update = self.update_positions(z_sampled, v_sampled, current_pos,
                        #                               current_mask)
                        linker_mask = torch.unsqueeze(current_mask, dim=-1) - self.ops['graph_state_mask_in']
                        self.ops['node_predicted_coors'] += update * linker_mask

                # if stop node is picked, break
                if neighbor == max_n_vertices:
                    break

                # or choose an edge type
                if not self.params['use_argmax_generation']:
                    bond = np.random.choice(np.arange(self.num_edge_types), p=edge_type_probs[0, :, neighbor].cpu().detach().numpy())
                else:
                    bond = torch.argmax(edge_type_probs[0, :, neighbor]).item()
                # update log prob
                total_log_prob += torch.log(edge_type_probs[0, bond, neighbor] + SMALL_NUMBER)
                # update valences
                valences[node_in_focus] -= (bond + 1)
                valences[neighbor] -= (bond + 1)
                # add the bond
                new_mol.AddBond(int(node_in_focus), int(neighbor), number_to_bond[bond])
                # add the edge to increment adj list
                incre_adj_list[node_in_focus].append((neighbor, bond))
                incre_adj_list[neighbor].append((node_in_focus, bond))
                # Explore neighbor nodes
                if color[neighbor] == 0:
                    queue.append(neighbor)
                    color[neighbor] = 1
            color[node_in_focus] = 2 # already visited
        # write positions into mol
        write_3d_pos(new_mol, self.ops['node_predicted_coors'].cpu().detach().numpy())
        # Remove unconnected nodes
        remove_extra_nodes(new_mol)
        # if valid
        # if not if_valid(new_mol, list(exit_points)):
        #    pass
        # new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))
        return new_mol, total_log_prob

    def node_symbol_one_hot(self, sampled_node_symbol, real_n_vertices, max_n_vertices):
        one_hot_representations = []
        for idx in range(max_n_vertices):
            representation = [0] * self.params["num_symbols"]
            if idx < real_n_vertices:
                atom_type = sampled_node_symbol[idx]
                representation[atom_type] = 1
            one_hot_representations.append(representation)
        return one_hot_representations

    def sample_node_symbol(self, all_node_symbol_prob, all_lengths, dataset):
        all_node_symbol = []
        for graph_idx, graph_prob in enumerate(all_node_symbol_prob):
            node_symbol = []
            for node_idx in range(all_lengths[graph_idx]):
                if self.params['use_argmax_generation']:
                    symbol = torch.argmax(graph_prob[node_idx]).item()
                else:
                    symbol = np.random.choice(np.arange(len(dataset_info(dataset)['atom_types'])),
                                              p=graph_prob[node_idx])
                node_symbol.append(symbol)
            all_node_symbol.append(node_symbol)
        return all_node_symbol


    def set_learning_rate(self, lr):
        self.params['learning_rate'] = lr
        for g in self.optimizer.param_groups:
            g['lr'] = lr


if __name__ == '__main__':
    args = docopt(__doc__)
    dataset = args.get('--dataset')
    model = Linker(args) # initialize model with random weights
    # load checkpoint if any
    if args.get('--load_cpt') != None:
        model.load_check_point(args.get('--load_cpt'))
    if args.get('--generation') == None:
        model.train(model.init_epoch) # model training
    else:
        model.generate_new_graphs(model.valid_data) # generate graphs
