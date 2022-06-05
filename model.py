import json
import pickle
import time
import os
import numpy as np
import random

import utils
from utils import dataset_info
import torch
import matplotlib.pyplot as plt


class ChemModel(torch.nn.Module):
    @classmethod
    def default_params(cls):
        return {}

    def __init__(self, args):
        super(ChemModel, self).__init__()

        # Collect argument things
        self.args = args
        data_dir = ''
        if args.get('--data_dir') is not None:
            data_dir = args['--data_dir']
        self.data_dir = data_dir

        # Collect parameters
        params = self.default_params()
        config_file = args.get('--config-file')
        config = args.get('--config')
        if config is not None and config_file is not None:
            print("Error: args 'config' and 'config_file' cannot be both specified!")
            exit(1)
        if config is None and config_file is None:
            print("Error: either args 'config' or 'config_file' is specified!")
            exit(1)
        if config_file is not None:
            with open(config_file, 'r') as f:
                params.update(json.load(f))
        else:
            params.update(json.loads(config))
        self.params = params

        # Get which dataset in use
        self.params['dataset'] = dataset = args.get('--dataset')
        # Number of atom types in this dataset
        self.params['num_symbols'] = len(dataset_info(dataset)['atom_types'])

        self.run_id = "_".join([time.strftime("%Y-%m-%d-%H-%M-%S"), str(os.getpid())])
        log_dir = args.get('--log_dir') or '.'
        self.log_file = os.path.join(log_dir, "%s_log_%s.json" % (self.run_id, dataset))
        self.best_model_file = os.path.join(log_dir, "%s_model.pickle" % self.run_id)

        if self.params['save_params_file']:
            with open(os.path.join(log_dir, "%s_params_%s.json" % (self.run_id, dataset)), "w") as f:
                json.dump(params, f)
        print("Run %s starting with following parameters:\n%s" % (self.run_id, json.dumps(self.params)))

        # set random seed
        seed = params.get('random_seed')
        if seed is None:
            print('Warning: random_seed is not specified and is chosen arbitrarily.\n')
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Load data:
        self.max_num_vertices = 0
        self.num_edge_types = 0
        self.annotation_size = 0
        self.train_data = self.load_data(params['train_file'], is_training_data=True)
        self.valid_data = self.load_data(params['valid_file'], is_training_data=False)

        # Make the actual model
        self.data = {}
        self.weights = {}
        self.ops = {}
        self.make_model()
        # transfer to designated device
        self.device = torch.device('cuda' if self.params['use_cuda'] and torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.init_epoch = 0

        # if self.params['if_use_autocast']:
        #    self.scaler = torch.cuda.amp.GradScaler()

    def load_data(self, file_name, is_training_data: bool):
        full_path = os.path.join(self.data_dir, file_name)

        print("Loading data from %s" % full_path)
        with open(full_path, 'r') as f:
            data = json.load(f)

        restrict = self.args.get("--restrict_data")  # FI - added int() as docopt was seeing as string
        if restrict is not None and int(restrict) > 0:
            data = data[:int(restrict)]

        # Get some common data out:
        num_fwd_edge_types = len(utils.bond_dict) - 1
        for g in data:
            self.max_num_vertices = max(self.max_num_vertices, max([v for e in g['graph_in'] for v in [e[0], e[2]]]),
                                        max([v for e in g['graph_out'] for v in [e[0], e[2]]]))
            
        self.num_edge_types = max(self.num_edge_types, num_fwd_edge_types * (1 if self.params['tie_fwd_bkwd'] else 2))
        self.annotation_size = max(self.annotation_size, len(data[0]["node_features_in"][0]))

        return self.process_raw_graphs(data, is_training_data, file_name)

    # encoder: from data to encodings
    def encoder(self):
        # Initial state embedding
        initial_state_in = self.get_node_embedding_state(self.data['initial_node_representation_in'], source=True)
        initial_state_out = self.get_node_embedding_state(self.data['initial_node_representation_out'],
                                                            source=False)
        # Initial node one-hot index
        temp = torch.tile(torch.unsqueeze(torch.eye(self.data['num_vertices'], device=self.device), dim=0),
                          [initial_state_out.size(0), 1, 1])
        node_index_in = self.get_idx_embedding(temp, source=True)
        node_index_out = self.get_idx_embedding(temp, source=False)

        initial_state_out = torch.cat([initial_state_out, node_index_out], dim=2)
        initial_state_in = torch.cat([initial_state_in, node_index_in], dim=2)

        # Initial 3d coordinates
        initial_pos_in = torch.clone(self.data['positions_in']) #+ self.data['pos_noise']
        initial_pos_out = torch.clone(self.data['positions_out']) #+ self.data['pos_noise']

        # Initial vector embedding
        initial_vec_in = torch.zeros([initial_state_in.size(0), self.data['num_vertices'], self.params['vector_size'],
                                      3], device=self.device)
        initial_vec_out = torch.zeros_like(initial_vec_in, device=self.device)

        # SE(3) equivariant Graph Neural Networks
        if self.params['use_graph']: # use EGNN
            self.ops['final_node_representations_in'], self.ops['final_node_vec_representations_in']\
                = self.compute_final_node_representations_VGNN(
                initial_state_in, initial_vec_in, initial_pos_in, self.data['adjacency_matrix_in'], '_encoder')
            #theta = 1
            #R = torch.tensor([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]], dtype=torch.float32, device=self.device)
            #pos_new = torch.einsum('ij, bni->bnj', R, initial_pos_in)
            #h_new, v_new \
                #= self.compute_final_node_representations_VGNN(
                #initial_state_in, initial_vec_in, pos_new, self.data['adjacency_matrix_in'], '_encoder')
            self.ops['final_node_representations_out'], self.ops['final_node_vec_representations_out']\
                = self.compute_final_node_representations_VGNN(
                initial_state_out, initial_vec_out, initial_pos_out, self.data['adjacency_matrix_out'], '_encoder')
        else: # only use embedding
            self.ops['final_node_representations_in'] = initial_state_in
            self.ops['final_node_representations_out'] = initial_state_out

        # Compute p(z|x) 's mean and log variance
        self.ops['mean_h'], self.ops['mean_h_out'], self.ops['logvariance_h_out'], self.ops['mean_h_out_all'],\
            self.ops['logvariance_h_out_all'], self.ops['mean_v_in'], self.ops['mean_v_out_all'], self.ops['logvariance_v_out_all'], \
            self.ops['mean_v_out'], self.ops['logvariance_v_out'] = self.compute_mean_and_logvariance()
        # Sample encodings z from p(z|x)
        self.ops['z_sampled_h_in'], self.ops['z_sampled_v_in'] = self.sample_with_mean_and_logvariance()
        # initialize predicted positions
        self.ops['node_predicted_coors'] = torch.clone(self.data['positions_in'])

    def train(self, init_epoch=0):
        train_loss = np.zeros([6, self.params['num_epochs']])
        valid_loss = np.zeros([6, self.params['num_epochs']])
        for epoch in range(init_epoch, self.params['num_epochs']):
            print('== Epoch %d' % epoch)
            self.data['iter_step'] = torch.tensor(epoch, device=self.device, dtype=torch.float32)
            # training
            train_loss[:, epoch] = self.run_epoch(self.train_data, is_training=True)
            # validation
            valid_loss[:, epoch] = self.run_epoch(self.valid_data, is_training=False)
            if self.params['if_save_check_point']:
                self.save_check_point(epoch)
            self.lr_scheduler.step()
        return train_loss, valid_loss

    def run_epoch(self, dataset, is_training):
        batch_dataset = self.make_minibatch_iterator(dataset, is_training)
        loss = np.zeros(6)
        acc_steps = self.params['accumulation_steps']
        # start_time = time.time()
        # processed_graphs = 0
        if is_training:
            for step, batch_data in enumerate(batch_dataset):
                #  self.set_zero_grad()  # initialize gradient
                # self.data = {} # clean out the previous data
                self.load_current_batch_as_tensor(batch_data)  # load a batch data
                num_graphs = self.data['num_graphs']
                if_step_and_zero_grad = ((step + 1) % acc_steps == 0)
                self.encoder()  # encode the data
                self.compute_loss()  # compute the loss from encodings
                self.optimize_step(if_step_and_zero_grad)  # back-propagation to update parameters
                total_loss, edge_loss, kl_loss, symbol_loss, pos_loss, exit_loss = float(self.ops['total_loss']),\
                            float(self.ops['mean_edge_loss_in']), float(self.ops['mean_kl_loss_in']), \
                            float(self.ops['mean_node_symbol_loss_in']), float(self.ops['mean_node_coors_loss']), \
                            float(self.ops['mean_exit_points_loss'])
                loss[0] += total_loss
                loss[1] += edge_loss
                loss[2] += kl_loss
                loss[3] += symbol_loss
                loss[4] += pos_loss
                loss[5] += exit_loss
                # if_step_and_zero_grad = ((step + 1) % acc_steps == 0)
                if if_step_and_zero_grad:
                    print(
                            "batch %i (has %i graphs). Loss so far: %.4f. Edge loss: %.4f, KL loss: %.4f, Node symbol loss: %.4f, Position loss: %.4f, Exit loss: %.4f"
                        % (step, num_graphs, loss[0]/(step+1), loss[1]/(step+1), loss[2]/(step+1), loss[3]/(step+1), loss[4]/(step+1), loss[5]/(step+1)))
                    # % (step, num_graphs, total_loss, edge_loss, kl_loss, symbol_loss, pos_loss))
                # self.optimize_step(if_step_and_zero_grad)  # back-propagation to update parameters
        else:
            with torch.no_grad():
                for step, batch_data in enumerate(batch_dataset):
                    self.load_current_batch_as_tensor(batch_data)  # load a batch data
                    num_graphs = self.data['num_graphs']
                    self.encoder()  # encode the data
                    self.compute_loss()  # compute the loss from encodings
                    # print('Valid: Total loss: %f | Edge loss: %f, KL loss: %f, Node symbol loss: %f' % (
                        # self.ops['total_loss'],
                        # self.ops['mean_edge_loss_in'],
                        # self.ops['mean_kl_loss_in'],
                        # self.ops['mean_node_symbol_loss_in']))
                    loss[0] += float(self.ops['total_loss'])
                    loss[1] += float(self.ops['mean_edge_loss_in'])
                    loss[2] += float(self.ops['mean_kl_loss_in'])
                    loss[3] += float(self.ops['mean_node_symbol_loss_in'])
                    loss[4] += float(self.ops['mean_node_coors_loss'])
                    loss[5] += float(self.ops['mean_exit_points_loss'])
                    print(
                            "batch %i (has %i graphs). Loss so far: %.4f. Edge loss: %.4f, KL loss: %.4f, Node symbol loss: %.4f, Position loss: %.4f, Exit loss: %.4f"
                        % (step, num_graphs, loss[0] / (step + 1),
                           loss[1] / (step + 1), loss[2] / (step + 1),
                           loss[3] / (step + 1), loss[4] / (step + 1), loss[5] / (step+1)))
        return loss / len(batch_dataset)
    def show_loss_curves(self, losses, names):
        fig, ax = plt.subplots()
        for i in range(losses.shape[0]):
            ax.plot(list(range((self.params['num_epochs']))), losses[i], label=names[i])
        ax.legend()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        plt.show()

    def save_model(self):
        log_dir = self.args.get('--log_dir') or '.'
        path = os.path.join(log_dir, "%s_model_%s.pickle" % (self.run_id, self.params['dataset']))
        weights_to_save = {}
        for var in self.weights.keys():
            weights_to_save[var] = self.weights[var]
        for var in self.units.keys():
            weights_to_save[var] = self.units[var]
        data_to_save = {'params': self.params, 'weights': weights_to_save}

        with open(path, 'wb') as out_file:
            pickle.dump(data_to_save, out_file, pickle.HIGHEST_PROTOCOL)

    def save_check_point(self, epoch):
        save_dir = self.params['check_point_path']
        path = os.path.join(save_dir, "%s_model_check_point_%s.pickle" % (self.run_id, self.params['dataset']))
        check_point = {'epoch': epoch, 'model_state_dict': self.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}
        torch.save(check_point,  path)

    def load_check_point(self, path: str):
        check_point = torch.load(path)
        self.load_state_dict(check_point['model_state_dict'])
        self.optimizer.load_state_dict(check_point['optimizer_state_dict'])
        self.init_epoch = check_point['epoch'] + 1

    def restore_model(self, path: str):
        print('Restoring weights and parameters from file %s.' %path)
        with open(path, 'rb') as in_file:
            data_to_load = pickle.load(in_file)

        weights_to_load = data_to_load['weights']
        params_to_load = data_to_load['params']
        self.params = params_to_load
        for var in self.weights.keys():
            self.weights[var] = weights_to_load[var]
        for var in self.units.keys():
            self.units[var] = weights_to_load[var]

    def make_model(self):
        raise Exception("Models have to implement make_model!")

    # make an iterator over batchs of data
    def make_minibatch_iterator(self, data, is_training):
        raise Exception("Models have to implement make_minibatch_iterator!")

    def process_raw_graphs(self, raw_data, is_training_data, file_name, bucket_sizes=None):
        raise Exception("Models have to implement process_raw_graphs!")

    def prepare_specific_graph_model(self) -> None:
        raise Exception("Models have to implement prepare_specific_graph_model!")

    def compute_final_node_representations(self, h, adj, scope_name):
        raise Exception("Models have to implement compute_final_node_representation!")

    def get_node_embedding_state(self, data, source):
        raise Exception("Models have to implement get_node_embedding_state!")

    # Load data of current batch to torch tensor
    def load_current_batch_as_tensor(self, batch_data):
        raise Exception("Models have to implement load_current_batch_as_tensor!")

    # Compute mean and log variance from graph representations
    def compute_mean_and_logvariance(self):
        raise Exception("Models have to implement compute_mean_and_logvariance!")

    # Sample a gaussian variable from mean and logvariance
    def sample_with_mean_and_logvariance(self):
        raise Exception("Models have to implement sample_with_mean_and_logvariance!")

    def compute_loss(self):
        raise Exception("Models have to implement compute_loss!")

    def optimize_step(self):
        raise Exception("Models have to implement optimize_step!")

    def set_zero_grad(self):
        raise Exception("Models have to implement set_zero_grad!")

    def generate_new_graphs(self, data):
        raise Exception("Models have to implement generate_new_graphs!")

