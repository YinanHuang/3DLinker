import numpy as np
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit import Chem
from collections import defaultdict, deque
from rdkit.Chem import rdmolops
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from rdkit.Chem import rdFMCS
import torch
from torch import nn
from rdkit.Geometry import Point3D
from copy import deepcopy
SMALL_NUMBER = 1e-7
LARGE_NUMBER = 1e10

# bond mapping
bond_dict = {'SINGLE': 0, 'DOUBLE': 1, 'TRIPLE': 2, 'AROMATIC': 3}
number_to_bond= {0: Chem.rdchem.BondType.SINGLE, 1:Chem.rdchem.BondType.DOUBLE,
                 2: Chem.rdchem.BondType.TRIPLE, 3:Chem.rdchem.BondType.AROMATIC}

def dataset_info(dataset):  # qm9, zinc, cep
    if dataset == 'qm9':
        return {'atom_types': ["H", "C", "N", "O", "F"],
                'maximum_valence': {0: 1, 1: 4, 2: 3, 3: 2, 4: 1},
                'number_to_atom': {0: "H", 1: "C", 2: "N", 3: "O", 4: "F"},
                'bucket_sizes': np.array(list(range(4, 28, 2)) + [29])
                }
    elif dataset == 'zinc':
        return {'atom_types': ['Br1(0)', 'C4(0)', 'Cl1(0)', 'F1(0)', 'H1(0)', 'I1(0)',
                               'N2(-1)', 'N3(0)', 'N4(1)', 'O1(-1)', 'O2(0)', 'S2(0)', 'S4(0)', 'S6(0)'],
                'maximum_valence': {0: 1, 1: 4, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 3, 8: 4, 9: 1, 10: 2, 11: 2, 12: 4,
                                    13: 6, 14: 3},
                'number_to_atom': {0: 'Br', 1: 'C', 2: 'Cl', 3: 'F', 4: 'H', 5: 'I', 6: 'N', 7: 'N', 8: 'N', 9: 'O',
                                   10: 'O', 11: 'S', 12: 'S', 13: 'S'},
                'bucket_sizes': np.array(
                    [28, 31, 33, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 53, 55, 58, 84])
                }

    elif dataset == "cep":
        return {'atom_types': ["C", "S", "N", "O", "Se", "Si"],
                'maximum_valence': {0: 4, 1: 2, 2: 3, 3: 2, 4: 2, 5: 4},
                'number_to_atom': {0: "C", 1: "S", 2: "N", 3: "O", 4: "Se", 5: "Si"},
                'bucket_sizes': np.array([25, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 43, 46])
                }
    else:
        print("the datasets in use are qm9|zinc|cep")
        exit(1)


def check_adjacent_sparse(adj_list, node, neighbor_in_doubt):
    for neighbor, edge_type in adj_list[node]:
        if neighbor == neighbor_in_doubt:
            return True, edge_type
    return False, None


def bfs_distance(start, adj_list, is_dense=False):
    distances={}
    visited=set()
    queue=deque([(start, 0)])
    visited.add(start)
    while len(queue) != 0:
        current, d=queue.popleft()
        for neighbor, edge_type in adj_list[current]:
            if neighbor not in visited:
                distances[neighbor]=d+1
                visited.add(neighbor)
                queue.append((neighbor, d+1))
    return [(start, node, d) for node, d in distances.items()]


# generate a new feature on whether adding the edges will generate more than two overlapped edges for rings
def get_overlapped_edge_feature(edge_mask, color, new_mol):
    overlapped_edge_feature=[]
    for node_in_focus, neighbor in edge_mask:
        if color[neighbor] == 1:
            # attempt to add the edge
            new_mol.AddBond(int(node_in_focus), int(neighbor), number_to_bond[0])
            # Check whether there are two cycles having more than two overlap edges
            try:
                ssr = Chem.GetSymmSSSR(new_mol)
            except:
                ssr = []
            overlap_flag = False
            for idx1 in range(len(ssr)):
                for idx2 in range(idx1+1, len(ssr)):
                    if len(set(ssr[idx1]) & set(ssr[idx2])) > 2:
                        overlap_flag=True
            # remove that edge
            new_mol.RemoveBond(int(node_in_focus), int(neighbor))
            if overlap_flag:
                overlapped_edge_feature.append((node_in_focus, neighbor))
    return overlapped_edge_feature


def get_initial_valence(node_symbol, dataset):
    return [dataset_info(dataset)['maximum_valence'][s] for s in node_symbol]


def add_atoms(new_mol, node_symbol, dataset):
    for number in node_symbol:
        if dataset=='qm9' or dataset=='cep':
            idx=new_mol.AddAtom(Chem.Atom(dataset_info(dataset)['number_to_atom'][number]))
        elif dataset=='zinc':
            new_atom = Chem.Atom(dataset_info(dataset)['number_to_atom'][number])
            charge_num=int(dataset_info(dataset)['atom_types'][number].split('(')[1].strip(')'))
            new_atom.SetFormalCharge(charge_num)
            new_mol.AddAtom(new_atom)


# sample node symbols based on node predictions
def sample_node_symbol(all_node_symbol_prob, all_lengths, dataset):
    all_node_symbol=[]
    for graph_idx, graph_prob in enumerate(all_node_symbol_prob):
        node_symbol=[]
        for node_idx in range(all_lengths[graph_idx]):
            symbol=np.random.choice(np.arange(len(dataset_info(dataset)['atom_types'])), p=graph_prob[node_idx])
            node_symbol.append(symbol)
        all_node_symbol.append(node_symbol)
    return all_node_symbol


def graph_to_adj_mat(graph, max_n_vertices, num_edge_types, tie_fwd_bkwd=True, considering_edge_type=True):
    if considering_edge_type:
        amat = np.zeros((num_edge_types, max_n_vertices, max_n_vertices))
        for src, e, dest in graph:
            add_edge_mat(amat, src, dest, e)
    else:
        amat = np.zeros((max_n_vertices, max_n_vertices))
        for src, e, dest in graph:
            add_edge_mat(amat, src, dest, e, considering_edge_type=False)
    return amat


# add one edge to adj matrix
def add_edge_mat(amat, src, dest, e, considering_edge_type=True):
    if considering_edge_type:
        amat[e, dest, src] = 1
        amat[e, src, dest] = 1
    else:
        amat[src, dest] = 1
        amat[dest, src] = 1


def node_keep_to_dense(nodes_to_keep, maximum_vertice_num):
    s=[0]*maximum_vertice_num
    for node in nodes_to_keep:
        s[node]=1
    return s

def incre_adj_mat_to_dense(incre_adj_mat, num_edge_types, maximum_vertice_num):
    new_incre_adj_mat=[]
    for sparse_incre_adj_mat in incre_adj_mat:
        dense_incre_adj_mat=np.zeros((num_edge_types, maximum_vertice_num,maximum_vertice_num))
        for current, adj_list in sparse_incre_adj_mat.items():
            for neighbor, edge_type in adj_list:
                dense_incre_adj_mat[edge_type][current][neighbor]=1
        new_incre_adj_mat.append(dense_incre_adj_mat)
    return new_incre_adj_mat


def adj_list_to_dense(adj_list, num_edge_types, maximum_vertice_num):
    adj_mat = np.zeros([num_edge_types, maximum_vertice_num, maximum_vertice_num])
    for current, adj_neighbors in adj_list.items():
        for neighbor, edge_type in adj_neighbors:
            adj_mat[edge_type][current][neighbor] = 1
    return adj_mat


def distance_to_others_dense(distance_to_others, maximum_vertice_num):
    new_all_distance=[]
    for sparse_distances in distance_to_others:
        dense_distances=np.zeros((maximum_vertice_num), dtype=int)
        for x, y, d in sparse_distances:
            dense_distances[y]=d
        new_all_distance.append(dense_distances)
    return new_all_distance

def overlapped_edge_features_to_dense(overlapped_edge_features, maximum_vertice_num):
    new_overlapped_edge_features=[]
    for sparse_overlapped_edge_features in overlapped_edge_features:
        dense_overlapped_edge_features=np.zeros((maximum_vertice_num), dtype=int)
        for node_in_focus, neighbor in sparse_overlapped_edge_features:
            dense_overlapped_edge_features[neighbor]=1
        new_overlapped_edge_features.append(dense_overlapped_edge_features)
    return new_overlapped_edge_features  # [number_iteration, maximum_vertice_num]

def node_sequence_to_dense(node_sequence,maximum_vertice_num):
    new_node_sequence=[]
    for node in node_sequence:
        s=[0]*maximum_vertice_num
        s[node]=1
        new_node_sequence.append(s)
    return new_node_sequence

def edge_type_masks_to_dense(edge_type_masks, maximum_vertice_num, num_edge_types):
    new_edge_type_masks=[]
    for mask_sparse in edge_type_masks:
        mask_dense=np.zeros([num_edge_types, maximum_vertice_num])
        for node_in_focus, neighbor, bond in mask_sparse:
            mask_dense[bond][neighbor]=1
        new_edge_type_masks.append(mask_dense)
    return new_edge_type_masks

def edge_type_labels_to_dense(edge_type_labels, maximum_vertice_num,num_edge_types):
    new_edge_type_labels=[]
    for labels_sparse in edge_type_labels:
        labels_dense=np.zeros([num_edge_types, maximum_vertice_num])
        for node_in_focus, neighbor, bond in labels_sparse:
            labels_dense[bond][neighbor]= 1/float(len(labels_sparse)) # fix the probability bug here.
        new_edge_type_labels.append(labels_dense)
    return new_edge_type_labels


def edge_masks_to_dense(edge_masks, maximum_vertice_num):
    new_edge_masks=[]
    for mask_sparse in edge_masks:
        mask_dense=[0] * maximum_vertice_num
        for node_in_focus, neighbor in mask_sparse:
            mask_dense[neighbor]=1
        new_edge_masks.append(mask_dense)
    return new_edge_masks


def edge_labels_to_dense(edge_labels, maximum_vertice_num):
    new_edge_labels=[]
    for label_sparse in edge_labels:
        label_dense=[0] * maximum_vertice_num
        for node_in_focus, neighbor in label_sparse:
            label_dense[neighbor]=1/float(len(label_sparse))
        new_edge_labels.append(label_dense)
    return new_edge_labels


def glorot_init(shape):
    initialization_range = np.sqrt(6.0 / (shape[-2] + shape[-1]))
    return torch.tensor(np.random.uniform(low=-initialization_range, high=initialization_range, size=shape).astype(np.float32))



def scalar_neuron(input, weight, bias, activation=torch.nn.SiLU()):
    output_shape = list(input.size())
    output_shape[-1] = weight.size(1)
    input = input.reshape([-1, input.size(-1)])
    output = torch.matmul(input, weight) + bias
    output = activation(output)
    return output.reshape(output_shape)

def vector_neuron(input, Q_weight, K_weight):
    output_shape = list(input.size())
    output_shape[-2] = Q_weight.size(1)
    input = input.reshape([-1, input.size(-2), input.size(-1)])
    input = torch.transpose(input, -1, -2)
    # output = torch.matmul(input, weight)
    Q = torch.matmul(input, Q_weight)
    K = torch.matmul(input, K_weight)
    inner_product = torch.einsum('nic,  nic->nc', Q, K)
    inner_product = torch.unsqueeze(inner_product * (inner_product < 0), dim=1)
    k_norm = torch.linalg.norm(K, dim=1)
    k_norm = torch.unsqueeze(k_norm, dim=1) + SMALL_NUMBER
    output = Q - inner_product * K / torch.square(k_norm)
    output = torch.transpose(output, -1, -2)
    return output.reshape(output_shape)

def vector_neuron_leaky(input, Q_weight, K_weight, alpha=0.3):
    output_shape = list(input.size())
    output_shape[-2] = Q_weight.size(1)
    input = input.reshape([-1, input.size(-2), input.size(-1)])
    input = torch.transpose(input, -1, -2)
    # output = torch.matmul(input, weight)
    Q = torch.matmul(input, Q_weight)
    K = torch.matmul(input, K_weight)
    inner_product = torch.einsum('nic,  nic->nc', Q, K)
    inner_product = torch.unsqueeze(inner_product * (inner_product < 0), dim=1)
    k_norm = torch.linalg.norm(K, dim=1)
    k_norm = torch.unsqueeze(k_norm, dim=1) + SMALL_NUMBER
    output = Q - inner_product * K / torch.square(k_norm)
    output = torch.transpose(output, -1, -2)
    input = torch.transpose(input, -1, -2)
    return alpha * input.reshape(output_shape) + (1 - alpha) * output.reshape(output_shape)

class vector_unit(torch.nn.Module):
    def __init__(self, v_dim, alpha=0.3):
        super(vector_unit, self).__init__()
        self.v_dim = v_dim
        self.alpha = alpha
        self.Q = nn.Linear(v_dim, v_dim, bias=False)
        self.K = nn.Linear(v_dim, v_dim, bias=False)
    def forward(self, v):
        output_shape = list(v.size())
        output_shape[-2] = self.v_dim
        v = v.reshape([-1, v.size(-2), v.size(-1)])
        v = torch.transpose(v, -1, -2)
        # output = torch.matmul(input, weight)
        Q = self.Q(v)
        K = self.K(v)
        inner_product = torch.einsum('nic,  nic->nc', Q, K)
        inner_product = torch.unsqueeze(inner_product * (inner_product < 0), dim=1)
        k_norm = torch.linalg.norm(K, dim=1)
        k_norm = torch.unsqueeze(k_norm, dim=1) + SMALL_NUMBER
        output = Q - inner_product * K / torch.square(k_norm)
        output = torch.transpose(output, -1, -2)
        input = torch.transpose(v, -1, -2)
        return self.alpha * input.reshape(output_shape) + (1 - self.alpha) * output.reshape(output_shape)



class vector_cross_unit(torch.nn.Module):
    def __init__(self, v_dim):
        super(vector_cross_unit, self).__init__()
        self.v_dim = v_dim
        self.W = nn.Linear(v_dim, v_dim, bias=False)
    def forward(self, v):
        output_shape = list(v.size())
        output_shape[-2] = self.v_dim
        v = v.reshape([-1, v.size(-2), v.size(-1)])
        v = torch.transpose(v, -1, -2)
        W = self.W(v)
        W = torch.transpose(W, -1, -2)
        v = torch.transpose(v, -1, -2)
        W = W.view(output_shape)
        v = v.view(output_shape)
        return torch.cross(W, v)

class vector_MLP(nn.Module):
    def __init__(self, in_dim, out_dim, alpha=0.3):
        super(vector_MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha = alpha
        self.vector_neuron = vector_unit(in_dim)
        self.output_weight = nn.Linear(in_dim, out_dim, bias=False)
    def forward(self, v):
        v = self.vector_neuron(v)
        output_shape = list(v.size())
        output_shape[-2] = self.out_dim
        input = v.reshape([-1, v.size(-2), v.size(-1)])
        input = torch.transpose(input, -1, -2)
        output = self.output_weight(input)
        output = torch.transpose(output, -1, -2)
        return output.reshape(output_shape)

def vector_linear(input, weight):
    output_shape = list(input.size())
    output_shape[-2] = weight.size(1)
    input = input.reshape([-1, input.size(-2), input.size(-1)])
    input = torch.transpose(input, -1, -2)
    output = torch.matmul(input, weight)
    output = torch.transpose(output, -1, -2)
    return output.reshape(output_shape)


def fully_connected(input, hidden_weight, hidden_bias, output_weight, activation=torch.nn.SiLU()):
    output = scalar_neuron(input, hidden_weight, hidden_bias, activation)
    output = torch.matmul(output, output_weight)
    return output

class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, if_prob=False):
        super(MLP, self).__init__()
        self.linear_h1 = nn.Linear(in_dim, in_dim)
        self.nonlinear_h1 = nn.SiLU()
        self.linear_h2 = nn.Linear(in_dim, in_dim)
        self.nonlinear_h2 = nn.SiLU()
        self.linear_out = nn.Linear(in_dim, out_dim)
        self.if_prob = if_prob
    def forward(self, x):
        x = self.nonlinear_h1(self.linear_h1(x))
        x = self.nonlinear_h2(self.linear_h2(x))
        x = self.linear_out(x)
        if self.if_prob:
            x = torch.nn.Softmax(dim=-1)(x)
        return x

def fully_connected_vec(vec, non_linear_Q, non_linear_K, output_weight, activation='leaky_relu'):
    if activation == 'leaky_relu':
        hidden = vector_neuron_leaky(vec, non_linear_Q, non_linear_K)
    else:
        hidden = vector_neuron(vec, non_linear_Q, non_linear_K)
    output = vector_linear(hidden, output_weight)
    return output

def compute_edge_type_probs(edge_type_logits):
    return torch.nn.Softmax(dim=1)(edge_type_logits)


def compute_edge_probs(edge_logits):
    return torch.nn.Softmax(dim=1)(edge_logits)

def str2float(mylist):
    float_list = []
    for item in mylist:
        float_list.append([float(item[0]), float(item[1])])
    return float_list


def length(mylist):
    if mylist is None:
        return 0
    else:
        return len(mylist)

# Get length for each graph based on node masks
def get_graph_length(all_node_mask):
    all_lengths = []
    for graph in all_node_mask:
        if 0 in graph:
            length = np.argmin(graph)
        else:
            length = len(graph)
        all_lengths.append(length)
    return all_lengths

# standard normal with shape [a1, a2, a3]
def generate_std_normal(a1, a2, a3):
    return torch.normal(0, 1, [a1, a2, a3])


def get_idx_of_largest_frag(frags):
    return np.argmax([len(frag) for frag in frags])


def remove_extra_nodes(new_mol):
    frags=Chem.rdmolops.GetMolFrags(new_mol)
    while len(frags) > 1:
        # Get the idx of the frag with largest length
        largest_idx = get_idx_of_largest_frag(frags)
        for idx in range(len(frags)):
            if idx != largest_idx:
                # Remove one atom that is not in the largest frag
                new_mol.RemoveAtom(frags[idx][0])
                break
        frags=Chem.rdmolops.GetMolFrags(new_mol)

def if_valid(new_mol, exit_points):
    frags = Chem.rdmolops.GetMolFrags(new_mol)
    largest_idx = get_idx_of_largest_frag(frags)
    return exit_points[0] in frags[largest_idx] and exit_points[1] in frags[largest_idx]
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

# select the best based on shapes and probs
def select_best(all_mol):
    # sort by shape
    all_mol=sorted(all_mol)
    best_shape=all_mol[-1][0]
    all_mol=[(p, m) for s, p, m in all_mol if s==best_shape]
    # sort by probs
    all_mol=sorted(all_mol)
    return all_mol[-1][1]


def dump(file_name, content):
    with open(file_name, 'wb') as out_file:
        pickle.dump(content, out_file, pickle.HIGHEST_PROTOCOL)

def need_kekulize(mol):
    for bond in mol.GetBonds():
        if bond_dict[str(bond.GetBondType())] >= 3:
            return True
    return False

def to_graph_mol(mol, dataset):
    if mol is None:
        return [], []
    # Kekulize it
    if need_kekulize(mol):
        rdmolops.Kekulize(mol)
        if mol is None:
            return None, None
    # remove stereo information, such as inward and outward edges
    Chem.RemoveStereochemistry(mol)

    edges = []
    nodes = []
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        begin_idx, end_idx = min(begin_idx, end_idx), max(begin_idx, end_idx)
        if mol.GetAtomWithIdx(begin_idx).GetAtomicNum() == 0 or mol.GetAtomWithIdx(end_idx).GetAtomicNum() == 0:
            continue
        else:
            edges.append((begin_idx, bond_dict[str(bond.GetBondType())], end_idx))
            assert bond_dict[str(bond.GetBondType())] != 3
    for atom in mol.GetAtoms():
        if dataset=='qm9' or dataset=="cep":
            nodes.append(onehot(dataset_info(dataset)['atom_types'].index(atom.GetSymbol()), len(dataset_info(dataset)['atom_types'])))
        elif dataset=='zinc': # transform using "<atom_symbol><valence>(<charge>)"  notation
            symbol = atom.GetSymbol()
            valence = atom.GetTotalValence()
            charge = atom.GetFormalCharge()
            atom_str = "%s%i(%i)" % (symbol, valence, charge)

            if atom_str not in dataset_info(dataset)['atom_types']:
                if "*" in atom_str:
                    continue
                else:
                    # print('unrecognized atom type %s' % atom_str)
                    return [], []

            nodes.append(onehot(dataset_info(dataset)['atom_types'].index(atom_str), len(dataset_info(dataset)['atom_types'])))

    return nodes, edges


def onehot(idx, len):
    z = [0 for _ in range(len)]
    z[idx] = 1
    return z


def compute_3d_coors(mol, random_seed=0):
    mol = Chem.AddHs(mol)
    success = AllChem.EmbedMolecule(mol, randomSeed=random_seed)
    if success == -1:
        return 0, 0
    mol = Chem.RemoveHs(mol)
    c = mol.GetConformer(0)
    pos = c.GetPositions()
    return pos, 1

def compute_3d_coors_multiple(mol, numConfs=20, maxIters=400, randomSeed=1):
    # mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol, numConfs=numConfs, numThreads=0, randomSeed=randomSeed)
    if mol.GetConformers() == ():
        return 0, 0
    result = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=maxIters, numThreads=0)
    mol = Chem.RemoveHs(mol)
    result = [tuple((result[i][0], result[i][1], i)) for i in range(len(result)) if result[i][0] == 0]
    if result == []: # no local minimum on energy surface is found
        return 0, 0
    result.sort()
    return mol.GetConformers()[result[0][-1]].GetPositions(), 1

def compute_3d_coors_frags(mol, numConfs=20, maxIters=400, randomSeed=1):
    du = Chem.MolFromSmiles('*')
    clean_frag = Chem.RemoveHs(AllChem.ReplaceSubstructs(Chem.MolFromSmiles(Chem.MolToSmiles(mol)),du,Chem.MolFromSmiles('[H]'),True)[0])
    frag = Chem.CombineMols(clean_frag, Chem.MolFromSmiles("*.*"))
    mol_to_link_carbon = AllChem.ReplaceSubstructs(mol, du, Chem.MolFromSmiles('C'), True)[0]
    pos, _ = compute_3d_coors_multiple(mol_to_link_carbon, numConfs, maxIters, randomSeed)


    return pos




def re_index(array, re_idx):
    array_re_idx = np.zeros_like(array)
    for i in range(len(re_idx)):
        array_re_idx[i] = array[re_idx[i]]
    return array_re_idx

def positions_padding(pos, pad_size):
    pos = np.array(pos)
    n = pos.shape[0]
    if pad_size > n:
        delta_n = pad_size - n
        pos = np.concatenate((pos, np.zeros([delta_n, 3])), axis=0)
    return pos

def adj_list_padding(adj_list, max_size, real_size):
    adj_list_pad = deepcopy(adj_list)
    for _ in range(max_size - real_size):
        adj_list_pad.append(defaultdict(list))
    return adj_list_pad

def incre_adj_list_to_adj_mat(adj_in, incre_adj_list, num_edge_type):
    max_iter = len(incre_adj_list)
    num_nodes = adj_in.shape[1]
    incre_adj_mat = np.zeros([max_iter, num_edge_type, num_nodes, num_nodes])
    current_adj_mat = np.copy(adj_in)
    for i, item in enumerate(incre_adj_list):
        current_adj_mat = assign_adj_list(item, current_adj_mat)
        incre_adj_mat[i] = current_adj_mat
    return incre_adj_mat

def assign_adj_list(adj_list, adj_mat):
    for current, edge in adj_list.items():
        for neighbor, edge_type in edge:
            adj_mat[edge_type][current][neighbor] = 1
    return adj_mat

def rearrange(re_idx_pre, re_idx_pos):
    temp = []
    for i in range(len(re_idx_pre)):
        temp.append(re_idx_pre[re_idx_pos[i]])
    return temp


def pairwise_construct(x, repeat_num):
    x_self = torch.unsqueeze(x, dim=2)
    x_self = x_self.repeat(1, 1, repeat_num, 1)
    x_others = torch.unsqueeze(x, dim=1)
    x_others = x_others.repeat(1, repeat_num, 1, 1)
    return x_self, x_others


def swish(x):
    return x * torch.nn.Sigmoid()(x)

def entropy(p):
    return torch.einsum('btv, btv->', p, torch.log(p + SMALL_NUMBER)) / p.size(0)

def show_graph(adj):
    G = nx.Graph()
    for b in range(adj.shape[0]):
        for i in range(adj.shape[1]):
            for j in range(i, adj.shape[2]):
                if adj[b, i, j] == 1:
                    G.add_edge(i, j)
    nx.draw_networkx(G)
    plt.show()
    return G

def show_adj_list(adj_list):
    G = nx.Graph()
    for i in range(len(adj_list)):
        G.add_edge(adj_list[i][0], adj_list[i][2])
    nx.draw_networkx(G)
    plt.show()
    return G

class GRUCell_vec(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCell_vec, self).__init__()
        self.weights_hidden_z = torch.nn.Parameter(glorot_init([hidden_size, input_size]))
        self.weights_input_z = torch.nn.Parameter(glorot_init([input_size, input_size]))
        self.biases_z = torch.nn.Parameter(torch.zeros([1, input_size], dtype=torch.float32))
        self.weights_hidden_r = torch.nn.Parameter(glorot_init([hidden_size, input_size]))
        self.weights_input_r = torch.nn.Parameter(glorot_init([input_size, input_size]))
        self.biases_r = torch.nn.Parameter(torch.zeros([1, input_size], dtype=torch.float32))
        self.weights_hidden_v = torch.nn.Parameter(glorot_init([hidden_size, input_size]))
        self.weights_input_v = torch.nn.Parameter(glorot_init([input_size, input_size]))
        self.weights_nonlinear_Q = torch.nn.Parameter(glorot_init([input_size, input_size]))
        self.weights_nonlinear_K = torch.nn.Parameter(glorot_init([input_size, input_size]))

    def forward(self, v, m_v):
        z = vector_linear(m_v, self.weights_hidden_z) + vector_linear(v, self.weights_input_z)
        z = torch.nn.Sigmoid()(torch.linalg.norm(z, dim=-1) + self.biases_z)
        z = torch.unsqueeze(z, dim=-1)
        r = vector_linear(m_v, self.weights_hidden_r) + vector_linear(v, self.weights_input_r)
        r = torch.nn.Sigmoid()(torch.linalg.norm(r, dim=-1) + self.biases_r)
        r = torch.unsqueeze(r, dim=-1)
        delta_v = vector_linear(m_v, self.weights_hidden_v) + vector_linear(r * v,
                                                                            self.weights_input_v)
        delta_v = vector_neuron_leaky(delta_v, self.weights_nonlinear_Q, self.weights_nonlinear_K)
        return (1 - z) * v + z * delta_v

class gated_regression(torch.nn.Module):
    def __init__(self, latent_h_dim, latent_v_dim):
        super(gated_regression, self).__init__()
        latent_dim = latent_h_dim + latent_v_dim
        self.U = nn.Parameter(glorot_init([latent_v_dim, latent_v_dim]))
        self.V = nn.Parameter(glorot_init([latent_v_dim, latent_v_dim]))
        self.hidden_1 = nn.Parameter(glorot_init([latent_dim, latent_dim]))
        self.biases_1 = nn.Parameter(torch.zeros([1, latent_dim]))
        self.weight_1 = nn.Parameter(glorot_init([latent_dim, 1]))
        self.hidden_2 = nn.Parameter(glorot_init([latent_dim, latent_dim]))
        self.biases_2 = nn.Parameter(torch.zeros([1, latent_dim]))
        self.weight_2 = nn.Parameter(glorot_init([latent_dim, 1]))

    def forward(self, h, v, mask):
        Uv = torch.linalg.norm(torch.einsum('vu, bnui->bnvi', self.U, v), dim=-1)
        Vv = torch.linalg.norm(torch.einsum('vu, bnui->bnvi', self.V, v), dim=-1)
        feature1 = torch.cat([h, Uv], dim=-1)
        feature2 = torch.cat([h, Vv], dim=-1)
        out1 = torch.nn.Sigmoid()(fully_connected(feature1, self.hidden_1, self.biases_1, self.weight_1))
        out2 = fully_connected(feature2, self.hidden_2, self.biases_2, self.weight_2)
        return torch.nn.Sigmoid()(torch.sum(out1 * out2 * mask, dim=[1, 2]))



def find_two_frags_with_idx(adj, exit_points, device):
    num_graphs = adj.shape[0]
    num_nodes = adj.shape[-1]
    mask = torch.zeros([num_graphs, 2, num_nodes, 1], device=device)
    for b in range(num_graphs):
        G = nx.convert_matrix.from_numpy_matrix(adj[b])
        G_sub = nx.algorithms.components.connected_components(G)
        i = 0
        reverse = exit_points[b][0] > exit_points[b][1]
        for c in sorted(G_sub):
            mask[b, i] = torch.tensor(nodes_idx_to_mask(np.array(G.subgraph(c).nodes), num_nodes), dtype=torch.float32,
                                      device=device)
            i += 1
            if i == 2:
                break
        if reverse:
            mask[b, 0], mask[b, 1] = mask[b, 1].clone(), mask[b, 0].clone()
    return mask


def generate_exit_mask(exit_points, num_nodes):
    num_graphs = exit_points.size(0)
    mask = torch.zeros([num_graphs, num_nodes, 1], device=exit_points.device)
    mask[torch.arange(num_graphs), exit_points[:, 0].type(torch.long)] = 1
    mask[torch.arange(num_graphs), exit_points[:, 1].type(torch.long)] = 1
    return mask

def nodes_idx_to_mask(nodes_idx, num_nodes):
    mask = np.zeros([num_nodes, 1])
    mask[nodes_idx] = 1
    return mask

def topology_from_rdkit(rdkit_molecule):
    Chem.rdmolops.Kekulize(rdkit_molecule, clearAromaticFlags=True)
    topology = nx.Graph()
    for atom in rdkit_molecule.GetAtoms():
        # Add the atoms as nodes
        topology.add_node(atom.GetIdx(), atom_type=atom.GetAtomicNum())

        # Add the bonds as edges
    for bond in rdkit_molecule.GetBonds():
        topology.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type=bond.GetBondType())

    return topology

def topology_from_graph(graphs):
    number_to_atom =  {0: 'Br', 1: 'C', 2: 'Cl', 3: 'F', 4: 'H', 5: 'I', 6: 'N', 7: 'N', 8: 'N', 9: 'O',
                       10: 'O', 11: 'S', 12: 'S', 13: 'S'}
    topology = nx.Graph()
    for i, node in enumerate(graphs['node_features_in']):
        # Add the atoms as nodes
        topology.add_node(i, atom_type=Chem.Atom(number_to_atom[np.argmax(np.array(node))]).GetAtomicNum())

        # Add the bonds as edges
    for src, bond, des in graphs['graph_in']:
        topology.add_edge(src, des, bond_type=number_to_bond[bond])

    return topology

def permutation(node, edge, GM):
    for i, m in enumerate(GM.subgraph_isomorphisms_iter()):
        if i == 0:
            mapping = m
            break
    len_graph = len(node)
    len_frag = len(mapping)
    start_idx = len_frag
    for i in range(len_graph):
        if i not in mapping:
            mapping[i] = start_idx
            start_idx += 1
    mapping_reverse = dict()
    for i in mapping:
        mapping_reverse[mapping[i]] = i
    node = deepcopy(np.array(node))
    node_p = deepcopy(node)
    edge_p = deepcopy(edge)
    for i in range(len(mapping)):
        node_p[mapping[i]] = node[i]
    for i, (src, bond, des) in enumerate(edge):
        if src in mapping:
            edge_p[i] = (mapping[src], edge_p[i][1], edge_p[i][2])
        if des in mapping:
            edge_p[i] = (edge_p[i][0], edge_p[i][1], mapping[des])
    return node_p.tolist(), edge_p


def node_match(n1, n2):
    return n1['atom_type'] == n2['atom_type']

def edge_match(e1, e2):
    return e1['bond_type'] == e2['bond_type']

def tensor_product(v1, v2):
    # v: b * n * c * 3
    # return v otimes v: b * n * n * c * 3 * 3
    repeat_num = v1.size(1)
    v_self = torch.unsqueeze(v1, dim=2)
    v_self = v_self.repeat(1, 1, repeat_num, 1, 1)
    v_others = torch.unsqueeze(v2, dim=1)
    v_others = v_others.repeat(1, repeat_num, 1, 1, 1)
    v_self = torch.unsqueeze(v_self, dim=-1)
    v_others = torch.unsqueeze(v_others, dim=-2)
    return torch.matmul(v_self, v_others)


def construct_fake_gt(smi):
    # construct a fake ground truth for fragment's coordinates generation by RDKit
    ori_mol = Chem.MolFromSmiles(smi)
    edmol = Chem.RWMol(Chem.MolFromSmiles(''))
    exits = []
    for atom in ori_mol.GetAtoms():
        edmol.AddAtom(Chem.Atom(atom.GetAtomicNum()))
        if atom.GetAtomicNum() == 0:
            exits.append(atom.GetIdx())
    for bond in ori_mol.GetBonds():
        edmol.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType())
    edmol.AddBond(exits[0], exits[1], Chem.BondType.SINGLE)
    for exit in exits:
        edmol.GetAtomWithIdx(exit).SetAtomicNum(6)
    return Chem.MolToSmiles(edmol.GetMol())
