import numpy as np
import pandas as pd
import os

from torch.utils.data import Dataset
import torch

def load_dataset(file_path: str, k, positional_encoding_dim) -> pd.DataFrame:

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    with open(file_path, 'r') as file:

        file_content = file.read()
        data = file_content.split('>')[1:]

        # for each data entry, split the first \n from the rest of the string
        data = [entry.split('\n', 1) for entry in data]
        data = [(entry[0].split("|")[1], entry[1].replace('\n', '')) for entry in data]
        
        df = pd.DataFrame(data, columns=['specimen', 'sequence'])
        df['node_features'], df['fwd_edges_index'], df['bwd_edges_index'], df['edges_attr'] = zip(*df['sequence'].apply(lambda x: de_bruijn_graph(x, k, positional_encoding_dim)))
        specimen_to_index = {specimen: index for index, specimen in enumerate(df['specimen'].unique())}
        df['specimen_id'] = df['specimen'].map(specimen_to_index)
        return df


def one_hot_encode(sequence: str) -> np.ndarray:
    nucleotide_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    indices = [nucleotide_to_index.get(nuc, -1) for nuc in sequence]
    one_hot = np.zeros((len(sequence), 4), dtype=int)
    valid = np.array(indices) >= 0
    one_hot[np.arange(len(sequence))[valid], np.array(indices)[valid]] = 1
    return one_hot.flatten()



def Kmers(sequence: str, k: int) -> list[str]:
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]


def sinusoidal_positional_encoding(position: int, d_model: int) -> np.ndarray:
    
    angle_rates = 1 / np.power(10000, (2 * (np.arange(d_model) // 2)) / np.float32(d_model))
    angle_rads = position * angle_rates
    pos_encoding = np.zeros((1, d_model))
    pos_encoding[0, 0::2] = np.sin(angle_rads[0::2])
    pos_encoding[0, 1::2] = np.cos(angle_rads[1::2])
    
    return pos_encoding


def de_bruijn_graph(sequence, k, positional_encoding_dim):

    # Pad the sequence so it’s divisible by k
    sequence = sequence + 'N' * (k - (len(sequence) % k))

    kmers = Kmers(sequence, k)

    edges = []
    node_set = set()

    for i in range(len(kmers) - 1):
        prefix = kmers[i][:-1]
        suffix = kmers[i + 1][:-1]
        node_set.update([prefix, suffix])
        pos_enc = sinusoidal_positional_encoding(i, positional_encoding_dim)[0]
        edges.append((prefix, suffix, pos_enc))

    # Assign an index to each unique (k-1)-mer
    node_list = sorted(node_set)
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}

    # Build edge_index and edge_attr
    src_indices = []
    dst_indices = []
    edge_attr = []
    for prefix, suffix, pos_enc in edges:
        src_indices.append(node_to_idx[prefix])
        dst_indices.append(node_to_idx[suffix])
        edge_attr.append(pos_enc)

        # src_indices.append(node_to_idx[suffix])
        # dst_indices.append(node_to_idx[prefix])
        # edge_dir.append(1)
        # edge_attr.append(pos_enc)

    fwd_edge_index = np.array([src_indices, dst_indices], dtype=np.int64)
    bwd_edge_index = np.array([dst_indices, src_indices], dtype=np.int64)    
    edge_attr = np.array(edge_attr, dtype=np.float32)

    node_features = np.array([one_hot_encode(node) for node in node_list], dtype=np.float32)

    return node_features, fwd_edge_index, bwd_edge_index, edge_attr


class GraphDataset(Dataset):
    def __init__(self, node_features, fwd_edges_index, bwd_edges_index, edges_attr, labels):
        self.node_features = node_features
        self.fwd_edges_index = fwd_edges_index
        self.bwd_edges_index = bwd_edges_index
        self.edges_attr = edges_attr

        self.labels = labels

    def __len__(self):
        return len(self.node_features)

    def __getitem__(self, idx):
        return {
            'node_features': self.node_features[idx],
            'fwd_edges_index': self.fwd_edges_index[idx],
            'bwd_edges_index': self.bwd_edges_index[idx],
            'edges_attr': self.edges_attr[idx],
            'label': self.labels[idx]
        }


def graph_collate_fn(batch):
    node_features = [item['node_features'] for item in batch]
    fwd_edges_index = [item['fwd_edges_index'] for item in batch]
    bwd_edges_index = [item['bwd_edges_index'] for item in batch]
    edges_attr = [item['edges_attr'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    return node_features, fwd_edges_index, bwd_edges_index, edges_attr, labels