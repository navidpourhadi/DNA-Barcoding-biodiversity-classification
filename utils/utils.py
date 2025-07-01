import numpy as np
import pandas as pd
import os

from torch_geometric.data import Data
import torch

from typing import Tuple, Optional


def load_dataset(file_path:str, specimen_map: Optional[dict] = None) -> Tuple[pd.DataFrame, dict]: 
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    with open(file_path, 'r') as file:

        file_content = file.read()
        data = file_content.split('>')[1:]

        # for each data entry, split the first \n from the rest of the string
        data = [entry.split('\n', 1) for entry in data]
        data = [(entry[0].split("|")[1], entry[1].replace('\n', '').replace('-', '')) for entry in data]
        df = pd.DataFrame(data, columns=['specimen', 'sequence'])
        if specimen_map is not None:
            df['specimen_id'] = df['specimen'].map(specimen_map)
        else:
            specimen_map = {specimen: index for index, specimen in enumerate(df['specimen'].unique())}
            df['specimen_id'] = df['specimen'].map(specimen_map)
        # print(df['specimen_id'].dtypes)
        # filter rows with specimen_id equal to nan
        df = df[df['specimen_id'].notna()]
    return df, specimen_map


def pre_process_dataset(df: pd.DataFrame, k: int, positional_encoding_dim: int) -> pd.DataFrame:
    df['data'] = df.apply(lambda row: de_bruijn_graph(row['sequence'], k, positional_encoding_dim, row['specimen_id']), axis=1)
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


def de_bruijn_graph(sequence, k, positional_encoding_dim, y):

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


    fwd_edge_index = np.array([src_indices, dst_indices], dtype=np.int64)
    edge_attr = np.array(edge_attr, dtype=np.float32)

    node_features = np.array([one_hot_encode(node) for node in node_list], dtype=np.float32)
    new_graph = Data(x=torch.tensor(node_features, dtype=torch.float32), edge_index=torch.tensor(fwd_edge_index, dtype=torch.long), edge_attr=torch.tensor(edge_attr, dtype=torch.float32), y=torch.tensor(y, dtype=torch.long))
    return new_graph
