import numpy as np
import pandas as pd
import os

from torch_geometric.data import Data
import torch

from typing import Tuple, Optional
from utils.utils import *
import argparse

# return the statistics for the graphs of the dataset (number of nodes, number of edges, node feature size, edge feature size)
# def statistics(df: pd.DataFrame, k: int, positional_encoding_dim: int):

if __name__ == "__main__":
    # argparser = argparse.ArgumentParser(description="Train a DNA De Bruijn Classifier")
    # argparser.add_argument("--dataset", type=str, required=True, help="Path to the dataset file")
    # argparser.add_argument("--k", type=int, default=4, help="Length of k-mers")
    # argparser.add_argument("--positional_encoding_dim", type=int, default=128, help="Dimension of positional encoding")
    
    # args = argparser.parse_args()
        
    datasets = {"train": {"bats": "./dataset/bats/batsTrain.fas",
                          "birds": "./dataset/birds/ATrain.fas",
                          "fishes": "./dataset/fishes/fishesTrain.fas",
                          "Inga": "./dataset/Inga/IngaTrain.fasta",
                          "Cypraeidae": "./dataset/Cypraeidae/CypraeidaeTrain.fasta"}, 
                "test": {"bats": "./dataset/bats/batsTest.fas",
                        "birds": "./dataset/birds/ATest.fas",
                        "fishes": "./dataset/fishes/fishesTest.fas",
                        "Inga": "./dataset/Inga/IngaTest.fasta",
                        "Cypraeidae": "./dataset/Cypraeidae/CypraeidaeTest.fasta"}}

    k = [3, 4, 5]

    positional_encoding_dim = 128

    for dataset in list(datasets["train"].values())[1:2]:
    

        # Load the dataset
        df_ref, label_map = load_dataset(dataset)
        
        for i in k:
            df = pre_process_dataset(df_ref, i, positional_encoding_dim)

            print(f"Dataset: {dataset}, k:{i}")
            print(f"Average node number: {df['data'].apply(lambda x: x.num_nodes).mean()}")
            print(f"Average edge number: {df['data'].apply(lambda x: x.num_edges).mean()}")
