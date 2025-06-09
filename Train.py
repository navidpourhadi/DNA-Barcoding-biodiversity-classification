import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch_geometric.data import Data
# from torch_geometric.loader import DataLoader

from sklearn.model_selection import train_test_split
import os
import argparse
from utils.utils import *
from utils.model import DNADeBruijnClassifier2
from sklearn.utils.class_weight import compute_class_weight

argparser = argparse.ArgumentParser(description="Train a DNA De Bruijn Classifier")
argparser.add_argument("--dataset", type=str, required=True, help="Path to the dataset file")
argparser.add_argument("--k", type=int, default=4, help="Length of k-mers")
argparser.add_argument("--positional_encoding_dim", type=int, default=64, help="Dimension of positional encoding")
argparser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
argparser.add_argument("--num_epochs", type=int, default=25, help="Number of epochs for training")
argparser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for the optimizer")

args = argparser.parse_args()

dataset_path = args.dataset
k = args.k
positional_encoding_dim = args.positional_encoding_dim
batch_size = args.batch_size
num_epochs = args.num_epochs
lr = args.learning_rate


# Load the dataset
df = load_dataset(dataset_path, k, positional_encoding_dim)
print(f"Loaded {len(df)} sequences from {dataset_path}")

# prepare the data for training
species_label = df['specimen'].unique()
num_classes = len(species_label)
species_to_index = {species: index for index, species in enumerate(species_label)}

df['label'] = df['specimen'].map(species_to_index)

df['node_features'] = df['node_features'].apply(lambda x: torch.tensor(x, dtype=torch.float32))
df['fwd_edges_index'] = df['fwd_edges_index'].apply(lambda x: torch.tensor(x, dtype=torch.long))
df['bwd_edges_index'] = df['bwd_edges_index'].apply(lambda x: torch.tensor(x, dtype=torch.long))
df['edges_attr'] = df['edges_attr'].apply(lambda x: torch.tensor(x, dtype=torch.float32))
df['specimen_id'] = df['specimen_id'].apply(lambda x: torch.tensor(x, dtype=torch.long))
df['label'] = df['label'].apply(lambda x: torch.tensor(x, dtype=torch.long))

print(f"Number of classes: {num_classes}")
print(f"Number of graphs: {len(df)}")

# Split the dataset into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

train_dataset = GraphDataset(train_df["node_features"], train_df["fwd_edges_index"], train_df["bwd_edges_index"], train_df["edges_attr"], train_df["label"])
val_dataset = GraphDataset(val_df["node_features"], val_df["fwd_edges_index"], val_df["bwd_edges_index"], val_df["edges_attr"], val_df["label"])


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=graph_collate_fn)


model = DNADeBruijnClassifier2(input_size= (k - 1) * 4, gnn_hidden_layers= [64, 128], linear_hidden_layers= [256], output_size= num_classes, edge_dim= positional_encoding_dim)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.5, min_lr=1e-5)

criterion = nn.CrossEntropyLoss()


for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    for node_features, fwd_edges_index, bwd_edges_index, edges_attr, y in train_loader:

        optimizer.zero_grad()
        
        # Forward pass for training
        y_pred = model(node_features, fwd_edges_index, bwd_edges_index, edges_attr)

        # Compute loss and backpropagate
        loss = criterion(y_pred, y)  # batch.y should be LongTensor of shape [batch_size]
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
    
    # Validation loop
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradients for validation
        
        val_node_features = val_df['node_features'].tolist()
        val_fwd_edges_index = val_df['fwd_edges_index'].tolist()
        val_bwd_edges_index = val_df['bwd_edges_index'].tolist()
        val_edges_attr = val_df['edges_attr'].tolist()
        val_y = torch.tensor(val_df['label'].tolist(), dtype=torch.long)

        # val_node_features, val_fwd_edges_index, val_edges_attr, val_y = graph_collate_fn(val_dataset)
        # Forward pass for validation
        val_output = model(val_node_features, val_fwd_edges_index, val_bwd_edges_index, val_edges_attr)

        # Compute validation loss
        val_loss = criterion(val_output, val_y)  # batch.y should be LongTensor of shape [batch_size]
        lr_scheduler.step(val_loss)

        # Compute accuracy (if needed)
        _, predicted = torch.max(val_output, 1)
        print(predicted)
        total += val_y.size(0)
        correct += (predicted == val_y).sum().item()

    # Print validation results for this epoch
    val_accuracy = 100 * correct / total  # Accuracy as a percentage
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")