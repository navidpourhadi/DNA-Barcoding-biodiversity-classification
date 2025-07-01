import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader

from tqdm import tqdm
import os
import argparse
import json
import time

import matplotlib.pyplot as plt
from utils.utils import *
from utils.model import DNADeBruijnClassifier

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train a DNA De Bruijn Classifier")
    argparser.add_argument("--dataset", type=str, required=True, help="Path to the dataset file")
    argparser.add_argument("--k", type=int, default=4, help="Length of k-mers")
    argparser.add_argument("--positional_encoding_dim", type=int, default=128, help="Dimension of positional encoding")
    argparser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    argparser.add_argument("--num_epochs", type=int, default=25, help="Number of epochs for training")
    argparser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for the optimizer")
    argparser.add_argument("--num_workers", type=int, default=2, help="Number of workers for data loading")
    argparser.add_argument("--label_map", type=str, default="./outputs/label_map.json", help="Path to the label map file")
    argparser.add_argument("--best_params", type=str, default="./outputs/best_params.json", help="Path to the best params file")
    
    args = argparser.parse_args()

    dataset_path = args.dataset
    dataset_name = os.path.basename(dataset_path).split(".")[0]
    output_dir = "./outputs/best_"+dataset_name
    os.makedirs(output_dir, exist_ok=True)

    num_epochs = args.num_epochs
    n_workers = args.num_workers
    best_params = None
    if os.path.exists(args.label_map):
        with open(args.label_map, "r") as f:
            label_map = json.load(f)
            label_map = {k: int(v) for k, v in label_map.items()}
    else:
        label_map = None

    if os.path.exists(args.best_params):
        with open(args.best_params, "r") as f:
            best_params = json.load(f)
            lr = float(best_params['learning_rate'])
            batch_size = int(best_params['batch_size'])
            k = int(best_params['k'])
            positional_encoding_dim = int(best_params['positional_encoding_dim'])
            weight_decay = float(best_params['weight_decay'])
            gnn_hidden_layers = [int(x) for x in best_params['gnn_hidden_layers']]
            linear_hidden_layers = [int(x) for x in best_params['linear_hidden_layers']]
    else:
        raise ValueError("Best params file not found, can not set hyperparameters")

    # Load the dataset
    df, label_map = load_dataset(dataset_path, label_map)
    df = pre_process_dataset(df, k, positional_encoding_dim).reset_index(drop=True)
    
    num_classes = len(label_map.keys())

    # prepare the data for training
    print(f"Loaded {len(df)} sequences from {dataset_path}")
    print(f"label map: {label_map}")
    print(f"Number of classes: {num_classes}")
    print(f"Number of graphs: {len(df)}")

    train_loader = DataLoader(df["data"].tolist(), batch_size=batch_size, shuffle=True, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = DNADeBruijnClassifier(input_size= (k - 1) * 4, gnn_hidden_layers= gnn_hidden_layers, linear_hidden_layers= linear_hidden_layers, output_size= num_classes, edge_dim= positional_encoding_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    criterion = nn.CrossEntropyLoss()

    model.to(device)
    criterion.to(device)

    train_loss_history = []
    best_loss = float('inf')
    best_model = None
    for epoch in range(num_epochs):
        train_loss = 0
        model.train()  # Set model to training mode
        for batch_item in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):

            batch_item = batch_item.to(device)
            optimizer.zero_grad()
            
            y_pred = model(batch_item)

            loss = criterion(y_pred, batch_item.y)  # batch.y should be LongTensor of shape [batch_size]
            train_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            loss.backward()
            optimizer.step()
        
        lr_scheduler.step()

        train_loss_history.append(train_loss / len(train_loader))

        if train_loss <= best_loss:
            best_loss = train_loss
            best_model = model

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss / len(train_loader):.4f}")

    # plot the train_loss_history
    plt.plot(train_loss_history)
    plt.title(f"Train Loss History with best params {best_params}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"{output_dir}/train_loss_history.png")
    plt.close()


    # Save final model
    torch.save(model.state_dict(), output_dir + "/final_model.pt")
    torch.save(best_model.state_dict(), output_dir + "/best_final_model.pt")

    with open(output_dir + "/label_map.json", "w") as f:
        json.dump(label_map, f)
    
    # update the best params
    best_params['learning_rate'] = lr
    best_params['batch_size'] = batch_size
    best_params['k'] = k
    best_params['positional_encoding_dim'] = positional_encoding_dim
    best_params['weight_decay'] = weight_decay
    best_params['gnn_hidden_layers'] = gnn_hidden_layers
    best_params['linear_hidden_layers'] = linear_hidden_layers
    with open(output_dir + "/best_params.json", "w") as f:
        json.dump(best_params, f)

    print(f"Best params saved to {output_dir}/best_params.json")     
    print(f"Label map saved to {output_dir}/label_map.json")
    print(f"Final model trained and saved to {output_dir}/final_model.pt")