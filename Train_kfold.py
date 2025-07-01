import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, accuracy_score
from itertools import product
from tqdm import tqdm
import os
import argparse
import time
import json
import copy

from utils.utils import load_dataset, pre_process_dataset
from utils.model import DNADeBruijnClassifier

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train DNA De Bruijn Classifier with K-Fold Cross-Validation")
    argparser.add_argument("--dataset", type=str, required=True, help="Path to the dataset file")
    argparser.add_argument("--k", type=int, default=4, help="Length of k-mers")
    argparser.add_argument("--positional_encoding_dim", type=int, default=128, help="Dimension of positional encoding")
    argparser.add_argument("--num_epochs", type=int, default=25, help="Number of epochs")
    argparser.add_argument("--num_wokers", type=int, default=2, help="Number of workers")
    argparser.add_argument("--n_splits", type=int, default=5, help="Number of folds for K-Fold CV")
    argparser.add_argument("--label_map", type=str, default="./outputs/label_map.json", help="Path to the label map file")

    args = argparser.parse_args()

    # Dataset and basic config
    dataset_path = args.dataset
    k = args.k
    pos_dim = args.positional_encoding_dim
    num_epochs = args.num_epochs
    n_splits = args.n_splits
    n_workers = args.num_wokers

    if os.path.exists(args.label_map):
        with open(args.label_map, "r") as f:
            label_map = json.load(f)
            label_map = {k: int(v) for k, v in label_map.items()}
    else:
        label_map = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and preprocess data
    df_ref, label_map = load_dataset(dataset_path, label_map)
    num_classes = len(label_map.keys())

    # Define hyperparameter grid
    hyperparams_grid = {
        'learning_rate': [0.01, 0.001, 0.0001],
        'batch_size': [16, 32],
        'k': [3, 4, 5],
        'positional_encoding_dim': [64, 128],
        'weight_decay': [
            1e-6,
            1e-5,
            1e-4,
        ],
        'gnn_hidden_layers': [
            [256, 256, 256, 256, 256, 256, 256, 256],
            [256, 256, 512, 512, 1028, 1028, 1028, 1028]
        ],
        'linear_hidden_layers': [
            [512, 1024, 2048],
            [1024, 2048]
        ],
    }

    param_names = list(hyperparams_grid.keys())
    param_values = list(hyperparams_grid.values())
    param_combinations = list(product(*param_values))

    best_avg_acc = 0.0
    best_params = None
    best_model_state = None

    for combo in param_combinations:
        params = dict(zip(param_names, combo))
        print(f"\nEvaluating Hyperparameters: {params}")
        fold_accuracies = []
    
        # creating the preprocessed dataset with the current hyperparameters of k and positional_encoding_dim
        df = pre_process_dataset(copy.deepcopy(df_ref), params['k'], params['positional_encoding_dim'])

        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kf.split(df, df['specimen_id'])):
            print(f"\nFold {fold + 1}/{n_splits}")

            train_df = df.iloc[train_idx].reset_index(drop=True)
            val_df = df.iloc[val_idx].reset_index(drop=True)

            train_loader = DataLoader(train_df["data"].tolist(), batch_size=params['batch_size'], shuffle=True, num_workers=n_workers)
            val_loader = DataLoader(val_df["data"].tolist(), batch_size=params['batch_size'], shuffle=False, num_workers=n_workers)

            model = DNADeBruijnClassifier(
                input_size=(k - 1) * 4,
                gnn_hidden_layers=params['gnn_hidden_layers'],
                linear_hidden_layers=params['linear_hidden_layers'],
                output_size=num_classes,
                edge_dim=pos_dim
            ).to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
            criterion = nn.CrossEntropyLoss().to(device)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
            # Training
            model.train()
            for epoch in range(num_epochs):
                total_loss = 0
                for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    output = model(batch)
                    loss = criterion(output, batch.y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    total_loss += loss.item()
                lr_scheduler.step()
                print(f"Training loss:{total_loss / len(train_loader):.4f}")    

            # Validation
            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    output = model(batch)
                    preds = torch.argmax(output, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch.y.cpu().numpy())

            acc = accuracy_score(all_labels, all_preds)
            print(f"Fold {fold+1} Accuracy: {acc * 100:.2f}%")
            fold_accuracies.append(acc)

        avg_acc = np.mean(fold_accuracies)
        print(f"\nAverage Accuracy for {params}: {avg_acc * 100:.2f}%")

        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            best_params = params
            best_model_state = model.state_dict()

    # Save best model
    print(f"\nBest Params: {best_params} with Accuracy: {best_avg_acc * 100:.2f}%")

    print("\nStarting final training on full dataset with best hyperparameters...")

    df = pre_process_dataset(copy.deepcopy(df_ref), best_params['k'], best_params['positional_encoding_dim'])

    full_loader = DataLoader(
        df["data"].tolist(),
        batch_size=best_params['batch_size'],
        shuffle=True,
        num_workers=n_workers
    )

    final_model = DNADeBruijnClassifier(
        input_size=(k - 1) * 4,
        gnn_hidden_layers=best_params['gnn_hidden_layers'],
        linear_hidden_layers=best_params['linear_hidden_layers'],
        output_size=num_classes,
        edge_dim=best_params['positional_encoding_dim']
    ).to(device)

    optimizer = torch.optim.AdamW(
        final_model.parameters(),
        lr=best_params['learning_rate'],
        weight_decay=best_params['weight_decay']
    )
    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(num_epochs):
        final_model.train()
        total_loss = 0
        for batch in tqdm(full_loader, desc=f"[Final Train] Epoch {epoch + 1}/{num_epochs}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            output = final_model(batch)
            loss = criterion(output, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"[Final Train] Epoch {epoch + 1} Loss: {total_loss / len(full_loader):.4f}")

    # Save final model
    output_dir = "./outputs/Train"+time.time()
    os.makedirs(output_dir, exist_ok=True)
    torch.save(final_model.state_dict(), output_dir + "/final_model.pt")
    with open(output_dir + "/label_map.json", "w") as f:
        json.dump(label_map, f)
    with open(output_dir + "/best_params.json", "w") as f:
        json.dump(best_params, f)

    print(f"Best params saved to {output_dir}/best_params.json")     
    print(f"Label map saved to {output_dir}/label_map.json")
    print(f"Final model trained and saved to {output_dir}/final_model.pt")