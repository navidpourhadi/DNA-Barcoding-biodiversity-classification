import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
import json
import time

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
        

    output_dir = "./outputs/Train"+str(time.time())
    os.makedirs(output_dir, exist_ok=True)

    dataset_path = args.dataset
    k = args.k
    positional_encoding_dim = args.positional_encoding_dim
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    lr = args.learning_rate
    n_workers = args.num_workers
    weight_decay = 1e-5
    gnn_hidden_layers = [256, 256, 256, 256, 256, 256, 256]
    linear_hidden_layers = [512, 1024, 1024]


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
        best_params = {}

    # Load the dataset
    df, label_map = load_dataset(dataset_path, label_map)
    df = pre_process_dataset(df, k, positional_encoding_dim)
    df = df[:100]
    num_classes = len(label_map.keys())

    # prepare the data for training
    print(f"Loaded {len(df)} sequences from {dataset_path}")
    print(f"label map: {label_map}")
    print(f"Number of classes: {num_classes}")
    print(f"Number of graphs: {len(df)}")

    # Split the dataset into training and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_loader = DataLoader(train_df["data"].tolist(), batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_df["data"].tolist(), batch_size=batch_size, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = DNADeBruijnClassifier(input_size= (k - 1) * 4, gnn_hidden_layers= [256, 256, 256, 256, 256, 256, 256], linear_hidden_layers= [512, 1024, 1024, 2048], output_size= num_classes, edge_dim= positional_encoding_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5, min_lr=1e-5)

    criterion = nn.CrossEntropyLoss()

    model.to(device)
    criterion.to(device)

    train_loss_history = []
    val_loss_history = []
    val_accuracy_history = []
    val_r2_history = []
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

        train_loss_history.append(train_loss / len(train_loader))
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss / len(train_loader):.4f}")
        
        # Validation loop
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []

        for val_batch in val_loader:
            val_batch = val_batch.to(device)

            val_output = model(val_batch)
            val_loss += criterion(val_output, val_batch.y).item()
            _, predicted = torch.max(val_output.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(val_batch.y.cpu().numpy())

            total += val_batch.y.size(0)
            correct += (predicted == val_batch.y).sum().item()
            
        val_loss /= len(val_loader)
        val_loss_history.append(val_loss)
        val_accuracy = 100 * correct / total  # Accuracy as a percentage
        
        val_accuracy_history.append(val_accuracy)
        val_r2_history.append(r2_score(all_labels, all_preds))

        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
        lr_scheduler.step(val_loss)

        if epoch == num_epochs - 1:
            class_indices = np.arange(num_classes)
            # Map index to class name using the same order
            index_to_species = {v: k for k, v in label_map.items()}
            ordered_species_names = [index_to_species[i] for i in class_indices]


            # creating the last epoch confusion matrix
            cm = confusion_matrix(all_labels, all_preds, labels=class_indices)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ordered_species_names)
            fig, ax = plt.subplots(figsize=(15, 15))  # Adjust figsize as needed (width, height in inches)
            disp.plot(ax=ax, xticks_rotation=90)
            plt.title(f"Confusion Matrix - Epoch {epoch+1}")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/confusion_matrix_epoch_{epoch+1}.png")
            plt.close()

    # creating plot for accuracy score of validation
    plt.plot(val_accuracy_history)
    plt.title(f"Validation Accuracy Score")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(f"{output_dir}/accuracy_score.png")
    plt.close()

    # creating plot for loss score of validation
    plt.plot(val_loss_history, color='orange')
    plt.plot(train_loss_history, color='blue')
    plt.title(f"Loss Score")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"{output_dir}/loss_score.png")
    plt.close()

    # creating plot for R2 score of the validation
    plt.plot(val_r2_history)
    plt.title(f"R2 Score")
    plt.xlabel("Epoch")
    plt.ylabel("R2")
    plt.savefig(f"{output_dir}/r2_score.png")
    plt.close()



    


    # Save final model
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), output_dir + "/final_model.pt")

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