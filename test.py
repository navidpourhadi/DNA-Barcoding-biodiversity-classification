import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader

import os
import argparse
import json

from utils.utils import *
from utils.model import DNADeBruijnClassifier
from sklearn.metrics import r2_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
import matplotlib.pyplot as plt

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train a DNA De Bruijn Classifier")
    argparser.add_argument("--dataset", type=str, required=True, help="Path to the dataset file")
    argparser.add_argument("--num_workers", type=int, default=2, help="Number of workers for data loading")
    argparser.add_argument("--label_map", type=str, default="./outputs/label_map.json", help="Path to the label map file")
    argparser.add_argument("--best_params", type=str, default="./outputs/best_params.json", help="Path to the best params file")
    argparser.add_argument("--weights", type=str, default="./outputs/model.pt", help="Path to the weights file")
    args = argparser.parse_args()

    dataset_path = args.dataset
    n_workers = args.num_workers
    model_path = args.weights

    # find the directory of model_path
    directory = os.path.dirname(model_path)

    if os.path.exists(args.label_map):
        with open(args.label_map, "r") as f:
            label_map = json.load(f)
            label_map = {k: int(v) for k, v in label_map.items()}
    else:
        raise ValueError("Label map file not found")

    if os.path.exists(args.best_params):
        with open(args.best_params, "r") as f:
            best_params = json.load(f)
            k = int(best_params['k'])
            positional_encoding_dim = int(best_params['positional_encoding_dim'])
            gnn_hidden_layers = [int(x) for x in best_params['gnn_hidden_layers']]
            linear_hidden_layers = [int(x) for x in best_params['linear_hidden_layers']]
    else:
        raise ValueError("Parameter's file not found, can not set hyperparameters")


    if not os.path.exists(model_path):
        raise ValueError("Model's file not found, can not load the model")

    # Load the dataset
    df, label_map = load_dataset(dataset_path, label_map)

    print(df.columns)
    print(df["specimen_id"].unique())

    # print df where specimen_id is nan
    print(df[df["specimen_id"].isna()])

    df = pre_process_dataset(df, k, positional_encoding_dim).reset_index(drop=True)
    
    num_classes = len(label_map.keys())

    # prepare the data for training
    print(f"Loaded {len(df)} sequences from {dataset_path}")
    print(f"label map: {label_map}")
    print(f"Number of classes: {num_classes}")

    test_loader = DataLoader(df["data"].tolist(), batch_size=8, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DNADeBruijnClassifier(input_size= (k - 1) * 4, gnn_hidden_layers= gnn_hidden_layers, linear_hidden_layers= linear_hidden_layers, output_size= num_classes, edge_dim= positional_encoding_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    criterion.to(device)


    # Validation loop
    model.eval()  # Set model to evaluation mode
    test_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []

    for test_batch in test_loader:
        test_batch = test_batch.to(device)

        test_output = model(test_batch)
        test_loss += criterion(test_output, test_batch.y).item()
        _, predicted = torch.max(test_output.data, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(test_batch.y.cpu().numpy())

        total += test_batch.y.size(0)
        correct += (predicted == test_batch.y).sum().item()
            
    test_loss /= len(test_loader)
    test_accuracy = 100 * correct / total  # Accuracy as a percentage
    r2 = r2_score(all_labels, all_preds)
    
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    precision_micro = precision_score(all_labels, all_preds, average='micro', zero_division=0)
    recall_micro = recall_score(all_labels, all_preds, average='micro', zero_division=0)


    print(f"Test Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_accuracy:.2f}%")
    print(f"R2 Score: {r2:.4f}")
    print(f"Precision (Macro): {precision_macro:.4f}, Recall (Macro): {recall_macro:.4f}")
    print(f"Precision (Micro): {precision_micro:.4f}, Recall (Micro): {recall_micro:.4f}")
    
    # save the results
    with open(f"{directory}/results.txt", "w") as f:
        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Accuracy: {test_accuracy:.2f}%\n")
        f.write(f"R2 Score: {r2:.4f}\n")
        f.write(f"Precision (Macro): {precision_macro:.4f}, Recall (Macro): {recall_macro:.4f}\n")
        f.write(f"Precision (Micro): {precision_micro:.4f}, Recall (Micro): {recall_micro:.4f}\n")


    class_indices = np.arange(num_classes)
    # Map index to class name using the same order
    index_to_species = {v: k for k, v in label_map.items()}
    ordered_species_names = [index_to_species[i] for i in class_indices]


    # creating the last epoch confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=class_indices)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ordered_species_names)
    fig, ax = plt.subplots(figsize=(15, 15))  # Adjust figsize as needed (width, height in inches)
    disp.plot(ax=ax, xticks_rotation=90)
    plt.title(f"Confusion Matrix with {r2:.4f} R2 Score, Accuracy: {test_accuracy:.2f}%, Loss: {test_loss:.4f}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{directory}/confusion_matrix.png")
    plt.close()


