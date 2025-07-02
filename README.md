
# 🧬 DNA Barcoding with De Bruijn Graphs and GNNs

A PyTorch Geometric project that integrates **De Bruijn graph** representations with **direction-aware Graph Neural Networks (GNNs)** for accurate and scalable classification of DNA barcode sequences. Developed as part of the Deep Learning course at the University of Padova.

## 📘 Overview

This project proposes a novel framework that transforms DNA sequences into De Bruijn graphs and applies a directional GATv2-based GNN to classify species. It leverages:

- One-hot encoded k-mers as **node features**
- Sinusoidal **positional encoding** as edge features
- Dual-direction GAT layers for better **message passing**
- Global mean/sum/max pooling and MLP for **graph-level prediction**

The approach shows strong generalization and achieves high accuracy on diverse DNA barcode datasets such as **bats**, **birds**, **fishes**, **Inga**, and **Cypraeidae**.

## 📈 Key Results

| Dataset     | Accuracy | Precision | Recall | R² Score |
|-------------|----------|-----------|--------|----------|
| Bats        | 99.3%    | 0.996     | 0.996  | 0.988    |
| Fishes      | 99.1%    | 0.972     | 0.981  | 0.955    |
| Cypraeidae  | 94.3%    | 0.893     | 0.897  | 0.998    |
| Inga        | 88.4%    | 0.823     | 0.837  | 0.715    |
| Birds       | 86.4%    | 0.736     | 0.768  | 0.725    |

## 🧪 Datasets

Each dataset contains DNA barcode sequences labeled by species. The sequences vary in length and number of classes. De Bruijn graphs are constructed using different `k` values (3, 4, 5), with best results for `k = 5`.
The datasets can be downloaded from [here](http://dmb.iasi.cnr.it/supbarcodes.php)

Example dataset:
- **Bats:** 95 classes, 839 samples, sequence length: 657 bp

## 🧬 Graph Construction

- Nodes = unique (k−1)-mers  
- Edges = k-mer overlaps (directed)  
- Node Features = One-hot encoding (A, C, G, T)  
- Edge Features = Sinusoidal positional encoding  

## 🧠 Model Architecture

- **GNN Backbone:** 8 stacked GATv2 layers with forward/backward directional separation
- **Pooling:** Mean, Sum, Max (concatenated)
- **MLP Classifier:** Deep MLP with layers [512, 1024, 2048]
- **Optimizer:** AdamW  
- **Scheduler:** CosineAnnealingLR  
- **Loss:** CrossEntropy

## ⚙️ Hyperparameters

```python
learning_rate = 0.001
batch_size = 32
k = 5
positional_encoding_dim = 128
gnn_hidden_layers = [256] * 8
linear_hidden_layers = [512, 1024, 2048]
weight_decay = 1e-5
```

## 🚀 Installation

```bash
git clone git@github.com:navidpourhadi/DNA-Barcoding-biodiversity-classification.git
cd DNA-Barcoding-biodiversity-classification
pip install -r requirements.txt
```

## 🛠️ Usage

### Training
```bash
python train.py --dataset <dataset_name> --k 5 --positional_encoding_dim 128
```

### Testing
```bash
python test.py --model_path ./models/best_model.pth
```

### Available Datasets
Download the datasets from this [link](http://dmb.iasi.cnr.it/supbarcodes.php).
Place your datasets under `./data/`. Dataset names should match:
- `bats`
- `birds`
- `fishes`
- `Inga`
- `Cypraeidae`

## 📊 Visualization

Confusion matrices and accuracy plots for k-mer analysis are generated for detailed evaluation.

## 🔍 Citation

This project was developed as part of the Deep Learning course at:

**Università degli Studi di Padova**  
*Dipartimento di Ingegneria dell’Informazione*  
Author: **Navid Pourhadi Hasanabad**

## 📄 License

This project is open-source and freely available under the MIT License.
