import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool, global_add_pool, BatchNorm, GCNConv, GINEConv
from torch.nn import Parameter, Linear, LeakyReLU, Sequential
from torch_geometric.utils import softmax


class GATmodel(nn.Module):
    def __init__(self, input_size, hidden_sizes, edge_dim, heads = 4):
        super(GATmodel, self).__init__()

        self.FWDGATLayer = nn.ModuleList()
        self.BWDGATLayer = nn.ModuleList()
        self.MergeMLP = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Initial GINEConv layers for forward and backward graphs
        self.FWDGATLayer.append(GATv2Conv(input_size, hidden_sizes[0], edge_dim=edge_dim, heads=heads, concat=False))
        self.BWDGATLayer.append(GATv2Conv(input_size, hidden_sizes[0], edge_dim=edge_dim, heads=heads, concat=False))
        self.MergeMLP.append(nn.Sequential(
            nn.Linear(2 * hidden_sizes[0], hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[0])
        ))
        self.batch_norms.append(BatchNorm(hidden_sizes[0]))

        # Additional layers for each hidden size
        for i in range(1, len(hidden_sizes)):
            self.FWDGATLayer.append(GATv2Conv(hidden_sizes[i-1], hidden_sizes[i], edge_dim=edge_dim, heads=heads, concat=False))
            self.BWDGATLayer.append(GATv2Conv(hidden_sizes[i-1], hidden_sizes[i], edge_dim=edge_dim, heads=heads, concat=False))
        
            self.MergeMLP.append(nn.Sequential(
                nn.Linear(2 * hidden_sizes[i], hidden_sizes[i]),
                nn.ReLU(),
                nn.Linear(hidden_sizes[i], hidden_sizes[i])
            ))
            self.batch_norms.append(BatchNorm(hidden_sizes[i]))
        
    def forward(self, x, fwd_edges_index, bwd_edges_index, edge_attr):
        for layer_idx in range(len(self.FWDGATLayer)):
            # print(layer_idx)
            fwd_x = self.FWDGATLayer[layer_idx](x, fwd_edges_index, edge_attr)
            bwd_x = self.BWDGATLayer[layer_idx](x, bwd_edges_index, edge_attr)

            x = torch.cat([fwd_x, bwd_x], dim=1)  

            x = self.MergeMLP[layer_idx](x)
            x = self.batch_norms[layer_idx](x)
            x = F.relu(x)
        
        return x


class LinearNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(LinearNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        self.layers.append(nn.ReLU())
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.layers.append(nn.BatchNorm1d(hidden_sizes[i]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
        x = self.layers[-1](x)
        return x


class DNADeBruijnClassifier(nn.Module):
    def __init__(self, input_size, gnn_hidden_layers, linear_hidden_layers, output_size, edge_dim):
        super(DNADeBruijnClassifier, self).__init__()
        self.gnn = GATmodel(input_size, gnn_hidden_layers, edge_dim)
        self.global_mean_pool = global_mean_pool
        self.global_add_pool = global_add_pool
        self.global_max_pool = global_max_pool
        self.linear = LinearNN(3 * gnn_hidden_layers[-1], linear_hidden_layers, output_size)
      

    def forward(self, data_batch):
        node_features = data_batch.x

        src_id = data_batch.edge_index[0]
        dst_id = data_batch.edge_index[1]
        fwd_edges_index = torch.stack([src_id, dst_id], dim=0)
        bwd_edges_index = torch.stack([dst_id, src_id], dim=0)
        edge_attr = data_batch.edge_attr

        gnn_out = self.gnn(node_features, fwd_edges_index, bwd_edges_index, edge_attr)  # (num_nodes, output_size)
        global_mean_out = self.global_mean_pool(gnn_out, data_batch.batch)
        global_add_out = self.global_add_pool(gnn_out, data_batch.batch)
        global_max_out = self.global_max_pool(gnn_out, data_batch.batch)
        global_out = torch.cat([global_mean_out, global_add_out, global_max_out], dim=1)
        linear_out = self.linear(global_out)

        return linear_out
