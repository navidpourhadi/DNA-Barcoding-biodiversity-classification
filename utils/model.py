import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, MessagePassing, GATConv, BatchNorm, GCNConv, GINEConv
from torch.nn import Parameter, Linear, LeakyReLU, Sequential
from torch_geometric.utils import softmax



class GNNAttentionResidualModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, heads=2):
        super(GNNAttentionResidualModel, self).__init__()

        self.FWDGATLayers = nn.ModuleList()
        self.BWDGATLayers = nn.ModuleList()
        self.MergeMLP = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.residual_transform = nn.ModuleList()

        self.FWDGATLayers.append(GATConv(input_size, hidden_sizes[0], heads=heads, concat=True))
        self.BWDGATLayers.append(GATConv(input_size, hidden_sizes[0], heads=heads, concat=True))
        self.MergeMLP.append(Linear(2 * hidden_sizes[0] * heads, hidden_sizes[0] * heads))
        self.batch_norms.append(BatchNorm(hidden_sizes[0] * heads))
        self.residual_transform.append(nn.Linear(input_size, hidden_sizes[0] * heads))

        for i in range(1, len(hidden_sizes)):
            self.FWDGATLayers.append(GATConv(hidden_sizes[i-1] * heads, hidden_sizes[i], heads=heads, concat=True))
            self.BWDGATLayers.append(GATConv(hidden_sizes[i-1] * heads, hidden_sizes[i], heads=heads, concat=True))
        
            self.MergeMLP.append(Linear(2 * hidden_sizes[i] * heads, hidden_sizes[i] * heads))
            
            self.batch_norms.append(BatchNorm(hidden_sizes[i] * heads))        

            self.residual_transform.append(nn.Linear(hidden_sizes[i-1] * heads, hidden_sizes[i] * heads))
        
    def forward(self, x, fwd_edges_index, bwd_edges_index, edge_attr):
        
        for layer_idx in range(len(self.FWDGATLayers)):
            x_initial = x
            fwd_x = self.FWDGATLayers[layer_idx](x, fwd_edges_index, edge_attr)  
            bwd_x = self.BWDGATLayers[layer_idx](x, bwd_edges_index, edge_attr)
            x = torch.cat([fwd_x, bwd_x], dim=1)  # Concatenate the outputs of the forward and backward GAT layers
            x = self.MergeMLP[layer_idx](x)  # Apply the linear layer to merge the outputs
            x = F.relu(x)
            x = self.batch_norms[layer_idx](x)
        
            # Residual connection
            if x_initial.shape[1] != x.shape[1]:
                x_residual = self.residual_transform[layer_idx](x_initial)
            else:
                x_residual = x_initial
            x = x + x_residual

        return x


class GCNResidualModel(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(GCNResidualModel, self).__init__()

        self.FWDGCNLayers = nn.ModuleList()
        self.BWDGCNLayers = nn.ModuleList()
        self.MergeMLP = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.residual_transform = nn.ModuleList()

        self.FWDGCNLayers.append(GCNConv(input_size, hidden_sizes[0]))
        self.BWDGCNLayers.append(GCNConv(input_size, hidden_sizes[0]))
        self.MergeMLP.append(Linear(2 * hidden_sizes[0], hidden_sizes[0]))
        self.batch_norms.append(BatchNorm(hidden_sizes[0]))
        self.residual_transform.append(nn.Linear(input_size, hidden_sizes[0]))

        for i in range(1, len(hidden_sizes)):
            self.FWDGCNLayers.append(GCNConv(hidden_sizes[i-1], hidden_sizes[i]))
            self.BWDGCNLayers.append(GCNConv(hidden_sizes[i-1], hidden_sizes[i]))
        
            self.MergeMLP.append(Linear(2 * hidden_sizes[i], hidden_sizes[i]))
            
            self.batch_norms.append(BatchNorm(hidden_sizes[i]))        

            self.residual_transform.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        
    def forward(self, x, fwd_edges_index, bwd_edges_index, edge_attr):
        
        for layer_idx in range(len(self.FWDGCNLayers)):
            x_initial = x
            fwd_x = self.FWDGCNLayers[layer_idx](x, fwd_edges_index)  
            bwd_x = self.BWDGCNLayers[layer_idx](x, bwd_edges_index)
            x = torch.cat([fwd_x, bwd_x], dim=1)  # Concatenate the outputs of the forward and backward GAT layers
            x = self.MergeMLP[layer_idx](x)  # Apply the linear layer to merge the outputs
            x = F.relu(x)
            x = self.batch_norms[layer_idx](x)
        
            # Residual connection
            if x_initial.shape[1] != x.shape[1]:
                x_residual = self.residual_transform[layer_idx](x_initial)
            else:
                x_residual = x_initial
            x = x + x_residual

        return x


def make_mlp(input_dim, hidden_dim):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim)
    )


class GINEResidualModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, edge_dim):
        super(GINEResidualModel, self).__init__()

        self.FWDGINELayers = nn.ModuleList()
        self.BWDGINELayers = nn.ModuleList()
        self.MergeMLP = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Initial GINEConv layers for forward and backward graphs
        self.FWDGINELayers.append(GINEConv(make_mlp(input_size, hidden_sizes[0]), edge_dim=edge_dim))
        self.BWDGINELayers.append(GINEConv(make_mlp(input_size, hidden_sizes[0]), edge_dim=edge_dim))
        self.MergeMLP.append(nn.Linear(2 * hidden_sizes[0], hidden_sizes[0]))
        self.batch_norms.append(BatchNorm(hidden_sizes[0]))

        # Additional layers for each hidden size
        for i in range(1, len(hidden_sizes)):
            self.FWDGINELayers.append(GINEConv(make_mlp(hidden_sizes[i-1], hidden_sizes[i]), edge_dim=edge_dim))
            self.BWDGINELayers.append(GINEConv(make_mlp(hidden_sizes[i-1], hidden_sizes[i]), edge_dim=edge_dim))
        
            self.MergeMLP.append(nn.Linear(2 * hidden_sizes[i], hidden_sizes[i]))
            self.batch_norms.append(BatchNorm(hidden_sizes[i]))
        
    def forward(self, x, fwd_edges_index, bwd_edges_index, edge_attr):
        for layer_idx in range(len(self.FWDGINELayers)):
            # Apply forward and backward GINEConv layers
            fwd_x = self.FWDGINELayers[layer_idx](x, fwd_edges_index, edge_attr)
            bwd_x = self.BWDGINELayers[layer_idx](x, bwd_edges_index, edge_attr)

            # Concatenate forward and backward results
            x = torch.cat([fwd_x, bwd_x], dim=1)  
            # Apply the merge MLP
            x = self.MergeMLP[layer_idx](x)
            x = self.batch_norms[layer_idx](x)
            x = F.leaky_relu(x)
        
        return x


class LinearNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(LinearNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x


class DNADeBruijnClassifier(nn.Module):
    def __init__(self, input_size, gnn_hidden_layers, linear_hidden_layers, output_size, edge_dim = None, num_heads = 2):
        super(DNADeBruijnClassifier, self).__init__()
        self.gnn = GNNAttentionResidualModel(input_size, gnn_hidden_layers, heads=num_heads)
        self.global_pool = global_mean_pool
        self.linear = LinearNN(gnn_hidden_layers[-1] * num_heads, linear_hidden_layers, output_size)
      
        
    def forward(self, node_features, fwd_edges_index, bwd_edges_index, edge_attr):
        
        predictions = []
        for i in range(len(node_features)):

            gnn_out = self.gnn(node_features[i], fwd_edges_index[i], bwd_edges_index[i], edge_attr[i])  # (num_nodes, output_size)

            # making a dummy batch for global pooling
            batch = torch.zeros(node_features[i].size(0), dtype=torch.long, device=node_features[i].device)
            global_out = self.global_pool(gnn_out, batch)

            linear_out = self.linear(global_out)  # (num_samples, output_size)

            predictions.append(linear_out)

        out =  torch.stack(predictions) # (num_samples, num_nodes, output_size)
        out = out.view(out.size(0), -1, out.size(-1)).mean(dim=1)
        return out


class DNADeBruijnClassifier2(nn.Module):
    def __init__(self, input_size, gnn_hidden_layers, linear_hidden_layers, output_size, edge_dim):
        super(DNADeBruijnClassifier2, self).__init__()
        self.gnn = GINEResidualModel(input_size, gnn_hidden_layers, edge_dim)
        self.global_pool = global_mean_pool
        self.linear = LinearNN(gnn_hidden_layers[-1], linear_hidden_layers, output_size)
      
        
    def forward(self, node_features, fwd_edges_index, bwd_edges_index, edge_attr):
        
        predictions = []
        for i in range(len(node_features)):

            gnn_out = self.gnn(node_features[i], fwd_edges_index[i], bwd_edges_index[i], edge_attr[i])  # (num_nodes, output_size)

            # making a dummy batch for global pooling
            batch = torch.zeros(node_features[i].size(0), dtype=torch.long, device=node_features[i].device)
            global_out = self.global_pool(gnn_out, batch)

            linear_out = self.linear(global_out)  # (num_samples, output_size)

            predictions.append(linear_out)

        out =  torch.stack(predictions) # (num_samples, num_nodes, output_size)
        out = out.view(out.size(0), -1, out.size(-1)).mean(dim=1)
        return out


