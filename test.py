import torch
from torch_geometric.nn import MessagePassing

class DirectedMessagePassing(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # Use 'mean', 'max' etc. as needed
        self.lin_forward = torch.nn.Linear(in_channels, out_channels)
        self.lin_backward = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_dir):
        """
        x         : Node features, shape [num_nodes, in_channels]
        edge_index: [2, num_edges] tensor, COO format
        edge_dir  : [num_edges] tensor. 0 = forward (source → target), 1 = backward (target → source)
        """
        return self.propagate(edge_index, x=x, edge_dir=edge_dir)

    def message(self, x_j, edge_dir):
        """
        x_j      : Features of source nodes (j indices from edge_index)
        edge_dir : Direction label for each edge
        """

        msg_forward = self.lin_forward(x_j)
        msg_backward = self.lin_backward(x_j)
        msg = torch.where(edge_dir.unsqueeze(-1) == 0, msg_forward, msg_backward)
        return msg

    def update(self, aggr_out):
        return aggr_out



x = torch.tensor([[1.], [2.], [3.], [4.]])  # 4 nodes with 1D features
edge_index = torch.tensor([[0, 2, 3], [1, 3, 2]])  # Edges: 0→1, 2→3, 3→2
edge_dir = torch.tensor([0, 0, 1])  # Last edge is backward, others forward

layer = DirectedMessagePassing(in_channels=1, out_channels=2)
out = layer(x, edge_index, edge_dir)
print(out.shape)  # [num_nodes, out_channels]
