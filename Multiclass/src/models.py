import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

# NN model
class NN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_action_nodes=7):
        super().__init__()
        torch.manual_seed(1234)
        self.num_action_nodes = num_action_nodes
        self.rt_meas_dim = in_dim  # Store rt_meas_dim for use in predict_prob
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.5)  # Increased dropout
        self.out_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        batch_size = x.size(0)
        h = self.lin1(x).relu()  # (batch_size, num_action_nodes, in_dim) -> (batch_size, num_action_nodes, hidden_dim)
        h = h.view(-1, h.size(-1))  # (batch_size * num_action_nodes, hidden_dim)
        h = self.bn1(h)  # (batch_size * num_action_nodes, hidden_dim)
        h = h.view(batch_size, self.num_action_nodes, -1)  # (batch_size, num_action_nodes, hidden_dim)
        h = self.dropout(h)
        h = self.lin2(h).relu()  # (batch_size, num_action_nodes, hidden_dim)
        h = h.view(-1, h.size(-1))  # (batch_size * num_action_nodes, hidden_dim)
        h = self.bn2(h)  # (batch_size * num_action_nodes, hidden_dim)
        h = h.view(batch_size, self.num_action_nodes, -1)  # (batch_size, num_action_nodes, hidden_dim)
        h = self.dropout(h)
        output = self.out_layer(h)  # (batch_size, num_action_nodes, out_dim)
        return output

# GCN model
class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.5)  # Increased dropout
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        batch_size = x.size(0)
        h = self.conv1(x, edge_index).relu()
        h = h.view(-1, h.size(-1))
        h = self.bn1(h)
        h = h.view(batch_size, -1, h.size(-1))
        h = self.dropout(h)
        h = self.conv2(h, edge_index).relu()
        h = h.view(-1, h.size(-1))
        h = self.bn2(h)
        h = h.view(batch_size, -1, h.size(-1))
        h = self.dropout(h)
        out = self.classifier(h)  # (batch_size, num_nodes, out_dim)
        return out

# GCN-EW model
class GCN_EW(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, edge_index, max_edges=100):
        super().__init__()
        torch.manual_seed(1234)
        self.edge_weight = nn.Parameter(torch.zeros(max_edges))
        self.num_edges = edge_index.shape[1]
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.5)  # Increased dropout
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def update_edge_weight(self, new_num_edges):
        if new_num_edges > self.edge_weight.size(0):
            raise ValueError(f"New edge count {new_num_edges} exceeds max_edges {self.edge_weight.size(0)}")
        self.num_edges = new_num_edges

    def forward(self, x, edge_index):
        batch_size = x.size(0)
        edge_weight = torch.exp(self.edge_weight[:self.num_edges])
        h = self.conv1(x, edge_index, edge_weight).relu()
        h = h.view(-1, h.size(-1))
        h = self.bn1(h)
        h = h.view(batch_size, -1, h.size(-1))
        h = self.dropout(h)
        h = self.conv2(h, edge_index, edge_weight).relu()
        h = h.view(-1, h.size(-1))
        h = self.bn2(h)
        h = h.view(batch_size, -1, h.size(-1))
        h = self.dropout(h)
        out = self.classifier(h)  # (batch_size, num_nodes, out_dim)
        return out

# GAT model
class GAT(nn.Module):
    def __init__(self, hidden_channels, heads, in_dim, out_dim):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GATConv(in_dim, hidden_channels, heads)
        self.conv2 = GATConv(heads * hidden_channels, hidden_channels, heads)
        self.bn1 = nn.BatchNorm1d(heads * hidden_channels)
        self.bn2 = nn.BatchNorm1d(heads * hidden_channels)
        self.dropout = nn.Dropout(0.5)  # Increased dropout
        self.classifier = nn.Linear(heads * hidden_channels, out_dim)

    def forward(self, x, edge_index):
        batch_size = x.size(0) if x.dim() == 3 else 1
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        # Process each sample in the batch
        h = torch.zeros(batch_size, x.size(1), self.conv1.out_channels * self.conv1.heads, device=x.device)
        for i in range(batch_size):
            h[i] = self.conv1(x[i], edge_index).relu()  # Process 2D slice: (num_nodes, in_dim)
        h = h.view(-1, h.size(-1))  # (batch_size * num_nodes, hidden_channels * heads)
        h = self.bn1(h)
        h = h.view(batch_size, -1, h.size(-1))  # (batch_size, num_nodes, hidden_channels * heads)
        h = self.dropout(h)
        # Second GAT layer
        h_out = torch.zeros(batch_size, h.size(1), self.conv2.out_channels * self.conv2.heads, device=h.device)
        for i in range(batch_size):
            h_out[i] = self.conv2(h[i], edge_index).relu()  # Process 2D slice
        h = h_out.view(-1, h_out.size(-1))
        h = self.bn2(h)
        h = h.view(batch_size, -1, h.size(-1))
        h = self.dropout(h)
        out = self.classifier(h)  # (batch_size, num_nodes, out_dim)
        return out

# GraphSAGE model
class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.5)  # Increased dropout
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        batch_size = x.size(0)
        h = self.conv1(x, edge_index).relu()
        h = h.view(-1, h.size(-1))
        h = self.bn1(h)
        h = h.view(batch_size, -1, h.size(-1))
        h = self.dropout(h)
        h = self.conv2(h, edge_index).relu()
        h = h.view(-1, h.size(-1))
        h = self.bn2(h)
        h = h.view(batch_size, -1, h.size(-1))
        h = self.dropout(h)
        out = self.classifier(h)  # (batch_size, num_nodes, out_dim)
        return out

# TAD-GAT model
class TAD_GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads, time_steps=3, node_feat_dim=57, rt_meas_dim=78):
        super().__init__()
        torch.manual_seed(1234)
        self.time_steps = time_steps
        self.node_feat_dim = node_feat_dim
        self.rt_meas_dim = rt_meas_dim
        self.total_in_dim = in_dim
        self.lstm = nn.LSTM(rt_meas_dim, hidden_dim, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)
        self.node_feat_transform = nn.Linear(node_feat_dim, hidden_dim)
        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads)
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)
        self.bn2 = nn.BatchNorm1d(hidden_dim * heads)
        self.dropout = nn.Dropout(0.5)  # Increased dropout
        self.classifier = nn.Linear(hidden_dim * heads, out_dim)

    def forward(self, x, edge_index):
        batch_size, num_nodes, total_in_dim = x.size()
        node_features = x[:, :, :self.node_feat_dim]
        temporal_features = x[:, :, self.node_feat_dim:]
        temporal_features = temporal_features.view(batch_size, num_nodes, self.time_steps, self.rt_meas_dim)
        temporal_features = temporal_features.view(batch_size * num_nodes, self.time_steps, self.rt_meas_dim)
        h_temporal, _ = self.lstm(temporal_features)
        attn_weights = F.softmax(self.attention(h_temporal), dim=1)
        h_temporal = (h_temporal * attn_weights).sum(dim=1)
        h_temporal = h_temporal.view(batch_size, num_nodes, -1)
        h_static = self.node_feat_transform(node_features).relu()
        h = h_temporal + h_static
        h_out = torch.zeros(batch_size, num_nodes, self.conv1.out_channels * self.conv1.heads, device=h.device)
        for i in range(batch_size):
            h_out[i] = self.conv1(h[i], edge_index).relu()
        h = h_out.view(-1, h_out.size(-1))
        h = self.bn1(h)
        h = h.view(batch_size, num_nodes, -1)
        h = self.dropout(h)
        h_out = torch.zeros(batch_size, num_nodes, self.conv2.out_channels * self.conv2.heads, device=h.device)
        for i in range(batch_size):
            h_out[i] = self.conv2(h[i], edge_index).relu()
        h = h_out.view(-1, h_out.size(-1))
        h = self.bn2(h)
        h = h.view(batch_size, num_nodes, -1)
        h = self.dropout(h)
        out = self.classifier(h)  # (batch_size, num_nodes, out_dim)
        return out