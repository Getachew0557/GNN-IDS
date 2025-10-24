"""
Neural Network Models for Attack Graph Classification

This module implements various neural network architectures for multiclass
classification on attack graphs, including GNNs and traditional NNs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv


class NN(nn.Module):
    """
    Traditional Neural Network for attack graph classification.
    
    Processes only action nodes without graph structure, suitable for
    baseline comparisons and scenarios where graph topology is not available.
    """
    
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_action_nodes: int = 7):
        """
        Initialize NN model.
        
        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            out_dim: Output class dimension
            num_action_nodes: Number of action nodes
        """
        super().__init__()
        torch.manual_seed(1234)
        self.num_action_nodes = num_action_nodes
        self.rt_meas_dim = in_dim
        self.name = "NN"
        
        # Network architecture
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.out_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, num_action_nodes, in_dim)
            
        Returns:
            Output tensor of shape (batch_size, num_action_nodes, out_dim)
        """
        batch_size = x.size(0)
        
        # First layer with batch norm
        h = self.lin1(x).relu()
        h = h.view(-1, h.size(-1))
        h = self.bn1(h)
        h = h.view(batch_size, self.num_action_nodes, -1)
        h = self.dropout(h)
        
        # Second layer with batch norm
        h = self.lin2(h).relu()
        h = h.view(-1, h.size(-1))
        h = self.bn2(h)
        h = h.view(batch_size, self.num_action_nodes, -1)
        h = self.dropout(h)
        
        # Output layer
        output = self.out_layer(h)
        return output


class GCN(nn.Module):
    """
    Graph Convolutional Network for attack graph classification.
    
    Leverages graph structure through message passing between connected nodes.
    Suitable for scenarios where graph topology is important.
    """
    
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        """
        Initialize GCN model.
        
        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            out_dim: Output class dimension
        """
        super().__init__()
        torch.manual_seed(1234)
        self.name = "GCN"
        
        # Graph convolutional layers
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with graph convolutions.
        
        Args:
            x: Node feature tensor of shape (batch_size, num_nodes, in_dim)
            edge_index: Graph connectivity of shape (2, num_edges)
            
        Returns:
            Output tensor of shape (batch_size, num_nodes, out_dim)
        """
        batch_size = x.size(0)
        
        # First graph convolution
        h = self.conv1(x, edge_index).relu()
        h = h.view(-1, h.size(-1))
        h = self.bn1(h)
        h = h.view(batch_size, -1, h.size(-1))
        h = self.dropout(h)
        
        # Second graph convolution
        h = self.conv2(h, edge_index).relu()
        h = h.view(-1, h.size(-1))
        h = self.bn2(h)
        h = h.view(batch_size, -1, h.size(-1))
        h = self.dropout(h)
        
        # Classification layer
        out = self.classifier(h)
        return out


class GCN_EW(nn.Module):
    """
    GCN with Learnable Edge Weights.
    
    Extends standard GCN by learning importance weights for different edges,
    allowing the model to focus on more critical attack paths.
    """
    
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, 
                 edge_index: torch.Tensor, max_edges: int = 100):
        """
        Initialize GCN-EW model.
        
        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            out_dim: Output class dimension
            edge_index: Graph connectivity
            max_edges: Maximum number of edges for weight initialization
        """
        super().__init__()
        torch.manual_seed(1234)
        self.name = "GCN-EW"
        
        # Learnable edge weights
        self.edge_weight = nn.Parameter(torch.zeros(max_edges))
        self.num_edges = edge_index.shape[1]
        
        # Graph convolutional layers
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def update_edge_weight(self, new_num_edges: int) -> None:
        """
        Update edge weight parameter for different graph sizes.
        
        Args:
            new_num_edges: New number of edges
        """
        if new_num_edges > self.edge_weight.size(0):
            raise ValueError(f"New edge count {new_num_edges} exceeds max_edges {self.edge_weight.size(0)}")
        self.num_edges = new_num_edges

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with learned edge weights.
        
        Args:
            x: Node feature tensor
            edge_index: Graph connectivity
            
        Returns:
            Output tensor
        """
        batch_size = x.size(0)
        
        # Apply learned edge weights with exponential activation
        edge_weight = torch.exp(self.edge_weight[:self.num_edges])
        
        # Graph convolutions with weighted edges
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
        
        out = self.classifier(h)
        return out


class GAT(nn.Module):
    """
    Graph Attention Network for attack graph classification.
    
    Uses attention mechanisms to weigh the importance of neighboring nodes
    differently, allowing the model to focus on more relevant attack steps.
    """
    
    def __init__(self, hidden_channels: int, heads: int, in_dim: int, out_dim: int):
        """
        Initialize GAT model.
        
        Args:
            hidden_channels: Hidden dimension per attention head
            heads: Number of attention heads
            in_dim: Input feature dimension
            out_dim: Output class dimension
        """
        super().__init__()
        torch.manual_seed(1234)
        self.name = "GAT"
        
        # Graph attention layers
        self.conv1 = GATConv(in_dim, hidden_channels, heads)
        self.conv2 = GATConv(heads * hidden_channels, hidden_channels, heads)
        self.bn1 = nn.BatchNorm1d(heads * hidden_channels)
        self.bn2 = nn.BatchNorm1d(heads * hidden_channels)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(heads * hidden_channels, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with graph attention.
        
        Args:
            x: Node feature tensor
            edge_index: Graph connectivity
            
        Returns:
            Output tensor
        """
        batch_size = x.size(0) if x.dim() == 3 else 1
        
        # Ensure batch dimension
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        # Process each sample in batch with attention
        h = torch.zeros(batch_size, x.size(1), self.conv1.out_channels * self.conv1.heads, device=x.device)
        for i in range(batch_size):
            h[i] = self.conv1(x[i], edge_index).relu()
        
        h = h.view(-1, h.size(-1))
        h = self.bn1(h)
        h = h.view(batch_size, -1, h.size(-1))
        h = self.dropout(h)
        
        # Second attention layer
        h_out = torch.zeros(batch_size, h.size(1), self.conv2.out_channels * self.conv2.heads, device=h.device)
        for i in range(batch_size):
            h_out[i] = self.conv2(h[i], edge_index).relu()
        
        h = h_out.view(-1, h_out.size(-1))
        h = self.bn2(h)
        h = h.view(batch_size, -1, h.size(-1))
        h = self.dropout(h)
        
        out = self.classifier(h)
        return out


class GraphSAGE(nn.Module):
    """
    GraphSAGE model for attack graph classification.
    
    Uses neighborhood sampling and aggregation, making it scalable to
    large graphs and suitable for dynamic attack graphs.
    """
    
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        """
        Initialize GraphSAGE model.
        
        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            out_dim: Output class dimension
        """
        super().__init__()
        torch.manual_seed(1234)
        self.name = "GraphSAGE"
        
        # GraphSAGE layers
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with GraphSAGE aggregations.
        
        Args:
            x: Node feature tensor
            edge_index: Graph connectivity
            
        Returns:
            Output tensor
        """
        batch_size = x.size(0)
        
        # First GraphSAGE layer
        h = self.conv1(x, edge_index).relu()
        h = h.view(-1, h.size(-1))
        h = self.bn1(h)
        h = h.view(batch_size, -1, h.size(-1))
        h = self.dropout(h)
        
        # Second GraphSAGE layer
        h = self.conv2(h, edge_index).relu()
        h = h.view(-1, h.size(-1))
        h = self.bn2(h)
        h = h.view(batch_size, -1, h.size(-1))
        h = self.dropout(h)
        
        out = self.classifier(h)
        return out


class TAD_GAT(nn.Module):
    """
    Temporal Attention-based Dynamic GAT for attack graph classification.
    
    Incorporates temporal dynamics through LSTM and combines with graph
    attention, suitable for time-evolving attack scenarios.
    """
    
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, heads: int, 
                 time_steps: int = 3, node_feat_dim: int = 57, rt_meas_dim: int = 78):
        """
        Initialize TAD-GAT model.
        
        Args:
            in_dim: Total input dimension
            hidden_dim: Hidden layer dimension
            out_dim: Output class dimension
            heads: Number of attention heads
            time_steps: Number of temporal steps
            node_feat_dim: Node feature dimension
            rt_meas_dim: Runtime measurement dimension
        """
        super().__init__()
        torch.manual_seed(1234)
        self.name = "TAD-GAT"
        
        # Temporal and structural parameters
        self.time_steps = time_steps
        self.node_feat_dim = node_feat_dim
        self.rt_meas_dim = rt_meas_dim
        self.total_in_dim = in_dim
        
        # Temporal processing with LSTM
        self.lstm = nn.LSTM(rt_meas_dim, hidden_dim, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)
        
        # Feature transformations
        self.node_feat_transform = nn.Linear(node_feat_dim, hidden_dim)
        
        # Graph attention layers
        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads)
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)
        self.bn2 = nn.BatchNorm1d(hidden_dim * heads)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_dim * heads, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with temporal and graph attention.
        
        Args:
            x: Input tensor with node and temporal features
            edge_index: Graph connectivity
            
        Returns:
            Output tensor
        """
        batch_size, num_nodes, total_in_dim = x.size()
        
        # Separate node and temporal features
        node_features = x[:, :, :self.node_feat_dim]
        temporal_features = x[:, :, self.node_feat_dim:]
        
        # Reshape for temporal processing
        temporal_features = temporal_features.view(batch_size, num_nodes, self.time_steps, self.rt_meas_dim)
        temporal_features = temporal_features.view(batch_size * num_nodes, self.time_steps, self.rt_meas_dim)
        
        # LSTM for temporal patterns
        h_temporal, _ = self.lstm(temporal_features)
        
        # Temporal attention
        attn_weights = F.softmax(self.attention(h_temporal), dim=1)
        h_temporal = (h_temporal * attn_weights).sum(dim=1)
        h_temporal = h_temporal.view(batch_size, num_nodes, -1)
        
        # Combine with static node features
        h_static = self.node_feat_transform(node_features).relu()
        h = h_temporal + h_static
        
        # Graph attention processing
        h_out = torch.zeros(batch_size, num_nodes, self.conv1.out_channels * self.conv1.heads, device=h.device)
        for i in range(batch_size):
            h_out[i] = self.conv1(h[i], edge_index).relu()
        
        h = h_out.view(-1, h_out.size(-1))
        h = self.bn1(h)
        h = h.view(batch_size, num_nodes, -1)
        h = self.dropout(h)
        
        # Second graph attention layer
        h_out = torch.zeros(batch_size, num_nodes, self.conv2.out_channels * self.conv2.heads, device=h.device)
        for i in range(batch_size):
            h_out[i] = self.conv2(h[i], edge_index).relu()
        
        h = h_out.view(-1, h_out.size(-1))
        h = self.bn2(h)
        h = h.view(batch_size, num_nodes, -1)
        h = self.dropout(h)
        
        out = self.classifier(h)
        return out