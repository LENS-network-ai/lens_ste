"""
Multi-Layer Graph Neural Network with Residual Connections

Implements a stack of GNN layers with:
- Residual connections for better gradient flow
- Layer normalization for stable training
- Dropout for regularization
- Spectral normalization for robustness

Author: Enhanced for LENS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class GNNLayerWithResidual(nn.Module):
    """
    Single GNN layer with residual connection
    
    Performs: h' = LayerNorm(h + ReLU(Linear(Aggregate(h))))
    """
    
    def __init__(self, in_dim, out_dim, dropout=0.3, use_spectral_norm=True):
        """
        Args:
            in_dim: Input feature dimension
            out_dim: Output feature dimension
            dropout: Dropout rate
            use_spectral_norm: Whether to use spectral normalization
        """
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Linear transformation
        linear = nn.Linear(in_dim, out_dim)
        self.conv = spectral_norm(linear) if use_spectral_norm else linear
        
        # Normalization
        self.norm = nn.LayerNorm(out_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection (project if dimensions don't match)
        if in_dim != out_dim:
            residual_linear = nn.Linear(in_dim, out_dim, bias=False)
            self.residual = spectral_norm(residual_linear) if use_spectral_norm else residual_linear
        else:
            self.residual = nn.Identity()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize layer weights"""
        nn.init.kaiming_normal_(self.conv.weight if not hasattr(self.conv, 'module') else self.conv.module.weight, 
                               mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)
        
        if isinstance(self.residual, nn.Linear) or hasattr(self.residual, 'module'):
            weight = self.residual.weight if not hasattr(self.residual, 'module') else self.residual.module.weight
            nn.init.kaiming_normal_(weight, mode='fan_out', nonlinearity='relu')
    
    def aggregate(self, node_feat, adj_matrix, edge_weights):
        """
        Neighborhood aggregation with learned edge weights
        
        Args:
            node_feat: Node features [B, N, D]
            adj_matrix: Original adjacency [B, N, N]
            edge_weights: Learned edge gates [B, N, N]
        
        Returns:
            aggregated: Aggregated features [B, N, D]
        """
        # Add self-loops to adjacency
        device = adj_matrix.device
        batch_size, num_nodes = adj_matrix.shape[:2]
        
        eye = torch.eye(num_nodes, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        adj_with_self = adj_matrix + eye
        adj_with_self = torch.clamp(adj_with_self, max=1.0)
        
        # Apply learned edge weights
        weighted_adj = adj_with_self * edge_weights
        
        # Row-normalize for stable aggregation
        row_sum = torch.sum(weighted_adj, dim=2, keepdim=True) + 1e-8
        norm_adj = weighted_adj / row_sum
        
        # Aggregate: sum of weighted neighbor features
        aggregated = torch.bmm(norm_adj, node_feat)
        
        return aggregated
    
    def forward(self, node_feat, edge_weights, adj_matrix):
        """
        Forward pass with residual connection
        
        Args:
            node_feat: Node features [B, N, D_in]
            edge_weights: Edge gates [B, N, N]
            adj_matrix: Original adjacency [B, N, N]
        
        Returns:
            output: Updated node features [B, N, D_out]
        """
        # Store input for residual
        residual = self.residual(node_feat)
        
        # 1. Aggregate neighbor features
        h = self.aggregate(node_feat, adj_matrix, edge_weights)
        
        # 2. Linear transformation
        h = self.conv(h)
        
        # 3. Normalization
        h = self.norm(h)
        
        # 4. Activation
        h = F.relu(h)
        
        # 5. Dropout
        h = self.dropout(h)
        
        # 6. Residual connection
        output = h + residual
        
        return output


class MultiLayerGNN(nn.Module):
    """
    Multi-layer Graph Neural Network
    
    Stack of GNN layers with:
    - Input projection to hidden dimension
    - Multiple GNN layers with residual connections
    - Layer normalization and dropout
    
    Architecture:
        Input [B, N, D_in] 
            → Project to [B, N, D_hidden]
            → GNN Layer 1 [B, N, D_hidden]
            → GNN Layer 2 [B, N, D_hidden]
            → ...
            → GNN Layer L [B, N, D_hidden]
        Output [B, N, D_hidden]
    """
    
    def __init__(self, feature_dim, hidden_dim, num_layers=3, dropout=0.3, 
                 use_spectral_norm=True):
        """
        Args:
            feature_dim: Input feature dimension (e.g., 1024 from ResNet)
            hidden_dim: Hidden dimension for GNN layers
            num_layers: Number of GNN layers (default: 3)
            dropout: Dropout rate (default: 0.3)
            use_spectral_norm: Use spectral normalization (default: True)
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection: project input features to hidden dimension
        input_linear = nn.Linear(feature_dim, hidden_dim)
        self.input_proj = nn.Sequential(
            spectral_norm(input_linear) if use_spectral_norm else input_linear,
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Stack of GNN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = GNNLayerWithResidual(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                dropout=dropout,
                use_spectral_norm=use_spectral_norm
            )
            self.gnn_layers.append(layer)
        
        # Initialize input projection
        self._init_input_proj()
        
        print(f"[MultiLayerGNN] Initialized:")
        print(f"  • Input dim: {feature_dim}")
        print(f"  • Hidden dim: {hidden_dim}")
        print(f"  • Num layers: {num_layers}")
        print(f"  • Dropout: {dropout}")
        print(f"  • Spectral norm: {use_spectral_norm}")
    
    def _init_input_proj(self):
        """Initialize input projection weights"""
        for m in self.input_proj:
            if isinstance(m, nn.Linear) or hasattr(m, 'module'):
                weight = m.weight if not hasattr(m, 'module') else m.module.weight
                nn.init.kaiming_normal_(weight, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif hasattr(m, 'module') and m.module.bias is not None:
                    nn.init.constant_(m.module.bias, 0)
    
    def forward(self, node_feat, edge_weights, adj_matrix):
        """
        Forward pass through all GNN layers
        
        Args:
            node_feat: Input node features [B, N, D_in]
            edge_weights: Learned edge gates [B, N, N]
            adj_matrix: Original adjacency [B, N, N]
        
        Returns:
            h: Updated node features [B, N, D_hidden]
        """
        # Input: [B, N, feature_dim] → [B, N, hidden_dim]
        h = self.input_proj(node_feat)
        
        # Pass through all GNN layers
        for i, layer in enumerate(self.gnn_layers):
            h = layer(h, edge_weights, adj_matrix)
        
        # Output: [B, N, hidden_dim]
        return h
    
    def get_receptive_field(self):
        """
        Get the receptive field (number of hops) of the GNN
        
        Returns:
            int: Number of hops = number of layers
        """
        return self.num_layers
    
    def __repr__(self):
        return (f"MultiLayerGNN(\n"
                f"  feature_dim={self.feature_dim},\n"
                f"  hidden_dim={self.hidden_dim},\n"
                f"  num_layers={self.num_layers},\n"
                f"  receptive_field={self.get_receptive_field()} hops\n"
                f")")
