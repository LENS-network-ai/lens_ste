# model/EdgeScoring.py
"""
Edge Scoring Network with support for Hard-Concrete, ARM, and STE
Works with DENSE adjacency matrices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from model.L0Utils import l0_train, l0_test, L0RegularizerParams
from model.L0Utils_ARM import arm_sample_gates, ARML0RegularizerParams


class EdgeScoringNetwork(nn.Module):
    """
    Edge Scoring Network that computes importance scores for edges
    Supports Hard-Concrete, ARM, and STE L0 regularization
    """
    
    def __init__(self, feature_dim, edge_dim=32, dropout=0.2, 
                 l0_params=None, l0_method='hard-concrete'):
        """
        Args:
            feature_dim: Node feature dimension
            edge_dim: Hidden dimension for edge scoring MLP
            dropout: Dropout rate
            l0_params: L0RegularizerParams or ARML0RegularizerParams
            l0_method: 'hard-concrete', 'arm', or 'ste'
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.edge_dim = edge_dim
        self.dropout = dropout
        self.l0_params = l0_params
        self.l0_method = l0_method
        
        # Edge scoring MLP
        # Input: [src_feat || tgt_feat || distance] = 2*feature_dim + 1
        self.edge_mlp = nn.Sequential(
            spectral_norm(nn.Linear(feature_dim * 2 + 1, edge_dim)),
            nn.ReLU(),
            nn.Dropout(dropout),
            spectral_norm(nn.Linear(edge_dim, edge_dim)),
            nn.ReLU(),
            nn.Dropout(dropout),
            spectral_norm(nn.Linear(edge_dim, 1))
        )
        
        # Initialize with Kaiming
        for m in self.edge_mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # ============================================================
        # CRITICAL FIX: Positive bias initialization for logAlpha
        # ============================================================
        # Get final layer (unwrap spectral_norm if needed)
        final_module = self.edge_mlp[-1]
        final_layer = final_module.module if hasattr(final_module, 'module') else final_module

        # Set positive bias → logAlpha starts positive → gates start open
        nn.init.constant_(final_layer.bias, 1.0)
        nn.init.xavier_uniform_(final_layer.weight, gain=0.1)

        print("[EdgeScoring] ✅ Initialized with positive bias (gates start ~73% open)")
        # ============================================================
        
        # Store last logAlpha for ARM gradient computation
        self.last_logAlpha = None
        
        print(f"[EdgeScoring] Initialized: method={l0_method}, feature_dim={feature_dim}, edge_dim={edge_dim}")
    def compute_edge_weights(
     self, node_feat, adj_matrix,
     current_epoch=0, warmup_epochs=5, temperature=None,
     graph_size_adaptation=True, min_edges_per_node=2,
     regularizer=None, use_l0=False, print_stats=False,
     l0_params=None, training=True
     ):
     """
     Compute edge weights using selected L0 method for sparse graphs.

     Args:
        node_feat: Node features [B, N, D]
        adj_matrix: Dense adjacency matrix [B, N, N]
        current_epoch: Current training epoch
        warmup_epochs: Number of warmup epochs
        temperature: Temperature for Gumbel-Softmax
        graph_size_adaptation: Whether to adapt to graph size
        min_edges_per_node: Minimum edges per node
        regularizer: EGLassoRegularization instance
        use_l0: Whether using L0 regularization
        print_stats: Whether to print statistics
        l0_params: L0 parameters (overrides self.l0_params)
        training: Whether in training mode

     Returns:
        Depends on L0 method:
            Hard-Concrete: (edge_weights, logAlpha)
            ARM (training): (edge_weights, edge_weights_anti, logAlpha)
            ARM (eval): (edge_weights, logAlpha)
            STE: (edge_weights, logAlpha)
     """
     batch_size, num_nodes, feat_dim = node_feat.shape
     device = node_feat.device

     params = l0_params if l0_params is not None else self.l0_params

     # Ensure adjacency is dense
     if adj_matrix.is_sparse:
        adj_matrix = adj_matrix.to_dense()

     # -------------------------
     # Build sparse edge index
     # -------------------------
     edge_index_list = []
     for b in range(batch_size):
        edges_b = (adj_matrix[b] > 0).nonzero(as_tuple=False)  # [E_b, 2]
        if edges_b.numel() == 0:
            continue
        batch_col = torch.full((edges_b.size(0), 1), b, device=device)
        edge_index_list.append(torch.cat([batch_col, edges_b], dim=1))  # [E_b, 3]

     if len(edge_index_list) == 0:
        # No edges in batch
        return None, None

     edge_index = torch.cat(edge_index_list, dim=0)  # [E_total, 3]
     batch_ids, src_nodes, tgt_nodes = edge_index[:, 0], edge_index[:, 1], edge_index[:, 2]

     # -------------------------
     # Gather node features
     # -------------------------
     src_feat = node_feat[batch_ids, src_nodes]  # [E_total, D]
     tgt_feat = node_feat[batch_ids, tgt_nodes]  # [E_total, D]

     distances = torch.norm(src_feat - tgt_feat, dim=-1, keepdim=True)  # [E_total, 1]
     edge_features = torch.cat([src_feat, tgt_feat, distances], dim=-1)  # [E_total, 2*D+1]

     # -------------------------
     # Compute edge logits
     # -------------------------
     logAlpha = self.edge_mlp(edge_features).squeeze(-1)  # [E_total]
     logAlpha = torch.clamp(logAlpha, min=-5.0, max=5.0)
     self.last_logAlpha = logAlpha

     # Debug check
     if not hasattr(self, '_checked'):
        mean_val = logAlpha.mean()
        print(f"🔍 LogAlpha mean: {mean_val:.4f} {'✅ POSITIVE' if mean_val > 0 else '⚠️ NEGATIVE'}")
        self._checked = True

     # -------------------------
     # Store logits in regularizer
     # -------------------------
     if use_l0 and regularizer is not None:
        regularizer.clear_logits()
        for b in range(batch_size):
            mask_b = batch_ids == b
            regularizer.store_logits(b, logAlpha[mask_b])

     # -------------------------
     # Apply L0 gating
     # -------------------------
     if use_l0 and params is not None:
        if self.l0_method == 'hard-concrete':
            edge_weights = l0_train(logAlpha, params=params, temperature=temperature) if training else l0_test(logAlpha, params=params, temperature=temperature)

        elif self.l0_method == 'arm':
            edge_weights, edge_weights_anti = arm_sample_gates(logAlpha, params, training=training)
            if training and edge_weights_anti is not None:
                return edge_weights, edge_weights_anti, logAlpha

        elif self.l0_method == 'ste':
            from model.L0Utils_STE import ste_sample_gates
            if training:
                edge_weights, _ = ste_sample_gates(logAlpha, temperature=temperature)
            else:
                probs = torch.sigmoid(logAlpha / temperature)
                edge_weights = (probs > 0.5).float()

        else:
            raise ValueError(f"Unknown l0_method: {self.l0_method}")

        if print_stats:
            active_edges = (edge_weights > 0.5).float().sum().item()
            print(f"   [EdgeScoring-{self.l0_method.upper()}] Active edges: {active_edges}/{edge_features.size(0)} "
                  f"({100*active_edges/max(edge_features.size(0),1):.1f}%)")

        return edge_weights, logAlpha

     # -------------------------
     # No L0 regularization
     # -------------------------
     if training and current_epoch < warmup_epochs:
        edge_probs = torch.sigmoid(logAlpha / temperature)
        edge_weights = edge_probs
     else:
        edge_probs = torch.sigmoid(logAlpha / temperature)
        edge_weights = edge_probs

     return edge_weights, logAlpha

       
