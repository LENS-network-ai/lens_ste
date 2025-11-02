# model/EdgeScoring.py
"""
Edge Scoring Network with support for Hard-Concrete and ARM
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
    Supports both Hard-Concrete and ARM L0 regularization
    """
    
    def __init__(self, feature_dim, edge_dim=32, dropout=0.2, 
                 l0_params=None, l0_method='hard-concrete'):
        """
        Args:
            feature_dim: Node feature dimension
            edge_dim: Hidden dimension for edge scoring MLP
            dropout: Dropout rate
            l0_params: L0RegularizerParams or ARML0RegularizerParams
            l0_method: 'hard-concrete' or 'arm'
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
    
    def compute_edge_weights(self, node_feat, adj_matrix, 
                            current_epoch=0, warmup_epochs=5, temperature=None,
                            graph_size_adaptation=True, min_edges_per_node=2,
                            regularizer=None, use_l0=False, print_stats=False,
                            l0_params=None, training=True):
        """
        Compute edge weights using selected L0 method
        
        Args:
            node_feat: Node features [B, N, D]
            adj_matrix: DENSE adjacency matrix [B, N, N]
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
            If Hard-Concrete: (edge_weights, logAlpha)
            If ARM (training): (edge_weights, edge_weights_anti, logAlpha)
            If ARM (eval): (edge_weights, logAlpha)
        """
        batch_size, num_nodes, feat_dim = node_feat.shape
        device = node_feat.device
        
        # Use provided l0_params or fall back to self.l0_params
        params = l0_params if l0_params is not None else self.l0_params
        
        # Ensure adjacency is dense
        if adj_matrix.is_sparse:
            adj_matrix = adj_matrix.to_dense()
        
        # Compute pairwise features for ALL potential edges
        # This creates features for every (i,j) pair, not just existing edges
        
        # Expand node features to compute all pairs
        # src_feat: [B, N, 1, D] -> [B, N, N, D]
        # tgt_feat: [B, 1, N, D] -> [B, N, N, D]
        src_feat = node_feat.unsqueeze(2).expand(batch_size, num_nodes, num_nodes, feat_dim)
        tgt_feat = node_feat.unsqueeze(1).expand(batch_size, num_nodes, num_nodes, feat_dim)
        
        # Compute pairwise distances [B, N, N, 1]
        distances = torch.norm(src_feat - tgt_feat, dim=-1, keepdim=True)
        
        # Concatenate: [src || tgt || dist] -> [B, N, N, 2*D+1]
        edge_features = torch.cat([src_feat, tgt_feat, distances], dim=-1)
        
        # Reshape to [B*N*N, 2*D+1] for MLP
        edge_features_flat = edge_features.reshape(-1, edge_features.size(-1))
        
        # Compute logAlpha (edge logits) for all pairs
        logAlpha_flat = self.edge_mlp(edge_features_flat).squeeze(-1)  # [B*N*N]
        logAlpha = logAlpha_flat.reshape(batch_size, num_nodes, num_nodes)  # [B, N, N]
        
        # Store for ARM gradient computation
        self.last_logAlpha = logAlpha
        
        # Create mask for valid edges (where adj_matrix > 0)
        edge_mask = (adj_matrix > 0).float()
        # DEBUG CHECK (first call only)
        if not hasattr(self, '_checked'):
          edge_mask_bool = (adj_matrix > 0)
          mean_val = logAlpha[edge_mask_bool].mean()
          print(f"🔍 LogAlpha mean: {mean_val:.4f} {'✅ POSITIVE' if mean_val > 0 else '⚠️ NEGATIVE'}")
          self._checked = True 
        # Mask out invalid edges by setting logAlpha to very negative
        logAlpha = logAlpha * edge_mask + (1 - edge_mask) * (-1e9)
        
        # Store logits in regularizer if using L0
        if use_l0 and regularizer is not None:
            regularizer.clear_logits()
            for b in range(batch_size):
                regularizer.store_logits(b, logAlpha[b])
        
        # Apply L0 gating based on method
        if use_l0 and params is not None:
            if self.l0_method == 'hard-concrete':
                # Hard-Concrete L0
                if training:
                    edge_weights = l0_train(logAlpha, params=params,temperature=temperature)
                else:
                    edge_weights = l0_test(logAlpha, params=params,temperature=temperature)
                
                # Apply edge mask
                edge_weights = edge_weights * edge_mask
                
                if print_stats:
                    active_edges = (edge_weights > 0.1).float().sum().item()
                    total_edges = edge_mask.sum().item()
                    print(f"   [EdgeScoring] Active edges: {active_edges}/{total_edges:.0f} "
                          f"({100*active_edges/max(total_edges,1):.1f}%)")
                
                return edge_weights, logAlpha
            
            elif self.l0_method == 'arm':
                # ARM L0
                edge_weights, edge_weights_anti = arm_sample_gates(
                    logAlpha, params, training=training
                )
                
                # Apply edge mask to both
                edge_weights = edge_weights * edge_mask
                if edge_weights_anti is not None:
                    edge_weights_anti = edge_weights_anti * edge_mask
                
                if print_stats:
                    active_edges = (edge_weights > 0.5).float().sum().item()
                    total_edges = edge_mask.sum().item()
                    print(f"   [EdgeScoring-ARM] Sampled edges: {active_edges}/{total_edges:.0f} "
                          f"({100*active_edges/max(total_edges,1):.1f}%)")
                
                # Return format depends on training mode
                if training and edge_weights_anti is not None:
                    return edge_weights, edge_weights_anti, logAlpha
                else:
                    return edge_weights, logAlpha
            
            else:
                raise ValueError(f"Unknown l0_method: {self.l0_method}")
        
        else:
            # No L0 regularization - use Gumbel-Softmax (legacy behavior)
            if training and current_epoch < warmup_epochs:
                # During warmup: soft edges
                edge_probs = torch.sigmoid(logAlpha / temperature)
                edge_weights = edge_probs * edge_mask
            else:
                # After warmup: apply thresholding
                edge_probs = torch.sigmoid(logAlpha / temperature)
                edge_weights = edge_probs * edge_mask
            
            return edge_weights, logAlpha
