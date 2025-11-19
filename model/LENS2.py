"""
LENS Model with Multi-Layer GNN and Attention Pooling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

# ✅ IMPORT NEW COMPONENTS
from model.multilayerGNN import MultiLayerGNN
from model.attPooling import MultiHeadAttentionPooling

# Original imports
from model.EGL_L0_Reg import EGLassoRegularization
from model.EdgeScoring import EdgeScoringNetwork
from model.StatsTracker import StatsTracker
from model.L0Utils import l0_train, l0_test, L0RegularizerParams
from model.L0Utils_ARM import ARML0RegularizerParams, arm_sample_gates


class ImprovedEdgeGNN(nn.Module):
    """
    LENS with Multi-Layer GNN and Multi-Head Attention Pooling
    """
    
    def __init__(self, feature_dim, hidden_dim, num_classes, 
                 # GNN parameters
                 num_gnn_layers=3,           # ← NEW
                 num_attention_heads=4,      # ← NEW
                 use_attention_pooling=True, # ← NEW
                 
                 # Regularization parameters
                 lambda_reg=0.01, 
                 lambda_density=0.03,
                 target_density=0.30,
                 reg_mode='l0', 
                 l0_method='hard-concrete',
                 
                 # Architecture parameters
                 edge_dim=32, 
                 dropout=0.2,
                 
                 # Training parameters
                 warmup_epochs=15,
                 ramp_epochs=20,
                 graph_size_adaptation=True, 
                 min_edges_per_node=2, 
                 
                 # L0 parameters
                 l0_gamma=-0.1, 
                 l0_zeta=1.1, 
                 l0_beta=0.66, 
                 baseline_ema=0.9, 
                 initial_temp=5.0,
                 
                 # Penalty mode parameters
                 enable_adaptive_lambda=True,
                 enable_density_loss=True,
                 alpha_min=0.2,
                 alpha_max=2.0,
                 
                 # Constrained mode parameters
                 use_constrained=False,
                 dual_lr=1e-3,
                 enable_dual_restarts=True,
                 constraint_target=0.30):
        
        super().__init__()
        
        self.num_classes = num_classes
        self.reg_mode = reg_mode
        self.l0_method = l0_method
        self.use_l0 = (reg_mode == 'l0')
        self.use_constrained = use_constrained
        self.num_gnn_layers = num_gnn_layers
        self.use_attention_pooling = use_attention_pooling
        
        # Create L0 parameters
        self.l0_params = None
        if self.use_l0:
            if l0_method == 'hard-concrete':
                self.l0_params = L0RegularizerParams(
                    gamma=l0_gamma, zeta=l0_zeta, beta_l0=l0_beta
                )
            elif l0_method == 'arm':
                self.l0_params = ARML0RegularizerParams(
                    gamma=l0_gamma, zeta=l0_zeta, baseline_ema=baseline_ema
                )
        
        # Edge scoring network
        self.edge_scorer = EdgeScoringNetwork(
            feature_dim=feature_dim,
            edge_dim=edge_dim,
            l0_method=l0_method,
            l0_params=self.l0_params,
        )
        
        # ============================================
        # ✅ NEW: Multi-Layer GNN
        # ============================================
        self.gnn = MultiLayerGNN(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            dropout=dropout,
            use_spectral_norm=True
        )
        
        # ============================================
        # ✅ NEW: Attention Pooling
        # ============================================
        if use_attention_pooling:
            self.pooling = MultiHeadAttentionPooling(
                hidden_dim=hidden_dim,
                num_heads=num_attention_heads,
                dropout=dropout,
                use_edge_masking=True
            )
        else:
            # Fallback to old pooling
            from model.GraphPooling import EdgeWeightedAttentionPooling
            self.pooling = EdgeWeightedAttentionPooling()
        
        # Regularizer
        self.regularizer = EGLassoRegularization(
            lambda_reg=lambda_reg,
            lambda_density=lambda_density,
            target_density=target_density,
            reg_mode=reg_mode,
            warmup_epochs=warmup_epochs,
            ramp_epochs=ramp_epochs,
            l0_params=self.l0_params,
            l0_method=l0_method,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            enable_adaptive_lambda=enable_adaptive_lambda,
            enable_density_loss=enable_density_loss,
            use_constrained=use_constrained,
            dual_lr=dual_lr,
            enable_dual_restarts=enable_dual_restarts,
            constraint_target=constraint_target,
        )
        
        # Stats tracker
        self.stats_tracker = StatsTracker()
        
        # ============================================
        # ✅ UPDATED: Stronger Classifier
        # ============================================
        self.classifier = nn.Sequential(
            spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            spectral_norm(nn.Linear(hidden_dim // 2, hidden_dim // 4)),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            spectral_norm(nn.Linear(hidden_dim // 4, num_classes))
        )
        
        # Initialize classifier
        for m in self.classifier:
            if isinstance(m, nn.Linear) or hasattr(m, 'module'):
                weight = m.weight if not hasattr(m, 'module') else m.module.weight
                nn.init.kaiming_normal_(weight, mode='fan_out', nonlinearity='relu')
                bias = m.bias if not hasattr(m, 'module') else m.module.bias
                if bias is not None:
                    nn.init.constant_(bias, 0)
        
        # Graph adaptation
        self.graph_size_adaptation = graph_size_adaptation
        self.min_edges_per_node = min_edges_per_node
        
        # Training state
        self.current_epoch = 0
        self.warmup_epochs = warmup_epochs
        self.initial_temp = initial_temp
        self.temperature = self.initial_temp
        
        # ARM-specific
        self.last_edge_weights_anti = None
        self.last_constraint_violation = None
        
        print(f"\n[LENS] Model initialized:")
        print(f"  → Architecture: Multi-Layer GNN + {'Multi-Head Attention' if use_attention_pooling else 'Standard'} Pooling")
        print(f"  → GNN Layers: {num_gnn_layers}")
        if use_attention_pooling:
            print(f"  → Attention Heads: {num_attention_heads}")
        print(f"  → Hidden Dim: {hidden_dim}")
        print(f"  → Optimization Mode: {'CONSTRAINED' if use_constrained else 'PENALTY'}")
        print(f"  → Regularization: {reg_mode}")
        print(f"  → L0 Method: {l0_method if self.use_l0 else 'N/A'}")
    
    def set_print_stats(self, value):
        """Control whether to print stats during forward pass"""
        self.stats_tracker.print_stats = value
    
    def update_temperature_and_lambda(self):
        """Update temperature and lambda schedules"""
        if hasattr(self.regularizer, 'update_all_schedules'):
            schedules = self.regularizer.update_all_schedules(
                current_epoch=self.current_epoch,
                initial_temp=self.initial_temp
            )
            self.temperature = schedules['temperature']
    
    def set_epoch(self, epoch):
        """Set current epoch and update schedules"""
        self.current_epoch = epoch
        self.regularizer.current_epoch = epoch
        self.update_temperature_and_lambda()
    
    def forward(self, node_feat, labels, adjs, masks=None, 
                return_edge_weights_anti=False):
        """
        Forward pass with multi-layer GNN and attention pooling
        
        Args:
            node_feat: Node features [B, N, D]
            labels: Ground truth labels [B]
            adjs: Adjacency matrices [B, N, N]
            masks: Node masks [B, N] (optional)
            return_edge_weights_anti: Return antithetic samples (ARM only)
        
        Returns:
            logits, labels, total_loss, weighted_adj[, edge_weights_anti]
        """
        # Normalize node features
        node_feat = F.normalize(node_feat, p=2, dim=2)
        batch_size = node_feat.shape[0]
        
        # ============================================
        # 1. COMPUTE EDGE WEIGHTS
        # ============================================
        edge_scorer_outputs = self.edge_scorer.compute_edge_weights(
            node_feat=node_feat,
            adj_matrix=adjs,
            current_epoch=self.current_epoch,
            warmup_epochs=self.warmup_epochs,
            temperature=self.temperature,
            graph_size_adaptation=self.graph_size_adaptation,
            min_edges_per_node=self.min_edges_per_node,
            regularizer=self.regularizer if self.use_l0 else None,
            use_l0=self.use_l0,
            print_stats=self.stats_tracker.print_stats,
            l0_params=self.l0_params,
            training=self.training
        )
        
        # Handle different return formats
        if self.use_l0 and self.l0_method == 'arm' and self.training:
            edge_weights, edge_weights_anti, logAlpha = edge_scorer_outputs
            self.last_edge_weights_anti = edge_weights_anti
        else:
            if isinstance(edge_scorer_outputs, tuple) and len(edge_scorer_outputs) == 2:
                edge_weights, logAlpha = edge_scorer_outputs
            else:
                edge_weights = edge_scorer_outputs
                logAlpha = None
            edge_weights_anti = None
        
        # ============================================
        # 2. ✅ MULTI-LAYER GNN
        # ============================================
        h = self.gnn(node_feat, edge_weights, adjs)
        
        # ============================================
        # 3. ✅ ATTENTION POOLING
        # ============================================
        if self.use_attention_pooling:
            graph_rep = self.pooling(h, edge_weights, adjs, masks)
        else:
            # Old pooling method
            graph_rep = self.pooling.edge_weighted_attention_pooling(h, edge_weights, adjs, masks)
        
        # ============================================
        # 4. CLASSIFICATION
        # ============================================
        logits = self.classifier(graph_rep)
        
        # Classification loss
        cls_loss = F.cross_entropy(logits, labels)
        
        # ============================================
        # 5. REGULARIZATION
        # ============================================
        if self.use_l0 and logAlpha is not None:
            from model.EGL_L0_Reg import compute_density
            current_density = compute_density(edge_weights, adjs)
            
            # Compute L0 penalty
            if self.l0_method == 'hard-concrete':
                from model.L0Utils import get_loss2
                l0_penalty = get_loss2(logAlpha, params=self.l0_params).sum()
            elif self.l0_method == 'arm':
                from model.L0Utils_ARM import get_expected_l0_arm
                l0_penalty = get_expected_l0_arm(logAlpha, self.l0_params)
            else:
                raise ValueError(f"Unknown l0_method: {self.l0_method}")
            
            # Compute regularization
            reg_loss, reg_stats = self.regularizer.compute_regularization_with_l0(
                l0_penalty=l0_penalty,
                edge_weights=edge_weights,
                adj_matrix=adjs,
                return_stats=True
            )
            
            if self.use_constrained:
                self.last_constraint_violation = reg_stats.get('constraint_violation', 0)
            
            l0_loss = reg_stats.get('l0_loss', 0)
            density_loss = reg_stats.get('density_loss', 0)
            lambda_eff = reg_stats.get('lambda_eff', self.regularizer.get_effective_lambda())
        
        else:
            reg_loss = self.regularizer.compute_regularization(edge_weights, adjs)
            l0_loss = reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss
            density_loss = 0
            lambda_eff = self.regularizer.get_effective_lambda()
            current_density = 0
            self.last_constraint_violation = None
        
        # Total loss
        total_loss = cls_loss + reg_loss
        
        # ============================================
        # 6. UPDATE STATS
        # ============================================
        self.stats_tracker.update_stats(
            edge_weights, adjs, cls_loss, reg_loss,
            self.current_epoch, lambda_eff
        )
        
        # Add mode-specific stats
        if not hasattr(self.stats_tracker, 'density_loss_history'):
            self.stats_tracker.density_loss_history = []
            self.stats_tracker.current_density_history = []
            self.stats_tracker.lambda_eff_history = []
            self.stats_tracker.constraint_violation_history = []
            self.stats_tracker.dual_lambda_history = []
        
        if self.use_l0:
            self.stats_tracker.density_loss_history.append(density_loss)
            self.stats_tracker.current_density_history.append(
                current_density.item() if isinstance(current_density, torch.Tensor) else current_density
            )
            self.stats_tracker.lambda_eff_history.append(lambda_eff)
            
            if self.use_constrained and self.last_constraint_violation is not None:
                self.stats_tracker.constraint_violation_history.append(self.last_constraint_violation)
                self.stats_tracker.dual_lambda_history.append(self.regularizer.dual_lambda)
        
        # Return
        if return_edge_weights_anti and edge_weights_anti is not None:
            return logits, labels, total_loss, adjs * edge_weights, edge_weights_anti
        else:
            return logits, labels, total_loss, adjs * edge_weights
    
    # ... (keep all other methods: forward_arm_antithetic, get_arm_gradient_loss, etc.)
    
    def __repr__(self):
        """String representation"""
        mode_info = "CONSTRAINED" if self.use_constrained else "PENALTY"
        return (f"ImprovedEdgeGNN(LENS-v2):\n"
                f"  Architecture: {self.num_gnn_layers}-layer GNN + "
                f"{'Multi-Head Attention' if self.use_attention_pooling else 'Standard'} Pooling\n"
                f"  Mode: {mode_info}\n"
                f"  Regularization: {self.reg_mode}\n"
                f"  L0 Method: {self.l0_method if self.use_l0 else 'N/A'}\n"
                f"  Classes: {self.num_classes}")
