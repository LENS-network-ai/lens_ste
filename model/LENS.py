"""
LENS Model with support for Hard-Concrete, ARM, and STE L0 regularization
WITH TARGET DENSITY CONTROL AND CONSTRAINED OPTIMIZATION

Supports TWO optimization modes:
1. PENALTY MODE: Scheduled lambda with adaptive scaling and density loss
2. CONSTRAINED MODE: Lagrangian optimization with dual variables (paper approach)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from model.EGL_L0_Reg import EGLassoRegularization
from model.EdgeScoring import EdgeScoringNetwork
from model.GraphPooling import EdgeWeightedAttentionPooling
from model.StatsTracker import StatsTracker
from model.L0Utils import l0_train, l0_test, L0RegularizerParams
from model.L0Utils_ARM import ARML0RegularizerParams, arm_sample_gates, compute_arm_loss, get_expected_l0_arm


class ImprovedEdgeGNN(nn.Module):
    """
    LENS: Learning Edge-Node Sparsity for Whole Slide Image Analysis
    
    Supports three L0 regularization methods:
    - hard-concrete: Continuous relaxation with Gumbel-Softmax (default)
    - arm: Augment-REINFORCE-Merge for direct binary sampling
    - ste: Straight-Through Estimator with binary gates
    
    Supports TWO optimization modes:
    - penalty: Scheduled lambda with adaptive scaling (default)
    - constrained: Lagrangian with dual variables (paper approach)
    """
    
    def __init__(self, feature_dim, hidden_dim, num_classes, 
                 lambda_reg=0.01, 
                 lambda_density=0.03,
                 target_density=0.30,
                 reg_mode='l0', 
                 l0_method='hard-concrete',
                 edge_dim=32, 
                 warmup_epochs=15,
                 ramp_epochs=20,
                 graph_size_adaptation=True, 
                 min_edges_per_node=2, 
                 dropout=0.2, 
                 l0_gamma=-0.1, 
                 l0_zeta=1.1, 
                 l0_beta=0.66, 
                 baseline_ema=0.9, 
                 initial_temp=5.0,
                 enable_adaptive_lambda=True,
                 enable_density_loss=True,
                 alpha_min=0.2,
                 alpha_max=2.0,
                 # 🆕 CONSTRAINED OPTIMIZATION PARAMETERS
                 use_constrained=False,
                 dual_lr=1e-3,
                 enable_dual_restarts=True,
                 constraint_target=0.30):
        """
        Args:
            feature_dim: Input feature dimension
            hidden_dim: Hidden dimension for GNN
            num_classes: Number of output classes
            
            PENALTY MODE Args:
                lambda_reg: Base L0 regularization strength (λ_base)
                lambda_density: Density loss weight (λ_ρ)
                target_density: Target edge retention rate (0.0-1.0)
                enable_adaptive_lambda: Enable adaptive lambda mechanism
                enable_density_loss: Enable density loss term
                alpha_min: Minimum adaptive scaling factor
                alpha_max: Maximum adaptive scaling factor
            
            CONSTRAINED MODE Args:
                use_constrained: If True, use constrained optimization (paper)
                dual_lr: Learning rate for dual variable (η_dual)
                enable_dual_restarts: Enable dual restart heuristic
                constraint_target: Constraint level ε (expected L0 density)
            
            Common Args:
                reg_mode: 'l0' or other regularization mode
                l0_method: 'hard-concrete', 'arm', or 'ste'
                edge_dim: Hidden dimension for edge scoring network
                warmup_epochs: Number of warmup epochs
                ramp_epochs: Number of epochs for lambda ramp after warmup
                graph_size_adaptation: Whether to adapt to graph size
                min_edges_per_node: Minimum edges per node
                dropout: Dropout rate
                l0_gamma: Lower stretch bound for L0
                l0_zeta: Upper stretch bound for L0
                l0_beta: Temperature parameter for Hard-Concrete
                baseline_ema: EMA coefficient for ARM baseline
                initial_temp: Initial temperature for annealing
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.reg_mode = reg_mode
        self.l0_method = l0_method
        self.use_l0 = (reg_mode == 'l0')
        self.use_constrained = use_constrained
        
        # Create L0 parameters based on method
        self.l0_params = None
        if self.use_l0:
            if l0_method == 'hard-concrete':
                self.l0_params = L0RegularizerParams(
                    gamma=l0_gamma, 
                    zeta=l0_zeta, 
                    beta_l0=l0_beta
                )
                print(f"[LENS] Using Hard-Concrete L0 regularization")
                print(f"  → Parameters: gamma={l0_gamma}, zeta={l0_zeta}, beta={l0_beta}")
                
            elif l0_method == 'arm':
                self.l0_params = ARML0RegularizerParams(
                    gamma=l0_gamma,
                    zeta=l0_zeta,
                    baseline_ema=baseline_ema
                )
                print(f"[LENS] Using ARM L0 regularization")
                print(f"  → Parameters: gamma={l0_gamma}, zeta={l0_zeta}, baseline_ema={baseline_ema}")
                
            elif l0_method == 'ste':
                from model.L0Utils_STE import STERegularizerParams
                self.l0_params = STERegularizerParams()
                print(f"[LENS] Using STE (Straight-Through Estimator)")
                print(f"  → Binary gates during training with straight-through gradients!")
                
            else:
                raise ValueError(f"Unknown l0_method: {l0_method}. Use 'hard-concrete', 'arm', or 'ste'")
        
        # Edge scoring network (supports all methods)
        self.edge_scorer = EdgeScoringNetwork(
            feature_dim=feature_dim,
            edge_dim=edge_dim,
            l0_method=l0_method,
            l0_params=self.l0_params,
        )
        
        # Pooling layer
        self.pooling = EdgeWeightedAttentionPooling()
        
        # ============================================
        # 🆕 REGULARIZER WITH PENALTY OR CONSTRAINED MODE
        # ============================================
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
            # Constrained parameters
            use_constrained=use_constrained,
            dual_lr=dual_lr,
            enable_dual_restarts=enable_dual_restarts,
            constraint_target=constraint_target,
        )
        
        # Stats tracker
        self.stats_tracker = StatsTracker()
        
        # GNN layer
        self.conv = nn.Linear(feature_dim, hidden_dim)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            spectral_norm(nn.Linear(hidden_dim // 2, num_classes))
        )
        
        # Graph adaptation parameters
        self.graph_size_adaptation = graph_size_adaptation
        self.min_edges_per_node = min_edges_per_node
        
        # Training state
        self.current_epoch = 0
        self.warmup_epochs = warmup_epochs
        self.initial_temp = initial_temp
        self.temperature = self.initial_temp
        
        # ARM-specific: store for gradient computation
        self.last_edge_weights_anti = None
        
        # 🆕 Store last constraint violation for dual update
        self.last_constraint_violation = None
        
        print(f"\n[LENS] Model initialized:")
        print(f"  → Optimization Mode: {'CONSTRAINED' if use_constrained else 'PENALTY'}")
        print(f"  → Regularization: {reg_mode}")
        print(f"  → L0 Method: {l0_method if self.use_l0 else 'N/A'}")
        
        if use_constrained:
            print(f"  → Constraint target (ε): {constraint_target*100:.1f}%")
            print(f"  → Dual learning rate: {dual_lr}")
            print(f"  → Dual restarts: {'enabled' if enable_dual_restarts else 'disabled'}")
        else:
            print(f"  → Base lambda: {lambda_reg}")
            print(f"  → Target density: {target_density*100:.1f}%")
            print(f"  → Density loss: {'enabled' if enable_density_loss else 'disabled'}")
            print(f"  → Adaptive lambda: {'enabled' if enable_adaptive_lambda else 'disabled'}")
        
        print(f"  → Warmup epochs: {warmup_epochs}")
        print(f"  → Ramp epochs: {ramp_epochs}")
        print(f"  → Graph size adaptation: {graph_size_adaptation}")
        print(f"  → Min edges per node: {min_edges_per_node}")
        print(f"  → Initial temperature: {initial_temp}")
    
    def set_print_stats(self, value):
        """Control whether to print stats during forward pass"""
        self.stats_tracker.print_stats = value
    
    def update_temperature_and_lambda(self):
        """
        Update temperature and all lambda values based on current epoch
        Uses the new unified schedule update from EGL_L0_Reg
        """
        if hasattr(self.regularizer, 'update_all_schedules'):
            # New system: update all schedules at once
            schedules = self.regularizer.update_all_schedules(
                current_epoch=self.current_epoch,
                initial_temp=self.initial_temp
            )
            
            # Extract temperature (lambdas are already updated in regularizer)
            self.temperature = schedules['temperature']
            
            # Optional: print schedules
            if self.stats_tracker.print_stats:
                if schedules['mode'] == 'constrained':
                    print(f"[Epoch {self.current_epoch}] "
                          f"temp={self.temperature:.3f}, "
                          f"λ_dual={schedules['lambda']:.6f}, "
                          f"ε={schedules['constraint_target']:.2f}")
                else:
                    print(f"[Epoch {self.current_epoch}] "
                          f"temp={self.temperature:.3f}, "
                          f"λ={schedules['lambda']:.6f}, "
                          f"λ_ρ={schedules['lambda_density']:.6f}")
        
        else:
            # Legacy fallback
            print("[LENS] Warning: Using legacy schedule updates")
            
            if hasattr(self.regularizer, 'update_lambda'):
                self.regularizer.update_lambda(self.current_epoch)
            
            if hasattr(self.regularizer, 'update_temperature'):
                self.temperature = self.regularizer.update_temperature(
                    self.current_epoch,
                    self.initial_temp
                )
            
            if self.stats_tracker.print_stats:
                print(f"[Epoch {self.current_epoch}] "
                      f"temp={self.temperature:.3f}, "
                      f"λ={self.regularizer.current_lambda:.6f}")
    
    def set_epoch(self, epoch):
        """Set the current epoch number and update temperature/lambda"""
        self.current_epoch = epoch
        self.regularizer.current_epoch = epoch
        self.update_temperature_and_lambda()
    
    def update_l0_params(self, gamma=None, zeta=None, beta_l0=None, baseline_ema=None):
        """Update L0 regularization parameters"""
        if self.l0_params is not None:
            if isinstance(self.l0_params, L0RegularizerParams):
                # Hard-Concrete
                self.l0_params.update_params(gamma, zeta, beta_l0)
                print(f"[LENS] Updated Hard-Concrete parameters: "
                      f"gamma={self.l0_params.gamma}, zeta={self.l0_params.zeta}, "
                      f"beta={self.l0_params.beta_l0}")
                
            elif isinstance(self.l0_params, ARML0RegularizerParams):
                # ARM
                if gamma is not None:
                    self.l0_params.gamma = gamma
                if zeta is not None:
                    self.l0_params.zeta = zeta
                if baseline_ema is not None:
                    self.l0_params.baseline_ema = baseline_ema
                
                print(f"[LENS] Updated ARM parameters: "
                      f"gamma={self.l0_params.gamma}, zeta={self.l0_params.zeta}, "
                      f"baseline_ema={self.l0_params.baseline_ema}")
            
            # Update in components
            if hasattr(self.edge_scorer, 'l0_params'):
                self.edge_scorer.l0_params = self.l0_params
            if hasattr(self.regularizer, 'l0_params'):
                self.regularizer.l0_params = self.l0_params
        else:
            print("[LENS] Warning: Attempting to update L0 parameters but not using L0 regularization")
    
    def aggregate(self, node_feat, adj_matrix, edge_weights):
        """Neighborhood aggregation with learned edge weights + self-loops"""
        # Add self-loops
        eye = torch.eye(adj_matrix.size(1), device=adj_matrix.device).unsqueeze(0)  # [1, N, N]
        adj_matrix = adj_matrix + eye
        adj_matrix = torch.clamp(adj_matrix, max=1.0)  # ensure no values > 1

        # Apply edge weights to adjacency matrix
        weighted_adj = adj_matrix * edge_weights

        # Row-normalize weighted adjacency matrix
        row_sum = torch.sum(weighted_adj, dim=2, keepdim=True) + 1e-8
        norm_adj = weighted_adj / row_sum

        # Aggregate neighbor features
        return torch.bmm(norm_adj, node_feat)
    
    def forward(self, node_feat, labels, adjs, masks=None, 
                return_edge_weights_anti=False):
        """
        Forward pass with support for Hard-Concrete, ARM, and STE
        WITH PENALTY OR CONSTRAINED MODE
        
        Args:
            node_feat: Node features [B, N, D]
            labels: Ground truth labels [B]
            adjs: Adjacency matrices [B, N, N]
            masks: Node masks [B, N] (optional)
            return_edge_weights_anti: If True, also return antithetic edge weights (for ARM)
        
        Returns:
            logits: Predicted logits [B, num_classes]
            labels: Ground truth labels [B]
            total_loss: Combined classification + regularization loss
            weighted_adj: Edge-weighted adjacency [B, N, N]
            edge_weights_anti: Antithetic edge weights (only if return_edge_weights_anti=True)
        """
        # Normalize node features
        node_feat = F.normalize(node_feat, p=2, dim=2)
        batch_size = node_feat.shape[0]
        
        # Compute edge weights
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
            # ARM returns (edge_weights, edge_weights_anti, logAlpha)
            edge_weights, edge_weights_anti, logAlpha = edge_scorer_outputs
            self.last_edge_weights_anti = edge_weights_anti
        else:
            # Hard-Concrete, STE, or eval mode returns (edge_weights, logAlpha)
            if isinstance(edge_scorer_outputs, tuple) and len(edge_scorer_outputs) == 2:
                edge_weights, logAlpha = edge_scorer_outputs
            else:
                edge_weights = edge_scorer_outputs
                logAlpha = None
            edge_weights_anti = None
        
        # Aggregate neighbor features
        h = self.aggregate(node_feat, adjs, edge_weights)
        
        # Apply GNN layer
        h = self.conv(h)
        h = F.relu(h)
        
        # Edge-weighted attention pooling
        graph_rep = self.pooling.edge_weighted_attention_pooling(h, edge_weights, adjs, masks)
        
        # Classification
        logits = self.classifier(graph_rep)
        
        # Compute classification loss
        cls_loss = F.cross_entropy(logits, labels)
        
        # ============================================
        # 🆕 COMPUTE REGULARIZATION (PENALTY OR CONSTRAINED)
        # ============================================
        if self.use_l0 and logAlpha is not None:
            # Compute current density for stats
            from model.EGL_L0_Reg import compute_density
            current_density = compute_density(edge_weights, adjs)
            
            # Compute L0 penalty based on method
            if self.l0_method == 'hard-concrete':
                from model.L0Utils import get_loss2
                l0_penalty = get_loss2(logAlpha, params=self.l0_params).sum()
                
            elif self.l0_method == 'arm':
                from model.L0Utils_ARM import get_expected_l0_arm
                l0_penalty = get_expected_l0_arm(logAlpha, self.l0_params)
                
            elif self.l0_method == 'ste':
                from model.L0Utils_STE import get_expected_l0_ste
                l0_penalty = get_expected_l0_ste(logAlpha, l0_params=self.l0_params)
            
            else:
                raise ValueError(f"Unknown l0_method: {self.l0_method}")
            
            # Compute full regularization (handles both modes)
            reg_loss, reg_stats = self.regularizer.compute_regularization_with_l0(
                l0_penalty=l0_penalty,
                edge_weights=edge_weights,
                adj_matrix=adjs,
                return_stats=True
            )
            
            # 🆕 Store constraint violation for dual update (constrained mode)
            if self.use_constrained:
                self.last_constraint_violation = reg_stats.get('constraint_violation', 0)
            
            # Extract components for tracking
            l0_loss = reg_stats.get('l0_loss', 0)
            density_loss = reg_stats.get('density_loss', 0)
            lambda_eff = reg_stats.get('lambda_eff', self.regularizer.get_effective_lambda())
            
        else:
            # Standard regularization (if not using L0)
            reg_loss = self.regularizer.compute_regularization(edge_weights, adjs)
            l0_loss = reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss
            density_loss = 0
            lambda_eff = self.regularizer.get_effective_lambda()
            current_density = 0
            self.last_constraint_violation = None
        
        # Total loss
        total_loss = cls_loss + reg_loss
        
        # ============================================
        # 🆕 TRACK STATISTICS INCLUDING CONSTRAINED MODE
        # ============================================
        # Update base stats
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
            
            # Constrained mode tracking
            if self.use_constrained and self.last_constraint_violation is not None:
                self.stats_tracker.constraint_violation_history.append(self.last_constraint_violation)
                self.stats_tracker.dual_lambda_history.append(self.regularizer.dual_lambda)
        
        # Return format
        if return_edge_weights_anti and edge_weights_anti is not None:
            return logits, labels, total_loss, adjs * edge_weights, edge_weights_anti
        else:
            return logits, labels, total_loss, adjs * edge_weights
    
    def forward_arm_antithetic(self, node_feat, labels, adjs, edge_weights_anti, masks=None):
        """
        Forward pass with pre-computed antithetic edge weights (for ARM)
        
        Args:
            node_feat: Node features [B, N, D]
            labels: Ground truth labels [B]
            adjs: Adjacency matrices [B, N, N]
            edge_weights_anti: Pre-computed antithetic edge weights [B, N, N]
            masks: Node masks [B, N] (optional)
        
        Returns:
            logits: Predicted logits [B, num_classes]
            cls_loss: Classification loss (scalar)
        """
        # Normalize node features
        node_feat = F.normalize(node_feat, p=2, dim=2)
        
        # Use provided antithetic edge weights
        h = self.aggregate(node_feat, adjs, edge_weights_anti)
        
        # Apply GNN layer
        h = self.conv(h)
        h = F.relu(h)
        
        # Edge-weighted attention pooling
        graph_rep = self.pooling.edge_weighted_attention_pooling(
            h, edge_weights_anti, adjs, masks
        )
        
        # Classification
        logits = self.classifier(graph_rep)
        
        # Compute classification loss
        cls_loss = F.cross_entropy(logits, labels)
        
        return logits, cls_loss
    
    def get_arm_gradient_loss(self, cls_loss_b, cls_loss_b_anti, logAlpha):
        """
        Compute ARM gradient contribution
        
        Args:
            cls_loss_b: Classification loss with sampled gates
            cls_loss_b_anti: Classification loss with antithetic gates
            logAlpha: Edge logits
        
        Returns:
            arm_loss: ARM gradient estimate (scalar)
        """
        if not self.use_l0 or self.l0_method != 'arm':
            return torch.tensor(0.0, device=logAlpha.device)
        
        arm_loss = compute_arm_loss(cls_loss_b, cls_loss_b_anti, logAlpha, self.l0_params)
        return arm_loss
    
    # Delegate visualization methods to stats tracker
    def save_graph_analysis(self, epoch, batch_idx, save_dir='./'):
        return self.stats_tracker.save_graph_analysis(
            epoch, batch_idx, save_dir, self.regularizer.get_effective_lambda(),
            self.temperature, self.warmup_epochs
        )
    
    def plot_edge_weight_distribution(self, weighted_adj, epoch, batch_idx=0, save_dir='./'):
        return self.stats_tracker.plot_edge_weight_distribution(
            weighted_adj, epoch, batch_idx, save_dir, self.regularizer.get_effective_lambda(),
            self.temperature, self.current_epoch, self.warmup_epochs
        )
    
    def plot_stats(self, save_path='stats.png'):
        return self.stats_tracker.plot_stats(
            save_path, self.regularizer.reg_mode, self.regularizer.base_lambda,
            self.warmup_epochs
        )
    
    def save_sparsification_report(self, epoch, save_dir='./'):
        return self.stats_tracker.save_sparsification_report(
            epoch, save_dir, self.regularizer.get_effective_lambda(), self.temperature,
            self.warmup_epochs
        )
    
    def get_edge_statistics(self):
        """Get current edge statistics for logging"""
        stats = {}
        
        if hasattr(self.stats_tracker, 'edge_density_history') and \
           len(self.stats_tracker.edge_density_history) > 0:
            stats['edge_density'] = self.stats_tracker.edge_density_history[-1]
            stats['mean_edge_weight'] = self.stats_tracker.mean_edge_weight_history[-1]
            stats['std_edge_weight'] = self.stats_tracker.std_edge_weight_history[-1]
        
        # Add density control stats
        if hasattr(self.stats_tracker, 'current_density_history') and \
           len(self.stats_tracker.current_density_history) > 0:
            stats['current_density'] = self.stats_tracker.current_density_history[-1]
        
        if hasattr(self.stats_tracker, 'lambda_eff_history') and \
           len(self.stats_tracker.lambda_eff_history) > 0:
            stats['lambda_eff'] = self.stats_tracker.lambda_eff_history[-1]
        
        # 🆕 Constrained mode stats
        if self.use_constrained:
            if hasattr(self.stats_tracker, 'constraint_violation_history') and \
               len(self.stats_tracker.constraint_violation_history) > 0:
                stats['constraint_violation'] = self.stats_tracker.constraint_violation_history[-1]
            
            if hasattr(self.stats_tracker, 'dual_lambda_history') and \
               len(self.stats_tracker.dual_lambda_history) > 0:
                stats['dual_lambda'] = self.stats_tracker.dual_lambda_history[-1]
        
        return stats
    
    def __repr__(self):
        """String representation"""
        if self.use_constrained:
            mode_info = (f"\n  Mode: CONSTRAINED"
                        f"\n  Constraint Target: {self.regularizer.constraint_target*100:.1f}%"
                        f"\n  Dual Lambda: {self.regularizer.dual_lambda:.6f}"
                        f"\n  Dual Restarts: {'enabled' if self.regularizer.enable_dual_restarts else 'disabled'}")
        else:
            mode_info = (f"\n  Mode: PENALTY"
                        f"\n  Base Lambda: {self.regularizer.base_lambda}"
                        f"\n  Target Density: {self.regularizer.target_density*100:.1f}%"
                        f"\n  Adaptive Lambda: {'enabled' if self.regularizer.enable_adaptive_lambda else 'disabled'}")
        
        return (f"ImprovedEdgeGNN(LENS):\n"
                f"  Regularization: {self.reg_mode}\n"
                f"  L0 Method: {self.l0_method if self.use_l0 else 'N/A'}"
                f"{mode_info}\n"
                f"  Warmup Epochs: {self.warmup_epochs}\n"
                f"  Classes: {self.num_classes}")
