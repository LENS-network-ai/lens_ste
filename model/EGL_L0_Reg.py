

"""
EGLasso Regularization with Target Density Control and Constrained Optimization

Supports:
- L0 regularization with Hard-Concrete (default)
- L0 regularization with ARM
- Exclusive Group Lasso (EGL)
- Target density control with adaptive lambda (PENALTY MODE)
- Constrained optimization with Lagrangian dual variables (CONSTRAINED MODE)

Author: Updated with constrained optimization support
"""

import torch
import torch.nn as nn
import numpy as np
from model.L0Utils import get_loss2, L0RegularizerParams
from model.L0Utils_ARM import get_expected_l0_arm, ARML0RegularizerParams


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_density(edge_weights, adj_matrix):
    """
    Compute current graph density: ρ = (1/|E|) Σ_{(i,j)∈E} z_ij
    
    Args:
        edge_weights: Edge gate values [batch_size, num_nodes, num_nodes]
        adj_matrix: Original adjacency matrix [batch_size, num_nodes, num_nodes]
    
    Returns:
        density: Scalar tensor - fraction of edges retained (0 to 1)
    """
    # Create mask for valid edges
    edge_mask = (adj_matrix > 0).float()
    
    # Count total possible edges
    num_edges = edge_mask.sum()
    
    # Count active edges (weighted sum)
    active_edges = (edge_weights * edge_mask).sum()
    
    # Compute density (with epsilon for numerical stability)
    density = active_edges / (num_edges + 1e-8)
    
    return density


def compute_keep_probabilities(logits, l0_params, l0_method='hard-concrete'):
    """
    Compute keep probabilities π_ij from logits
    
    Args:
        logits: Edge logits [batch_size, num_nodes, num_nodes]
        l0_params: L0 regularizer parameters
        l0_method: 'hard-concrete' or 'arm'
    
    Returns:
        keep_probs: Keep probabilities [batch_size, num_nodes, num_nodes]
    """
    if l0_method == 'hard-concrete':
        if isinstance(l0_params, L0RegularizerParams):
            # π_ij = σ(log α - τ log(-γ/ζ))
            keep_probs = l0_params.sig(logits - l0_params.const1)
        else:
            keep_probs = torch.sigmoid(logits)
    
    elif l0_method == 'arm':
        # For ARM, keep probability is sigmoid
        keep_probs = torch.sigmoid(logits)
    
    else:
        # Fallback
        keep_probs = torch.sigmoid(logits)
    
    return keep_probs


# ============================================================================
# MAIN REGULARIZATION CLASS
# ============================================================================

class EGLassoRegularization:
    """
    Edge-Graph Lasso Regularization with TWO MODES:
    
    1. PENALTY MODE (default):
       - Scheduled lambda with adaptive scaling
       - Density loss for explicit sparsity targets
       - L_total = λ_eff * L_L0 + λ_ρ * (ρ - ρ_target)²
    
    2. CONSTRAINED MODE (paper approach):
       - Lagrangian optimization with dual variables
       - Constraint: E[||z||_0] / |E| ≤ ε
       - L_total = L_task + λ_dual * (g_const - ε)
       - λ_dual updated via gradient ascent
    """
    
    def __init__(self, 
                 lambda_reg=0.0001, 
                 lambda_density=0.03,
                 target_density=0.30,
                 reg_mode='l0', 
                 warmup_epochs=15, 
                 ramp_epochs=20,
                 l0_params=None, 
                 l0_method='hard-concrete',
                 alpha_min=0.2,
                 alpha_max=2.0,
                 enable_adaptive_lambda=True,
                 enable_density_loss=True,
                 # 🆕 CONSTRAINED OPTIMIZATION PARAMETERS
                 use_constrained=False,
                 dual_lr=1e-3,
                 enable_dual_restarts=True,
                 constraint_target=0.30):
        """
        Initialize the regularization module
        
        PENALTY MODE Args:
            lambda_reg: Base L0 regularization strength (λ_base)
            lambda_density: Density loss weight (λ_ρ)
            target_density: Target edge retention rate (ρ_target) in [0, 1]
            enable_adaptive_lambda: Whether to use adaptive lambda mechanism
            enable_density_loss: Whether to use density loss
            alpha_min: Minimum adaptive scaling factor
            alpha_max: Maximum adaptive scaling factor
        
        CONSTRAINED MODE Args:
            use_constrained: If True, use constrained optimization (paper method)
            dual_lr: Learning rate for dual variable (η_dual)
            enable_dual_restarts: Enable dual restart heuristic
            constraint_target: Constraint level ε (expected L0 density)
        
        Common Args:
            reg_mode: Regularization type ('l0' or 'egl')
            warmup_epochs: Number of warmup epochs
            ramp_epochs: Number of ramp-up epochs after warmup
            l0_params: L0RegularizerParams or ARML0RegularizerParams instance
            l0_method: 'hard-concrete' or 'arm'
        """
        # ====================================================================
        # CORE REGULARIZATION PARAMETERS
        # ====================================================================
        self.base_lambda = lambda_reg
        self.current_lambda = 0.0  # Will increase during training (penalty mode)
        self.reg_mode = reg_mode
        self.l0_method = l0_method
        self.logits_storage = {}  # For L0 regularization
        
        # ====================================================================
        # OPTIMIZATION MODE
        # ====================================================================
        self.use_constrained = use_constrained
        
        # ====================================================================
        # PENALTY MODE PARAMETERS
        # ====================================================================
        self.target_density = target_density  # ρ_target
        self.base_lambda_density = lambda_density  # λ_ρ^base
        self.current_lambda_density = 0.0  # Will increase during training
        self.enable_adaptive_lambda = enable_adaptive_lambda and not use_constrained
        self.enable_density_loss = enable_density_loss and not use_constrained
        
        # Adaptive lambda bounds (penalty mode only)
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        
        # ====================================================================
        # CONSTRAINED MODE PARAMETERS
        # ====================================================================
        self.dual_lr = dual_lr  # η_dual in paper
        self.enable_dual_restarts = enable_dual_restarts
        self.constraint_target = constraint_target  # ε in paper
        
        # Dual variable (Lagrange multiplier) - λ_co in paper
        self.dual_lambda = 0.0  # Initialize at 0 as in paper (Eq. 5)
        
        # Track constraint violations
        self.constraint_violation_history = []
        self.dual_lambda_history = []
        self.constraint_satisfied_history = []
        
        # ====================================================================
        # SCHEDULE PARAMETERS
        # ====================================================================
        self.current_epoch = 0
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs
        
        # ====================================================================
        # L0 REGULARIZER PARAMETERS
        # ====================================================================
        if l0_params is not None:
            self.l0_params = l0_params
        else:
            # Create default params based on method
            if l0_method == 'hard-concrete':
                self.l0_params = L0RegularizerParams()
            elif l0_method == 'arm':
                self.l0_params = ARML0RegularizerParams()
            else:
                self.l0_params = L0RegularizerParams()
        
        # ====================================================================
        # LOGGING
        # ====================================================================
        print(f"\n{'='*70}")
        if use_constrained:
            print(f"[EGLassoReg] CONSTRAINED OPTIMIZATION MODE")
        else:
            print(f"[EGLassoReg] PENALTY MODE with Density Control")
        print(f"{'='*70}")
        print(f"  Regularization Mode: {reg_mode}")
        print(f"  L0 Method: {l0_method}")
        
        if use_constrained:
            print(f"\n  🎯 CONSTRAINED PARAMETERS:")
            print(f"  Constraint Target (ε): {constraint_target*100:.1f}%")
            print(f"  Dual Learning Rate (η_dual): {dual_lr}")
            print(f"  Dual Restarts: {'Enabled' if enable_dual_restarts else 'Disabled'}")
            print(f"  Initial λ_dual: {self.dual_lambda}")
        else:
            print(f"\n  🎯 PENALTY PARAMETERS:")
            print(f"  Base λ: {lambda_reg:.6f}")
            print(f"  Base λ_ρ: {lambda_density:.6f}")
            print(f"  Target Density: {target_density*100:.1f}%")
            print(f"  Adaptive λ Range: [{alpha_min}, {alpha_max}]")
            print(f"  Adaptive λ Enabled: {enable_adaptive_lambda}")
            print(f"  Density Loss Enabled: {enable_density_loss}")
        
        print(f"\n  Warmup Epochs: {warmup_epochs}")
        print(f"  Ramp Epochs: {ramp_epochs}")
        print(f"{'='*70}\n")
    
    # ========================================================================
    # LOGIT STORAGE FOR L0 REGULARIZATION
    # ========================================================================
    
    def clear_logits(self):
        """Clear stored logits for L0 regularization"""
        self.logits_storage = {}
    
    def store_logits(self, batch_idx, logits):
        """
        Store logits for a batch for L0 regularization
        
        Args:
            batch_idx: Batch index
            logits: Edge logits [num_nodes, num_nodes]
        """
        self.logits_storage[batch_idx] = logits
    
    # ========================================================================
    # CONSTRAINED OPTIMIZATION: DUAL VARIABLE UPDATE
    # ========================================================================
    
    def update_dual_variable(self, constraint_violation):
        """
        Update dual variable (Lagrange multiplier) via projected gradient ascent
        
        From paper Eq. (5):
            λ̂^{t+1} = λ^t + η_dual * (g_const(φ) - ε)
            λ^{t+1} = max(0, λ̂^{t+1})
        
        With dual restarts (paper Eq. 6):
            λ^{t+1} = 0 if constraint satisfied, else standard update
        
        Args:
            constraint_violation: Current constraint violation (g_const - ε)
        
        Returns:
            updated_lambda: New dual variable value
        """
        if self.current_epoch < self.warmup_epochs:
            self.dual_lambda = 0.0
            return 0.0
        if not self.use_constrained:
            return self.current_lambda
        
        # Convert to scalar if needed
        if isinstance(constraint_violation, torch.Tensor):
            violation_value = constraint_violation.item()
        else:
            violation_value = constraint_violation
        
        # Check if constraint is satisfied (g_const ≤ ε)
        constraint_satisfied = (violation_value <= 0)
        
        if self.enable_dual_restarts and constraint_satisfied:
            # Paper Eq. (6): Dual restart - reset to 0 when constraint satisfied
            self.dual_lambda = 0.0
        else:
            # Paper Eq. (5): Standard gradient ascent update
            lambda_hat = self.dual_lambda + self.dual_lr * violation_value
            
            # Project to non-negative orthant
            self.dual_lambda = max(0.0, lambda_hat)
        
        # Track history
        self.dual_lambda_history.append(self.dual_lambda)
        self.constraint_violation_history.append(violation_value)
        self.constraint_satisfied_history.append(constraint_satisfied)
        
        return self.dual_lambda
    
    def get_effective_lambda(self):
        """
        Get effective lambda for current iteration
        
        Returns:
            CONSTRAINED MODE: dual_lambda (learned via GDA)
            PENALTY MODE: current_lambda (scheduled)
        """
        if self.use_constrained:
            return self.dual_lambda
        else:
            return self.current_lambda
    
    # ========================================================================
    # SCHEDULE UPDATES (Penalty Mode)
    # ========================================================================
    
    def update_lambda(self, current_epoch):
        """
        Update L0 lambda regularization strength: λ(e)
        
        Three-phase linear schedule (PENALTY MODE ONLY):
        - Phase 1 (warmup): Linear from 0 to 0.1 × λ_base over E_warmup epochs
        - Phase 2 (ramp): Linear from 0.1 × λ_base to λ_base over E_ramp epochs
        - Phase 3 (plateau): Fixed at λ_base
        
        Args:
            current_epoch: Current training epoch
        """
        if self.use_constrained:
            # In constrained mode, don't use scheduled lambda
            self.current_lambda = 0.0
            return
        
        self.current_epoch = current_epoch
        
        if current_epoch < self.warmup_epochs:
            # Phase 1: Warmup - linear increase to 10% of base
            progress = current_epoch / self.warmup_epochs
            self.current_lambda = progress * 0.1 * self.base_lambda
        
        elif current_epoch < self.warmup_epochs + self.ramp_epochs:
            # Phase 2: Ramp - linear increase from 10% to 100%
            post_warmup_epochs = current_epoch - self.warmup_epochs
            progress = post_warmup_epochs / self.ramp_epochs
            self.current_lambda = self.base_lambda * (0.1 + 0.9 * progress)
        
        else:
            # Phase 3: Plateau - fixed at full strength
            self.current_lambda = self.base_lambda
    
    def update_lambda_density(self, current_epoch):
        """
        Update density lambda: λ_ρ(e) (PENALTY MODE ONLY)
        
        Two-phase schedule:
        - Phase 1 (warmup): Linear from 0 to 0.5 × λ_ρ^base over E_warmup epochs
        - Phase 2 (ramp): Linear from 0.5 × λ_ρ^base to λ_ρ^base over E_ramp/2 epochs
        
        Args:
            current_epoch: Current training epoch
        """
        if self.use_constrained:
            # In constrained mode, don't use density loss
            self.current_lambda_density = 0.0
            return
        
        if current_epoch < self.warmup_epochs:
            # Phase 1: Warmup - linear increase to 50% of base
            progress = current_epoch / self.warmup_epochs
            self.current_lambda_density = progress * 0.5 * self.base_lambda_density
        
        else:
            # Phase 2: Ramp - linear increase from 50% to 100%
            post_warmup_epochs = current_epoch - self.warmup_epochs
            ramp_duration = self.ramp_epochs / 2  # Faster ramp for density
            
            if post_warmup_epochs < ramp_duration:
                progress = post_warmup_epochs / ramp_duration
                scale = 0.5 + 0.5 * progress
            else:
                scale = 1.0
            
            self.current_lambda_density = scale * self.base_lambda_density
    
    def update_temperature(self, current_epoch, initial_temp=5.0):
        """
        Update temperature based on current epoch: τ(e)
        
        Three-phase cosine annealing:
        - Phase 1: Constant at τ_init during warmup
        - Phase 2: Cosine decay from τ_init to τ_min
        - Phase 3: Plateau at τ_min
        
        Args:
            current_epoch: Current training epoch
            initial_temp: Initial temperature (τ_init)
        
        Returns:
            temperature: Current temperature value
        """
        tau_init = initial_temp
        tau_min = 1.0
        t_warmup = self.warmup_epochs
        t_anneal = self.warmup_epochs + self.ramp_epochs
        mu_min = 0.1
        
        if current_epoch < t_warmup:
            # Phase 1: Constant during warmup
            temperature = tau_init
        
        elif current_epoch <= t_anneal:
            # Phase 2: Cosine annealing
            progress = (current_epoch - t_warmup) / (t_anneal - t_warmup)
            progress = min(1.0, max(0.0, progress))
            mu = mu_min + (1 - mu_min) * (np.cos(progress * np.pi) + 1) / 2
            temperature = max(tau_min, tau_init * mu)
        
        else:
            # Phase 3: Plateau at minimum
            temperature = tau_min
        
        return temperature
    
    def update_all_schedules(self, current_epoch, initial_temp=5.0):
        """
        Convenience function to update all schedules at once
        
        CONSTRAINED MODE: Only update temperature (λ is learned via GDA)
        PENALTY MODE: Update lambda, lambda_density, and temperature
        
        Args:
            current_epoch: Current training epoch
            initial_temp: Initial temperature
        
        Returns:
            dict with updated values
        """
        self.current_epoch = current_epoch
        temperature = self.update_temperature(current_epoch, initial_temp)
        
        if self.use_constrained:
            # Constrained mode: dual variable is updated via gradient ascent
            return {
                'lambda': self.dual_lambda,
                'lambda_density': 0.0,
                'temperature': temperature,
                'epoch': current_epoch,
                'mode': 'constrained',
                'constraint_target': self.constraint_target,
                'dual_restarts': self.enable_dual_restarts,
            }
        else:
            # Penalty mode: scheduled lambda updates
            self.update_lambda(current_epoch)
            self.update_lambda_density(current_epoch)
            
            return {
                'lambda': self.current_lambda,
                'lambda_density': self.current_lambda_density,
                'temperature': temperature,
                'epoch': current_epoch,
                'mode': 'penalty',
            }
    
    # ========================================================================
    # ADAPTIVE LAMBDA MECHANISM (Penalty Mode Only)
    # ========================================================================
    
    def compute_adaptive_lambda(self, current_density):
        """
        Compute adaptive lambda: λ_eff(t) = λ_base · α(t)
        (PENALTY MODE ONLY)
        
        Adaptive scaling factor:
            α(t) = clip(1 + [ρ(t) - ρ_target], α_min, α_max)
        
        Args:
            current_density: Current graph density ρ(t) ∈ [0, 1]
        
        Returns:
            lambda_eff: Effective lambda value (scalar or tensor)
            alpha: Adaptive scaling factor (float)
        """
        if self.use_constrained or not self.enable_adaptive_lambda:
            return self.current_lambda, 1.0
        
        # Convert density to scalar if it's a tensor
        if isinstance(current_density, torch.Tensor):
            density_value = current_density.item()
        else:
            density_value = current_density
        
        # Compute density error
        density_error = density_value - self.target_density
        
        # Compute adaptive scaling factor
        alpha = 1.0 + density_error
        
        # Clip to bounds
        alpha = max(self.alpha_min, min(self.alpha_max, alpha))
        
        # Compute effective lambda
        lambda_eff = self.current_lambda * alpha
        
        return lambda_eff, alpha
    
    # ========================================================================
    # DENSITY LOSS (Penalty Mode Only)
    # ========================================================================
    
    def compute_density_loss(self, current_density):
        """
        Compute density loss: L_density = λ_ρ · (ρ - ρ_target)²
        (PENALTY MODE ONLY)
        
        Args:
            current_density: Current graph density ρ(t)
        
        Returns:
            density_loss: Density loss value (tensor)
        """
        if self.use_constrained or not self.enable_density_loss or self.current_lambda_density == 0.0:
            device = current_density.device if isinstance(current_density, torch.Tensor) else 'cpu'
            return torch.tensor(0.0, device=device)
        
        # Compute density error
        density_error = current_density - self.target_density
        
        # Quadratic penalty
        density_loss = self.current_lambda_density * (density_error ** 2)
        
        return density_loss
    
    # ========================================================================
    # L0 REGULARIZATION
    # ========================================================================
    
    def compute_l0_loss(self):
        """
        Compute L0 penalty from stored logits (without lambda scaling)
        
        Returns:
            l0_loss: L0 penalty (tensor)
        """
        if len(self.logits_storage) == 0:
            return torch.tensor(0.0)
        
        # Get device from first stored logits
        first_key = next(iter(self.logits_storage))
        device = self.logits_storage[first_key].device
        
        l0_loss = torch.tensor(0.0, device=device)
        
        for batch_idx, logits in self.logits_storage.items():
            # Calculate L0 loss based on method
            if self.l0_method == 'hard-concrete':
                batch_l0 = get_loss2(logits, params=self.l0_params).sum()
            
            elif self.l0_method == 'arm':
                batch_l0 = get_expected_l0_arm(logits, self.l0_params)
            
            else:
                # Fallback to hard-concrete
                batch_l0 = get_loss2(logits, params=self.l0_params).sum()
            
            l0_loss = l0_loss + batch_l0
        
        # Average over batches
        l0_loss = l0_loss / len(self.logits_storage)
        
        return l0_loss
    
    # ========================================================================
    # MAIN REGULARIZATION COMPUTATION
    # ========================================================================
    
    def compute_regularization_with_l0(self, l0_penalty, edge_weights, adj_matrix, 
                                        return_stats=False):
        """
        Compute regularization when L0 penalty is already computed
        
        CONSTRAINED MODE (use_constrained=True):
            L_reg = λ_dual * (g_const - ε)
            where:
              - g_const = E[||z||_0] / |E|  (expected L0 density)
              - ε = constraint_target
              - λ_dual is updated via gradient ascent (paper Eq. 5-6)
        
        PENALTY MODE (use_constrained=False):
            L_reg = λ_eff * L_L0 + λ_ρ * (ρ - ρ_target)²
            where:
              - λ_eff = λ_base * α (adaptive)
              - λ_ρ = density lambda (scheduled)
        
        Args:
            l0_penalty: Pre-computed L0 penalty (expected number of active gates)
            edge_weights: Edge weights [B, N, N]
            adj_matrix: Original adjacency [B, N, N]
            return_stats: If True, return detailed statistics
        
        Returns:
            reg_loss: Total regularization loss
            stats: Dictionary with loss components (if return_stats=True)
        """
        device = edge_weights.device
        
        # Compute current density
        current_density = compute_density(edge_weights, adj_matrix)
        
        # ====================================================================
        # CONSTRAINED MODE
        # ====================================================================
        if self.use_constrained:
            # Convert L0 penalty to normalized form (expected density)
            # Paper: g_const(φ) = E_z|φ[||z||_0] / #(θ̃)
            edge_mask = (adj_matrix > 0).float()
            num_edges = edge_mask.sum()
            expected_l0_density = l0_penalty / (num_edges + 1e-8)
            
            # Compute constraint violation: g_const(φ) - ε
            # Constraint is: g_const(φ) ≤ ε
            constraint_violation = expected_l0_density - self.constraint_target
            
            # Use current dual variable (will be updated after backward pass)
            # Paper Eq. (4): L(θ̃, φ, λ_co) = f_obj(θ̃, φ) + λ_co * (g_const(φ) - ε)
            if self.current_epoch < self.warmup_epochs:
                    lambda_eff = 0.0
                    constraint_violation = torch.tensor(0.0, device=device)
            else:
                    lambda_eff = self.dual_lambda                                
           
            
            # Lagrangian term
            reg_loss = lambda_eff * constraint_violation
            
            # No density loss in pure constrained mode
            density_loss = torch.tensor(0.0, device=device)
            
            if return_stats:
                stats = {
                    'l0_loss': (lambda_eff * expected_l0_density).item() if isinstance(lambda_eff, float) else (lambda_eff * expected_l0_density).item(),
                    'density_loss': 0.0,
                    'reg_loss': reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss,
                    'lambda_eff': lambda_eff,
                    'dual_lambda': self.dual_lambda,
                    'constraint_violation': constraint_violation.item() if isinstance(constraint_violation, torch.Tensor) else constraint_violation,
                    'constraint_target': self.constraint_target,
                    'expected_l0_density': expected_l0_density.item() if isinstance(expected_l0_density, torch.Tensor) else expected_l0_density,
                    'current_density': current_density.item() if isinstance(current_density, torch.Tensor) else current_density,
                    'constraint_satisfied': constraint_violation.item() <= 0 if isinstance(constraint_violation, torch.Tensor) else constraint_violation <= 0,
                    'alpha': 1.0,  # Not used in constrained mode
                    'mode': 'constrained',
                }
                return reg_loss, stats
            
            return reg_loss
        
        # ====================================================================
        # PENALTY MODE
        # ====================================================================
        else:
            # Compute effective lambda (with adaptive mechanism if enabled)
            if self.enable_adaptive_lambda:
                lambda_eff, alpha = self.compute_adaptive_lambda(current_density)
            else:
                lambda_eff = self.current_lambda
                alpha = 1.0
            
            # L0 regularization loss
            l0_loss = lambda_eff * l0_penalty
            
            # Density loss (if enabled)
            if self.enable_density_loss and self.current_epoch >= self.warmup_epochs:
                density_deviation = torch.abs(current_density - self.target_density)
                density_loss = self.current_lambda_density * density_deviation
            else:
                density_loss = torch.tensor(0.0, device=device)
            
            # Total regularization
            reg_loss = l0_loss + density_loss
            
            if return_stats:
                stats = {
                    'l0_loss': l0_loss.item() if isinstance(l0_loss, torch.Tensor) else l0_loss,
                    'density_loss': density_loss.item() if isinstance(density_loss, torch.Tensor) else density_loss,
                    'reg_loss': reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss,
                    'lambda_eff': lambda_eff if isinstance(lambda_eff, float) else lambda_eff.item(),
                    'alpha': alpha,
                    'current_density': current_density.item() if isinstance(current_density, torch.Tensor) else current_density,
                    'constraint_violation': 0.0,  # Not applicable in penalty mode
                    'constraint_satisfied': True,  # Not applicable
                    'dual_lambda': 0.0,  # Not used
                    'mode': 'penalty',
                }
                return reg_loss, stats
            
            return reg_loss
    
    def compute_regularization(self, edge_weights, adj_matrix, 
                               return_stats=True):
        """
        Compute complete regularization loss with density control
        
        Args:
            edge_weights: Edge gate values [batch_size, num_nodes, num_nodes]
            adj_matrix: Original adjacency [batch_size, num_nodes, num_nodes]
            return_stats: Whether to return detailed statistics
        
        Returns:
            If return_stats=False:
                reg_loss: Total regularization loss (tensor)
            If return_stats=True:
                (reg_loss, stats_dict): Loss and statistics dictionary
        """
        # Check if regularization is active
        if (not self.use_constrained and self.current_lambda == 0.0) or not edge_weights.requires_grad:
            if return_stats:
                return torch.tensor(0.0, device=edge_weights.device), {
                    'total_reg_loss': 0.0,
                    'l0_loss': 0.0,
                    'density_loss': 0.0,
                    'lambda_eff': 0.0,
                    'alpha': 1.0,
                    'current_density': 0.0,
                    'target_density': self.target_density if not self.use_constrained else self.constraint_target,
                    'density_deviation': 0.0,
                    'mode': 'constrained' if self.use_constrained else 'penalty',
                }
            else:
                return torch.tensor(0.0, device=edge_weights.device)
        
        device = edge_weights.device
        
        # Compute current density
        current_density = compute_density(edge_weights, adj_matrix)
        
        # Compute L0 loss
        if self.reg_mode == 'l0':
            l0_loss = self.compute_l0_loss()
        
        elif self.reg_mode == 'egl':
            # Exclusive Group Lasso - group by nodes
            edge_mask = (adj_matrix > 0).float()
            
            # Sum weights per source node
            source_sum = torch.sum(edge_weights * edge_mask, dim=2)
            source_reg = torch.sum(source_sum ** 2)
            
            # Sum weights per target node
            target_sum = torch.sum(edge_weights * edge_mask, dim=1)
            target_reg = torch.sum(target_sum ** 2)
            
            # Average
            batch_size = adj_matrix.shape[0]
            l0_loss = (source_reg + target_reg) / (2 * batch_size)
        
        else:
            raise ValueError(f"Unsupported regularization mode: {self.reg_mode}")
        
        # Use the unified method
        return self.compute_regularization_with_l0(
            l0_loss, edge_weights, adj_matrix, return_stats
        )
    
    # ========================================================================
    # UTILITY FUNCTIONS
    # ========================================================================
    
    def get_keep_probabilities(self, logits):
        """
        Get keep probabilities from logits: π_ij
        
        Args:
            logits: Edge logits [batch_size, num_nodes, num_nodes]
        
        Returns:
            keep_probs: Keep probabilities [batch_size, num_nodes, num_nodes]
        """
        return compute_keep_probabilities(logits, self.l0_params, self.l0_method)
    
    def get_statistics(self):
        """
        Get current regularization statistics for logging
        
        Returns:
            stats: Dictionary of current state
        """
        base_stats = {
            'current_epoch': self.current_epoch,
            'reg_mode': self.reg_mode,
            'l0_method': self.l0_method,
            'use_constrained': self.use_constrained,
            'warmup_epochs': self.warmup_epochs,
            'ramp_epochs': self.ramp_epochs,
        }
        
        if self.use_constrained:
            base_stats.update({
                'mode': 'constrained',
                'dual_lambda': self.dual_lambda,
                'dual_lr': self.dual_lr,
                'constraint_target': self.constraint_target,
                'enable_dual_restarts': self.enable_dual_restarts,
                'num_violations': len(self.constraint_violation_history),
                'num_satisfied': sum(self.constraint_satisfied_history) if self.constraint_satisfied_history else 0,
            })
        else:
            base_stats.update({
                'mode': 'penalty',
                'base_lambda': self.base_lambda,
                'current_lambda': self.current_lambda,
                'base_lambda_density': self.base_lambda_density,
                'current_lambda_density': self.current_lambda_density,
                'target_density': self.target_density,
                'alpha_min': self.alpha_min,
                'alpha_max': self.alpha_max,
                'enable_adaptive_lambda': self.enable_adaptive_lambda,
                'enable_density_loss': self.enable_density_loss,
            })
        
        return base_stats
    
    def print_configuration(self):
        """Print current configuration"""
        stats = self.get_statistics()
        print(f"\n{'='*70}")
        print(f"EGLasso Regularization Configuration")
        print(f"{'='*70}")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
        print(f"{'='*70}\n")
    
    def __repr__(self):
        if self.use_constrained:
            return (f"EGLassoRegularization(CONSTRAINED):\n"
                    f"  mode={self.reg_mode},\n"
                    f"  l0_method={self.l0_method},\n"
                    f"  dual_lambda={self.dual_lambda:.6f},\n"
                    f"  dual_lr={self.dual_lr:.6f},\n"
                    f"  constraint_target={self.constraint_target:.2f},\n"
                    f"  dual_restarts={'enabled' if self.enable_dual_restarts else 'disabled'},\n"
                    f"  current_epoch={self.current_epoch},\n"
                    f"  warmup_epochs={self.warmup_epochs}\n"
                    f")")
        else:
            return (f"EGLassoRegularization(PENALTY):\n"
                    f"  mode={self.reg_mode},\n"
                    f"  l0_method={self.l0_method},\n"
                    f"  base_lambda={self.base_lambda:.6f},\n"
                    f"  current_lambda={self.current_lambda:.6f},\n"
                    f"  target_density={self.target_density:.2f},\n"
                    f"  current_lambda_density={self.current_lambda_density:.6f},\n"
                    f"  adaptive_lambda={'enabled' if self.enable_adaptive_lambda else 'disabled'},\n"
                    f"  density_loss={'enabled' if self.enable_density_loss else 'disabled'},\n"
                    f"  current_epoch={self.current_epoch},\n"
                    f"  warmup_epochs={self.warmup_epochs}\n"
                    f")")


# ============================================================================
# STANDALONE SCHEDULE CLASSES (for convenience)
# ============================================================================

class LambdaSchedule:
    """Standalone lambda schedule for use in training scripts"""
    
    def __init__(self, base_lambda=0.0001, warmup_epochs=15, ramp_epochs=20):
        self.base_lambda = base_lambda
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs
    
    def get_lambda(self, epoch):
        """Get lambda for given epoch"""
        if epoch < self.warmup_epochs:
            progress = epoch / self.warmup_epochs
            return progress * 0.1 * self.base_lambda
        elif epoch < self.warmup_epochs + self.ramp_epochs:
            post_warmup = epoch - self.warmup_epochs
            progress = post_warmup / self.ramp_epochs
            return self.base_lambda * (0.1 + 0.9 * progress)
        else:
            return self.base_lambda


class DensityLambdaSchedule:
    """Standalone density lambda schedule"""
    
    def __init__(self, base_lambda_density=0.03, warmup_epochs=15, ramp_epochs=10):
        self.base_lambda_density = base_lambda_density
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs
    
    def get_lambda_density(self, epoch):
        """Get density lambda for given epoch"""
        if epoch < self.warmup_epochs:
            progress = epoch / self.warmup_epochs
            return progress * 0.5 * self.base_lambda_density
        elif epoch < self.warmup_epochs + self.ramp_epochs:
            post_warmup = epoch - self.warmup_epochs
            progress = post_warmup / self.ramp_epochs
            scale = 0.5 + 0.5 * progress
            return scale * self.base_lambda_density
        else:
            return self.base_lambda_density


class TemperatureSchedule:
    """Standalone temperature schedule with cosine annealing"""
    
    def __init__(self, tau_init=5.0, tau_min=1.0, t_warmup=15, t_anneal=35):
        self.tau_init = tau_init
        self.tau_min = tau_min
        self.t_warmup = t_warmup
        self.t_anneal = t_anneal
        self.mu_min = 0.1
    
    def get_temperature(self, epoch):
        """Get temperature for given epoch"""
        if epoch < self.t_warmup:
            return self.tau_init
        elif epoch <= self.t_anneal:
            progress = (epoch - self.t_warmup) / (self.t_anneal - self.t_warmup)
            progress = min(1.0, max(0.0, progress))
            mu = self.mu_min + (1 - self.mu_min) * (np.cos(progress * np.pi) + 1) / 2
            return max(self.tau_min, self.tau_init * mu)
        else:
            return self.tau_min
