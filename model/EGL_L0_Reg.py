# model/EGL_L0_Reg.py
"""
EGLasso Regularization with support for Hard-Concrete and ARM L0 methods
"""

import torch
import numpy as np
from model.L0Utils import get_loss2, L0RegularizerParams
from model.L0Utils_ARM import get_expected_l0_arm, ARML0RegularizerParams


class EGLassoRegularization:
    """
    Edge-Graph Lasso Regularization
    
    Supports:
    - L0 regularization with Hard-Concrete (default)
    - L0 regularization with ARM
    - Exclusive Group Lasso (EGL)
    """
    
    def __init__(self, lambda_reg, reg_mode='l0', warmup_epochs=5, 
                 l0_params=None, l0_method='hard-concrete'):
        """
        Initialize the regularization module
        
        Args:
            lambda_reg: Base regularization strength (λ)
            reg_mode: Regularization type ('l0' or 'egl')
            warmup_epochs: Number of warmup epochs
            l0_params: L0RegularizerParams or ARML0RegularizerParams instance
            l0_method: 'hard-concrete' or 'arm' (only used if reg_mode='l0')
        """
        self.base_lambda = lambda_reg
        self.current_lambda = 0.0  # Will increase during training
        self.reg_mode = reg_mode
        self.current_epoch = 0
        self.warmup_epochs = warmup_epochs
        self.logits_storage = {}  # For L0 regularization
        self.l0_method = l0_method
        
        # Store L0 regularization parameters
        if l0_params is not None:
            self.l0_params = l0_params
        else:
            # Create default params based on method
            if l0_method == 'hard-concrete':
                self.l0_params = L0RegularizerParams()
            elif l0_method == 'arm':
                self.l0_params = ARML0RegularizerParams()
            else:
                self.l0_params = L0RegularizerParams()  # Default fallback
        
        print(f"[EGLassoReg] Initialized: mode={reg_mode}, l0_method={l0_method}, base_lambda={lambda_reg}")
        self.target_density = target_density  # ρ_target
        self.lambda_density = lambda_density  # λ_ρ
        self.alpha_min = 0.2
        self.alpha_max = 2.0
    
    def compute_adaptive_lambda(self, current_density):
        """
        Compute adaptive lambda: λ_eff = λ_base · α(t)
        where α(t) = clip(1 + (ρ - ρ_target), α_min, α_max)
        """
        # Density error
        density_error = current_density - self.target_density
        
        # ⭐ ADAPTIVE SCALING
        alpha = 1.0 + density_error
        alpha = torch.clamp(alpha, self.alpha_min, self.alpha_max)
        
        # Effective lambda
        lambda_eff = self.current_lambda * alpha
        
        return lambda_eff, alpha.item()
    def clear_logits(self):
        """Clear stored logits for L0 regularization"""
        self.logits_storage = {}
    
    def store_logits(self, batch_idx, logits):
        """Store logits for a batch for L0 regularization"""
        self.logits_storage[batch_idx] = logits
    
    # Add this to EGL_L0_Reg.py or as a standalone function

    def compute_density(edge_weights, adj_matrix):
     """
     Compute current density: ρ = (1/|E|) Σ_{(i,j)∈E} z_ij
    
     Args:
        edge_weights: [B, N, N] - gate values
        adj_matrix: [B, N, N] - original adjacency
    
     Returns:
        density: scalar - current edge retention rate
     """
     edge_mask = (adj_matrix > 0).float()
     num_edges = edge_mask.sum()
     active_edges = (edge_weights * edge_mask).sum()
     density = active_edges / (num_edges + 1e-8)
     return density
    def update_temperature(self, current_epoch, warmup_epochs, initial_temp):
        """
        Update temperature based on current epoch and return the new value
        
        Three-phase cosine annealing:
        - Phase 1: Constant at initial_temp during warmup
        - Phase 2: Cosine decay from initial_temp to tau_min
        - Phase 3: Plateau at tau_min
        """
        tau_init = initial_temp
        tau_min = 1.0
        t_warmup = warmup_epochs
        t_anneal = warmup_epochs + 20  # 20-epoch annealing window
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
    
    def update_lambda(self, current_epoch, warmup_epochs):
        """
        Update lambda regularization strength based on current epoch
        
        Three-phase linear schedule:
        - Phase 1 (warmup): Linear from 0 to 0.1 * base_lambda
        - Phase 2 (ramp): Linear from 0.1 * base_lambda to base_lambda over 20 epochs
        - Phase 3 (plateau): Fixed at base_lambda
        """
        self.current_epoch = current_epoch
        
        # Phase 1: Warmup
        if current_epoch < warmup_epochs:
            # Linear increase from 0 to 0.1 * base_lambda
            self.current_lambda = (current_epoch / warmup_epochs) * self.base_lambda * 0.1
        else:
            # Phase 2 & 3: Ramp and Plateau
            post_warmup_epochs = current_epoch - warmup_epochs
            min_lambda = self.base_lambda * 0.1
            max_lambda = self.base_lambda
            
            if post_warmup_epochs < 20:
                # Phase 2: Linear ramp over 20 epochs
                lambda_factor = min_lambda + (max_lambda - min_lambda) * (post_warmup_epochs / 20)
            else:
                # Phase 3: Plateau at max value
                lambda_factor = max_lambda
            
            self.current_lambda = lambda_factor
    
    def compute_regularization(self, edge_weights, adj_matrix):
        """
        Compute regularization loss based on selected mode
        
        Args:
            edge_weights: Edge weights tensor [batch_size, num_nodes, num_nodes]
            adj_matrix: Original adjacency matrix [batch_size, num_nodes, num_nodes]
            
        Returns:
            Regularization loss (scalar)
        """
        if self.current_lambda == 0.0 or not edge_weights.requires_grad:
            return torch.tensor(0.0, device=edge_weights.device)
        
        batch_size = adj_matrix.shape[0]
        
        # L0 regularization
        if self.reg_mode == 'l0':
            # L0 regularization based on stored logits
            reg_loss = torch.tensor(0.0, device=edge_weights.device)
            
            if len(self.logits_storage) == 0:
                # No logits stored - shouldn't happen in normal use
                # Return 0 to avoid breaking training
                return reg_loss
            
            for b in self.logits_storage:
                logits = self.logits_storage[b]
                
                # Calculate L0 loss based on method
                if self.l0_method == 'hard-concrete':
                    # Hard-Concrete: use get_loss2
                    l0_loss = get_loss2(logits, params=self.l0_params).sum()
                    
                elif self.l0_method == 'arm':
                    # ARM: use expected L0
                    l0_loss = get_expected_l0_arm(logits, self.l0_params)
                    
                else:
                    # Fallback to hard-concrete
                    l0_loss = get_loss2(logits, params=self.l0_params).sum()
                
                reg_loss = reg_loss + l0_loss
            
            # Average over batches
            if len(self.logits_storage) > 0:
                reg_loss = reg_loss / len(self.logits_storage)
        
        # Exclusive Group Lasso
        elif self.reg_mode == 'egl':
            # Create mask for existing edges
            edge_mask = (adj_matrix > 0).float()
            
            # Exclusive Group Lasso - group by nodes
            reg_loss = torch.tensor(0.0, device=edge_weights.device)
            
            # Sum weights per node (for each source node)
            source_sum = torch.sum(edge_weights * edge_mask, dim=2)  # [batch_size, num_nodes]
            source_reg = torch.sum(source_sum**2)
            
            # Sum weights per node (for each target node)
            target_sum = torch.sum(edge_weights * edge_mask, dim=1)  # [batch_size, num_nodes]
            target_reg = torch.sum(target_sum**2)
            
            # Average between source and target node regularization
            reg_loss = (source_reg + target_reg) / (2 * batch_size)
        
        else:
            raise ValueError(f"Unsupported regularization mode: {self.reg_mode}")
        
        # Return combined loss with current lambda scaling
        return self.current_lambda * reg_loss
    
    def compute_l0_from_logits(self, logits):
        """
        Compute L0 penalty from logits (without lambda scaling)
        
        Useful for monitoring sparsity
        
        Args:
            logits: Edge logits [batch_size, num_nodes, num_nodes]
            
        Returns:
            l0_penalty: L0 penalty value (scalar)
        """
        if self.l0_method == 'hard-concrete':
            penalty = get_loss2(logits, params=self.l0_params).sum()
        elif self.l0_method == 'arm':
            penalty = get_expected_l0_arm(logits, self.l0_params)
        else:
            penalty = get_loss2(logits, params=self.l0_params).sum()
        
        return penalty
    
    def get_keep_probabilities(self, logits):
        """
        Get keep probabilities from logits
        
        Args:
            logits: Edge logits [batch_size, num_nodes, num_nodes]
            
        Returns:
            keep_probs: Keep probabilities [batch_size, num_nodes, num_nodes]
        """
        if self.l0_method == 'hard-concrete':
            # π_ij = σ(log α - const1)
            if isinstance(self.l0_params, L0RegularizerParams):
                keep_probs = self.l0_params.sig(logits - self.l0_params.const1)
            else:
                # Fallback
                keep_probs = torch.sigmoid(logits)
        
        elif self.l0_method == 'arm':
            # For ARM, keep prob is sigmoid
            keep_probs = torch.sigmoid(logits)
        
        else:
            # Fallback
            keep_probs = torch.sigmoid(logits)
        
        return keep_probs
    
    def get_statistics(self):
        """Get current regularization statistics for logging"""
        return {
            'current_lambda': self.current_lambda,
            'base_lambda': self.base_lambda,
            'current_epoch': self.current_epoch,
            'reg_mode': self.reg_mode,
            'l0_method': self.l0_method,
            'warmup_epochs': self.warmup_epochs,
        }
    
    def __repr__(self):
        return (f"EGLassoRegularization(\n"
                f"  mode={self.reg_mode},\n"
                f"  l0_method={self.l0_method},\n"
                f"  base_lambda={self.base_lambda},\n"
                f"  current_lambda={self.current_lambda:.6f},\n"
                f"  current_epoch={self.current_epoch},\n"
                f"  warmup_epochs={self.warmup_epochs}\n"
                f")")


# Standalone schedule classes for convenience
class LambdaSchedule:
    """
    Standalone lambda schedule for use in training scripts
    """
    
    def __init__(self, base_lambda=0.01, warmup_epochs=5, ramp_epochs=20):
        """
        Args:
            base_lambda: Target lambda value
            warmup_epochs: Warmup duration
            ramp_epochs: Ramp duration (after warmup)
        """
        self.base_lambda = base_lambda
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs
    
    def get_lambda(self, epoch):
        """Get lambda for given epoch"""
        if epoch < self.warmup_epochs:
            # Warmup: 0 to 0.1 * base_lambda
            return (epoch / self.warmup_epochs) * self.base_lambda * 0.1
        elif epoch < self.warmup_epochs + self.ramp_epochs:
            # Ramp: 0.1 to 1.0 * base_lambda
            post_warmup = epoch - self.warmup_epochs
            progress = post_warmup / self.ramp_epochs
            return self.base_lambda * (0.1 + 0.9 * progress)
        else:
            # Plateau: base_lambda
            return self.base_lambda


class TemperatureSchedule:
    """
    Standalone temperature schedule for use in training scripts
    """
    
    def __init__(self, tau_init=5.0, tau_min=1.0, t_warmup=5, t_anneal=25):
        """
        Args:
            tau_init: Initial temperature
            tau_min: Minimum temperature
            t_warmup: Warmup end epoch
            t_anneal: Annealing end epoch
        """
        self.tau_init = tau_init
        self.tau_min = tau_min
        self.t_warmup = t_warmup
        self.t_anneal = t_anneal
        self.mu_min = 0.1
    
    def get_temperature(self, epoch):
        """Get temperature for given epoch"""
        if epoch < self.t_warmup:
            # Phase 1: Constant
            return self.tau_init
        elif epoch <= self.t_anneal:
            # Phase 2: Cosine annealing
            progress = (epoch - self.t_warmup) / (self.t_anneal - self.t_warmup)
            progress = min(1.0, max(0.0, progress))
            mu = self.mu_min + (1 - self.mu_min) * (np.cos(progress * np.pi) + 1) / 2
            return max(self.tau_min, self.tau_init * mu)
        else:
            # Phase 3: Plateau
            return self.tau_min
