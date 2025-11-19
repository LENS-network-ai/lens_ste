# model/L0Utils_STE.py
"""
Straight-Through Estimator (STE) for L0 Regularization
Binary gates during training with straight-through gradients

Simplified version - no stretching parameters needed!
"""

import torch
import torch.nn as nn


class STEBinarize(torch.autograd.Function):
    """
    Straight-Through Estimator for binary gates
    
    Forward: Hard threshold (binary output)
    Backward: Straight-through gradient (ignores threshold)
    """
    
    @staticmethod
    def forward(ctx, input):
        """
        Forward: Binary gate based on threshold
        
        Args:
            input: Probabilities [B, N, N]
        
        Returns:
            output: Binary gates {0, 1}
        """
        # Hard threshold at 0.5
        output = (input > 0.5).float()
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward: Pass gradient straight through (no modification)
        
        Args:
            grad_output: Gradient from next layer
        
        Returns:
            grad_input: Same gradient (straight-through)
        """
        # Straight-through: gradient flows as if threshold didn't exist
        return grad_output


def ste_sample_gates(logAlpha, temperature=1.0):
    """
    Sample binary gates using STE
    
    Args:
        logAlpha: Edge logits [B, N, N]
        temperature: Temperature for sigmoid (default: 1.0)
    
    Returns:
        gates: Binary gates {0, 1} [B, N, N]
        probs: Gate probabilities [B, N, N] (for L0 penalty)
    """
    # Compute probabilities
    probs = torch.sigmoid(logAlpha / temperature)
    
    # Binarize with STE (no stretching!)
    gates = STEBinarize.apply(probs)
    
    return gates, probs


def get_expected_l0_ste(logAlpha, temperature=1.0):
    """
    Compute expected L0 penalty for STE
    
    Args:
        logAlpha: Edge logits [B, N, N]
        temperature: Temperature (optional, default: 1.0)
    
    Returns:
        l0_penalty: Expected number of active gates (scalar)
    """
    # Probability of gate being active
    probs = torch.sigmoid(logAlpha / temperature)
    
    # Expected L0 = sum of probabilities (no stretching!)
    l0_penalty = probs.sum()
    
    return l0_penalty


class STERegularizerParams:
    """
    Parameters for STE regularization
    
    Simplified: No parameters needed for STE!
    (Kept for API consistency with Hard-Concrete and ARM)
    """
    def __init__(self):
        pass  # No parameters needed!
