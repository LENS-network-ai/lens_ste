"""
Quick Diagnostic Script for L0 Over-Pruning
Run this IMMEDIATELY to identify the problem

Usage:
    python debug_l0.py

This will test your current L0 implementation and show what's wrong.
"""

import torch
import numpy as np

def quick_diagnostic():
    """Quick check of L0 behavior"""
    
    print("="*70)
    print("🔍 L0 OVER-PRUNING DIAGNOSTIC")
    print("="*70)
    
    # Test parameters
    num_edges = 1000
    
    # Test 1: Check initialization impact
    print("\n1️⃣ TESTING INITIALIZATION IMPACT:")
    print("-"*70)
    
    test_inits = {
        "zeros": torch.zeros(num_edges),
        "ones": torch.ones(num_edges),
        "randn (mean=0)": torch.randn(num_edges),
        "randn (mean=-2)": torch.randn(num_edges) - 2,
        "randn (mean=-5)": torch.randn(num_edges) - 5,
    }
    
    try:
        from model.l0_hc_STE import get_expected_l0_sparsity, L0RegularizerParams
        params = L0RegularizerParams(gamma=-0.1, zeta=1.1, beta_l0=0.66)
        
        for name, logAlpha in test_inits.items():
            sparsity = get_expected_l0_sparsity(logAlpha, params)
            print(f"   {name:20s}: {sparsity:>6.1%} gates expected open")
            
            if sparsity < 0.1:
                print(f"      ⚠️⚠️⚠️ PROBLEM: < 10% gates open!")
            elif sparsity < 0.3:
                print(f"      ⚠️ Low: < 30% gates open")
            elif sparsity > 0.7:
                print(f"      ⚠️ High: > 70% gates open")
            else:
                print(f"      ✅ Good: 30-70% gates open")
    
    except ImportError:
        print("   ⚠️ Cannot import l0_hc_STE - using sigmoid approximation")
        for name, logAlpha in test_inits.items():
            # Approximate: p(open) ≈ sigmoid(logAlpha)
            prob_open = torch.sigmoid(logAlpha).mean().item()
            print(f"   {name:20s}: {prob_open:>6.1%} gates expected open (approx)")
    
    # Test 2: Check loss balance for different lambdas
    print("\n2️⃣ TESTING LAMBDA VALUES:")
    print("-"*70)
    
    # Typical values
    cls_loss = 1.0  # Typical classification loss
    num_edges = 1000
    expected_open = 0.5  # 50% gates open
    l0_penalty_raw = num_edges * expected_open  # Expected penalty
    
    test_lambdas = [0.00001, 0.00005, 0.0001, 0.0003, 0.001]
    
    print(f"   Assuming: cls_loss={cls_loss:.2f}, {num_edges} edges, {expected_open:.0%} open")
    print(f"   L0 penalty (raw) ≈ {l0_penalty_raw:.1f}")
    print()
    
    for lambda_val in test_lambdas:
        reg_loss = lambda_val * l0_penalty_raw
        ratio = reg_loss / cls_loss
        total = cls_loss + reg_loss
        
        status = ""
        if ratio > 5.0:
            status = "⚠️⚠️⚠️ CATASTROPHIC (reg >> cls)"
        elif ratio > 1.0:
            status = "⚠️⚠️ TOO HIGH (reg > cls)"
        elif ratio > 0.5:
            status = "⚠️ HIGH (reg = 50% of cls)"
        elif ratio > 0.1:
            status = "✅ GOOD (reg = 10-50% of cls)"
        else:
            status = "ℹ️ LOW (reg < 10% of cls)"
        
        print(f"   λ = {lambda_val:.6f}:")
        print(f"      reg_loss = {reg_loss:.4f}, ratio = {ratio:.2f} ({ratio*100:.0f}%) {status}")
    
    # Test 3: Check what happens with normalized lambda
    print("\n3️⃣ TESTING NORMALIZED LAMBDA (scaled by num_edges):")
    print("-"*70)
    
    base_lambda = 0.0001
    normalized_lambda = base_lambda / num_edges
    
    reg_loss_unnormalized = base_lambda * l0_penalty_raw
    reg_loss_normalized = normalized_lambda * l0_penalty_raw
    
    ratio_unnormalized = reg_loss_unnormalized / cls_loss
    ratio_normalized = reg_loss_normalized / cls_loss
    
    print(f"   Base lambda: {base_lambda:.6f}")
    print(f"   Normalized (/ {num_edges} edges): {normalized_lambda:.8f}")
    print()
    print(f"   WITHOUT normalization:")
    print(f"      reg_loss = {reg_loss_unnormalized:.4f}, ratio = {ratio_unnormalized:.2f}")
    print(f"      {'⚠️ TOO HIGH' if ratio_unnormalized > 0.5 else '✅ GOOD'}")
    print()
    print(f"   WITH normalization:")
    print(f"      reg_loss = {reg_loss_normalized:.4f}, ratio = {ratio_normalized:.2f}")
    print(f"      {'⚠️ TOO HIGH' if ratio_normalized > 0.5 else '✅ GOOD'}")
    
    # Test 4: Graph size impact
    print("\n4️⃣ TESTING GRAPH SIZE IMPACT:")
    print("-"*70)
    
    lambda_fixed = 0.0001
    
    for size in [100, 1000, 5000, 10000]:
        l0_penalty = size * 0.5  # 50% gates open
        reg_loss = lambda_fixed * l0_penalty
        ratio = reg_loss / cls_loss
        
        status = "✅" if ratio < 0.5 else "⚠️"
        print(f"   {size:5d} edges: reg_loss = {reg_loss:.4f}, ratio = {ratio:.2f} {status}")
    
    print()
    print(f"   Observation: Larger graphs → higher reg_loss with fixed lambda")
    print(f"   Solution: Scale lambda by 1/num_edges")
    
    # Summary and Recommendations
    print("\n" + "="*70)
    print("📋 DIAGNOSTIC SUMMARY")
    print("="*70)
    
    print("\n🔴 MOST LIKELY PROBLEMS:")
    print("   1. LogAlpha initialized too negative (mean < -1)")
    print("      → Gates start closed, hard to open")
    print("   2. Lambda not scaled by graph size")
    print("      → Reg loss >> cls loss for large graphs")
    print("   3. Classification loss very small")
    print("      → Even tiny lambda dominates")
    
    print("\n✅ RECOMMENDED FIXES:")
    print("   1. Initialize logAlpha around 0:")
    print("      self.logAlpha = nn.Parameter(torch.zeros(num_edges))")
    print()
    print("   2. Scale lambda by graph size:")
    print("      effective_lambda = base_lambda / num_edges")
    print("      reg_loss = effective_lambda * l0_penalty * num_edges")
    print()
    print("   3. Target reg/cls ratio of 0.1 to 0.3 (10-30%)")
    print()
    print("   4. Add warmup: no L0 for first 5 epochs")
    
    print("\n" + "="*70)
    print("🔍 NEXT STEP: Run training with diagnostics")
    print("="*70)
    print()
    print("Add this to your training loop:")
    print()
    print("   if batch_idx == 0:")
    print("       print(f'LogAlpha mean: {logAlpha.mean():.4f}')")
    print("       print(f'Cls loss: {cls_loss:.4f}')")
    print("       print(f'Reg loss: {reg_loss:.4f}')")
    print("       print(f'Ratio: {reg_loss/cls_loss:.2f}')")
    print()


if __name__ == "__main__":
    quick_diagnostic()
