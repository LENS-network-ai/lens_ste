# train.py (or your training file name)
"""
Training script with WandB logging and support for Hard-Concrete + ARM
FIXED VERSION with FORCED DIRECT CLEAR (bypasses broken clear_logits method)
"""

import os
import torch
import gc
import wandb
import numpy as np
from helper import Trainer, Evaluator, preparefeatureLabel


def train_and_evaluate(model, train_loader, val_loader, optimizer, scheduler, device,
                      num_epochs, n_features, output_dir, warmup_epochs=5,
                      wandb_config=None, l0_method='hard-concrete'):
    """
    Train and evaluate a model with WandB logging
    
    Args:
        model: LENS model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device (cuda/cpu)
        num_epochs: Number of training epochs
        n_features: Number of input features
        output_dir: Directory for outputs
        warmup_epochs: Number of warmup epochs
        wandb_config: WandB configuration dict (if None, don't use WandB)
        l0_method: 'hard-concrete' or 'arm'
    """
    # Initialize WandB if config provided
    use_wandb = wandb_config is not None
    if use_wandb:
        wandb.init(**wandb_config)
        wandb.watch(model, log='all', log_freq=100)
    
    # Initialize trainer and evaluator
    trainer = Trainer(n_class=model.num_classes)
    evaluator = Evaluator(n_class=model.num_classes)
    
    # Create output directories
    adj_output_dir = os.path.join(output_dir, 'pruned_adjacencies')
    analysis_dir = os.path.join(output_dir, 'graph_analysis')
    edge_dist_dir = os.path.join(output_dir, 'edge_distributions')
    report_dir = os.path.join(output_dir, 'sparsification_reports')
    
    for dir_path in [adj_output_dir, analysis_dir, edge_dist_dir, report_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Track metrics
    train_accs = []
    train_losses = []
    val_accs = []
    val_losses = []
    best_val_acc = 0.0
    best_epoch = 0
    best_edge_sparsity = 0.0
    
    # Main training loop
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"🚀 Epoch {epoch+1}/{num_epochs} ({l0_method})")
        print(f"{'='*60}")
        
        # Training phase
        train_metrics = train_epoch(
            epoch, model, train_loader, optimizer, scheduler,
            trainer, n_features, num_epochs, warmup_epochs,
            analysis_dir, edge_dist_dir, l0_method
        )
        
        train_accs.append(float(train_metrics['accuracy']))
        train_losses.append(float(train_metrics['loss']))
        
        # Log training metrics to WandB
        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'train/accuracy': train_metrics['accuracy'],
                'train/loss': train_metrics['loss'],
                'train/cls_loss': train_metrics.get('cls_loss', 0),
                'train/reg_loss': train_metrics.get('reg_loss', 0),
                'train/edge_density': train_metrics.get('edge_density', 0),
                'train/mean_edge_weight': train_metrics.get('mean_edge_weight', 0),
                'train/grad_norm': train_metrics.get('grad_norm', 0),
                'train/grad_variance': train_metrics.get('grad_variance', 0),
                'hyperparams/lambda': train_metrics.get('current_lambda', 0),
                'hyperparams/temperature': train_metrics.get('temperature', 0),
                'hyperparams/learning_rate': optimizer.param_groups[0]['lr'],
            })
            
            # ARM-specific metrics
            if l0_method == 'arm' and 'arm_loss' in train_metrics:
                wandb.log({
                    'train/arm_loss': train_metrics['arm_loss'],
                    'train/baseline': train_metrics.get('arm_baseline', 0),
                })
        
        torch.cuda.empty_cache()
        
        # Validation phase
        val_metrics = validate_epoch(
            model, val_loader, evaluator, n_features, epoch
        )
        
        val_accs.append(float(val_metrics['accuracy']))
        val_losses.append(float(val_metrics['loss']))
        
        print(f"   🔹 Validation Accuracy: {val_metrics['accuracy']:.4f}, "
              f"Loss: {val_metrics['loss']:.4f}")
        
        # Log validation metrics to WandB
        if use_wandb:
            wandb.log({
                'val/accuracy': val_metrics['accuracy'],
                'val/loss': val_metrics['loss'],
                'val/edge_density': val_metrics.get('edge_density', 0),
                'val/edge_sparsity': val_metrics.get('edge_sparsity', 0),
            })
        
        # Save sparsification report
        if hasattr(model, 'save_sparsification_report'):
            try:
                report_path = model.save_sparsification_report(
                    epoch + 1,
                    save_dir=report_dir
                )
                if report_path:
                    print(f"   📊 Sparsification report saved: {os.path.basename(report_path)}")
            except Exception as e:
                print(f"   ⚠️ Error generating sparsification report: {str(e)}")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_edge_sparsity = val_metrics.get('edge_sparsity', 0)
            best_epoch = epoch + 1
            
            # Save model
            model_path = os.path.join(output_dir, f'best_model_epoch{best_epoch}.pt')
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': best_val_acc,
                'edge_sparsity': best_edge_sparsity,
                'l0_method': l0_method,
            }, model_path)
            
            # Save to WandB
            if use_wandb:
                wandb.save(model_path)
                wandb.run.summary['best_val_accuracy'] = best_val_acc
                wandb.run.summary['best_epoch'] = best_epoch
                wandb.run.summary['best_edge_sparsity'] = best_edge_sparsity
            
            print(f"   ✅ New Best Model! Accuracy: {best_val_acc:.4f}, "
                  f"Sparsity: {best_edge_sparsity:.1f}% edges > 0.1")
    
    # Save pruned adjacencies
    save_pruned_adjacencies(trainer, adj_output_dir)
    
    # Final WandB summary
    if use_wandb:
        wandb.run.summary.update({
            'final_train_accuracy': train_accs[-1],
            'final_val_accuracy': val_accs[-1],
            'best_val_accuracy': best_val_acc,
            'best_epoch': best_epoch,
        })
        wandb.finish()
    
    results = {
        "train_accs": train_accs,
        "train_losses": train_losses,
        "val_accs": val_accs,
        "val_losses": val_losses,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "best_edge_sparsity": best_edge_sparsity
    }
    
    return results


def train_epoch(epoch, model, train_loader, optimizer, scheduler, trainer,
               n_features, num_epochs, warmup_epochs, analysis_dir, edge_dist_dir,
               l0_method='hard-concrete'):
    """
    Run one epoch of training with FORCED DIRECT CLEAR
    
    Returns:
        dict: Training metrics
    """
    # ============================================
    # 🔍 EPOCH SETUP - Set lambda/temp ONCE
    # ============================================
    model.set_epoch(epoch)
    
    # Store initial values to verify they don't change
    initial_lambda = model.regularizer.current_lambda if hasattr(model.regularizer, 'current_lambda') else 0
    initial_temp = model.temperature if hasattr(model, 'temperature') else 0
    
    # Debug: Print model configuration (once per epoch)
    print(f"\n{'─'*70}")
    print(f"🔍 EPOCH {epoch+1} CONFIGURATION")
    print(f"{'─'*70}")
    print(f"   use_l0: {model.use_l0 if hasattr(model, 'use_l0') else 'N/A'}")
    print(f"   reg_mode: {model.regularizer.reg_mode if hasattr(model.regularizer, 'reg_mode') else 'N/A'}")
    print(f"   l0_method: {l0_method}")
    print(f"   Initial λ: {initial_lambda:.6f}")
    print(f"   Initial temp: {initial_temp:.4f}")
    
    # Check regularizer capabilities
    has_logits_storage = hasattr(model, 'regularizer') and hasattr(model.regularizer, 'logits_storage')
    
    print(f"   Regularizer has logits_storage: {has_logits_storage}")
    
    # Print warmup status
    if epoch < warmup_epochs:
        progress = 100 * (epoch / warmup_epochs)
        print(f"   🔹 Warmup: {progress:.1f}% complete ({epoch+1}/{warmup_epochs} epochs)")
    else:
        print(f"   🔹 Warmup completed. Full sparsification active.")
    print(f"{'─'*70}\n")
    
    # Training mode
    model.train()
    train_loss = 0.0
    cls_loss_sum = 0.0
    reg_loss_sum = 0.0
    arm_loss_sum = 0.0
    trainer.reset_metrics()
    
    # For gradient and edge statistics
    grad_norms = []
    edge_densities = []
    edge_weights_list = []
    
    # For debugging per-batch changes
    batch_debug_info = []
    
    # ============================================
    # 🔍 BATCH LOOP WITH FORCED DIRECT CLEAR
    # ============================================
    for batch_idx, sample in enumerate(train_loader):
        optimizer.zero_grad()
        
        # ============================================
        # 🔍 DEBUG SECTION 1: BEFORE BATCH
        # ============================================
        if batch_idx % 50 == 0:
            print(f"\n{'┌'+'─'*68+'┐'}")
            print(f"│ 📍 BATCH {batch_idx:<4} - BEFORE TRAINING{' '*38}│")
            print(f"{'└'+'─'*68+'┘'}")
            
            # Check regularizer state BEFORE clear
            if hasattr(model, 'regularizer'):
                current_lambda = model.regularizer.current_lambda
                current_temp = model.temperature if hasattr(model, 'temperature') else 0
                
                print(f"   Lambda: {current_lambda:.6f} (initial: {initial_lambda:.6f})")
                print(f"   Temp: {current_temp:.4f} (initial: {initial_temp:.4f})")
                
                if abs(current_lambda - initial_lambda) > 1e-6:
                    print(f"   ⚠️⚠️⚠️ LAMBDA HAS CHANGED!")
                if abs(current_temp - initial_temp) > 1e-6:
                    print(f"   ⚠️⚠️⚠️ TEMPERATURE HAS CHANGED!")
                
                # Check logits_storage BEFORE clear
                if has_logits_storage:
                    num_entries_before = len(model.regularizer.logits_storage)
                    print(f"   logits_storage entries BEFORE clear: {num_entries_before}")
                    
                    if num_entries_before > 0:
                        print(f"   ⚠️ WARNING: logits_storage NOT EMPTY before clear!")
                        print(f"   Keys: {list(model.regularizer.logits_storage.keys())[:10]}")
        
        # ============================================
        # 🔧 CRITICAL FIX: FORCED DIRECT CLEAR
        # ============================================
        clear_success = False
        entries_after_clear = -1
        
        if has_logits_storage:
            try:
                # Get count before
                entries_before_clear = len(model.regularizer.logits_storage)
                
                # FORCE DIRECT CLEAR - bypasses any broken clear_logits() method
                model.regularizer.logits_storage = {}
                
                # Verify it actually cleared
                entries_after_clear = len(model.regularizer.logits_storage)
                
                if entries_after_clear == 0:
                    clear_success = True
                    if batch_idx % 50 == 0:
                        print(f"   ✅ FORCED DIRECT CLEAR: {entries_before_clear} → {entries_after_clear} entries")
                else:
                    if batch_idx % 50 == 0:
                        print(f"   ⚠️⚠️⚠️ CLEAR FAILED! Still has {entries_after_clear} entries after clear!")
                        print(f"   This should be IMPOSSIBLE - Python dict assignment failed?")
                
            except Exception as e:
                if batch_idx % 50 == 0:
                    print(f"   ❌ FORCED CLEAR EXCEPTION: {e}")
        else:
            if batch_idx % 50 == 0:
                print(f"   ⚠️ No logits_storage attribute found!")
        
        # Double-check: Try to clear any other possible storage locations
        if hasattr(model, 'regularizer'):
            # Clear any other potential storage variables
            if hasattr(model.regularizer, 'logits_list'):
                model.regularizer.logits_list = []
            if hasattr(model.regularizer, 'all_logits'):
                model.regularizer.all_logits = []
            if hasattr(model.regularizer, 'stored_logits'):
                model.regularizer.stored_logits = {}
        
        # Only print stats occasionally
        if hasattr(model, 'set_print_stats'):
            model.set_print_stats(batch_idx % 50 == 0)
        
        # ============================================
        # 🚀 FORWARD PASS
        # ============================================
        try:
            # Different training logic for Hard-Concrete vs ARM
            if l0_method == 'hard-concrete':
                # Standard Hard-Concrete training
                pred, labels, loss, weighted_adj = trainer.train(
                    sample, model, n_features=n_features
                )
                
                # Extract cls_loss and reg_loss if available
                if hasattr(model, 'stats_tracker') and hasattr(model.stats_tracker, 'cls_loss_history'):
                    if len(model.stats_tracker.cls_loss_history) > 0:
                        cls_loss = model.stats_tracker.cls_loss_history[-1]
                        reg_loss = model.stats_tracker.reg_loss_history[-1]
                        cls_loss_sum += cls_loss
                        reg_loss_sum += reg_loss
                
            elif l0_method == 'arm':
                # ARM training requires two forward passes
                
                # Prepare input
                node_feat, labels, adjs, masks = preparefeatureLabel(
                    sample['image'], sample['label'], sample['adj_s'], n_features=n_features
                )
                
                # First forward pass with sampled gates
                pred, labels_out, loss_b, weighted_adj, edge_weights_anti = model(
                    node_feat, labels, adjs, masks,
                    return_edge_weights_anti=True
                )
                
                # Extract classification loss
                cls_loss_b = loss_b
                
                # Get logAlpha for ARM gradient computation
                if hasattr(model.edge_scorer, 'last_logAlpha'):
                    logAlpha = model.edge_scorer.last_logAlpha
                else:
                    logAlpha = None
                
                # Second forward pass with antithetic gates (if available)
                if edge_weights_anti is not None and logAlpha is not None:
                    pred_anti, cls_loss_anti = model.forward_arm_antithetic(
                        node_feat, labels, adjs, edge_weights_anti, masks
                    )
                    
                    # Compute ARM gradient contribution
                    arm_loss = model.get_arm_gradient_loss(
                        cls_loss_b, cls_loss_anti, logAlpha
                    )
                    arm_loss_sum += arm_loss.item() if isinstance(arm_loss, torch.Tensor) else 0
                else:
                    cls_loss_anti = cls_loss_b
                    arm_loss = torch.tensor(0.0)
                
                # Total loss for ARM
                loss = loss_b + arm_loss
                
                # For tracking
                cls_loss_sum += cls_loss_b.item() if isinstance(cls_loss_b, torch.Tensor) else 0
                
                # Update trainer metrics manually for ARM
                trainer.update_metrics(pred, labels)
            
            else:
                raise ValueError(f"Unknown l0_method: {l0_method}")
            
            # ============================================
            # 🔍 DEBUG SECTION 2: AFTER FORWARD
            # ============================================
            if batch_idx % 50 == 0:
                print(f"\n{'┌'+'─'*68+'┐'}")
                print(f"│ 📊 BATCH {batch_idx:<4} - AFTER FORWARD{' '*40}│")
                print(f"{'└'+'─'*68+'┘'}")
                
                # Check logits_storage AFTER forward
                if has_logits_storage:
                    num_entries_after_forward = len(model.regularizer.logits_storage)
                    print(f"   logits_storage entries AFTER forward: {num_entries_after_forward}")
                    
                    # Diagnose what happened
                    if num_entries_after_forward == 0:
                        print(f"   ✅ Empty - PATH A used (direct L0 computation)")
                    elif num_entries_after_forward == 1:
                        print(f"   ✅ Single entry - PATH B used (storage), clear worked")
                    else:
                        print(f"   ⚠️⚠️⚠️ MULTIPLE ENTRIES ({num_entries_after_forward})!")
                        print(f"   THIS IS THE ACCUMULATION BUG!")
                        print(f"   Keys: {list(model.regularizer.logits_storage.keys())[:10]}")
                        
                        # This should be impossible if direct clear worked
                        if entries_after_clear == 0:
                            print(f"   🤔 MYSTERY: Clear worked (0 entries after clear)")
                            print(f"   But now has {num_entries_after_forward} entries after forward!")
                            print(f"   → Forward pass must be storing MULTIPLE entries!")
                
                # Check loss components
                if hasattr(model, 'stats_tracker'):
                    if hasattr(model.stats_tracker, 'cls_loss_history') and len(model.stats_tracker.cls_loss_history) > 0:
                        cls_loss = model.stats_tracker.cls_loss_history[-1]
                        reg_loss = model.stats_tracker.reg_loss_history[-1]
                        ratio = reg_loss / (cls_loss + 1e-8)
                        print(f"   cls_loss: {cls_loss:.4f}")
                        print(f"   reg_loss: {reg_loss:.4f}")
                        print(f"   ratio (reg/cls): {ratio:.4f}")
                        
                        if ratio > 5.0:
                            print(f"   ⚠️ Reg loss >> cls loss! Lambda may be too high")
                        elif ratio < 0.05:
                            print(f"   ⚠️ Reg loss << cls loss! Lambda may be too low")
                
                # Check edge density
                if weighted_adj is not None:
                    edge_density = (weighted_adj > 0.1).float().mean().item()
                    print(f"   edge_density (>0.1): {edge_density:.4f}")
            
            # Save visualizations periodically
            if batch_idx % 50 == 0:
                if hasattr(model, 'save_graph_analysis'):
                    model.save_graph_analysis(epoch + 1, batch_idx, save_dir=analysis_dir)
                
                if hasattr(model, 'plot_edge_weight_distribution'):
                    model.plot_edge_weight_distribution(
                        weighted_adj, epoch + 1, batch_idx, save_dir=edge_dist_dir
                    )
            
            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️ Warning: Loss is {loss.item()} in batch {batch_idx}, skipping")
                continue
            
            if batch_idx % 20 == 0 and batch_idx % 50 != 0:
                print(f"   Batch {batch_idx}: Loss = {loss.item():.4f}")
            
            # ============================================
            # 🔧 BACKWARD PASS
            # ============================================
            loss.backward()
            
            # Track gradient norm
            total_grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2).item()
                    total_grad_norm += param_norm ** 2
            total_grad_norm = total_grad_norm ** 0.5
            grad_norms.append(total_grad_norm)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # ============================================
            # 🔧 OPTIMIZER STEP
            # ============================================
            optimizer.step()
            
            # ============================================
            # 🔍 DEBUG SECTION 3: CHECK IF SCHEDULER CHANGES LAMBDA/TEMP
            # ============================================
            lambda_before_sched = model.regularizer.current_lambda
            temp_before_sched = model.temperature if hasattr(model, 'temperature') else 0
            
            scheduler(optimizer, batch_idx, epoch, 0)
            
            lambda_after_sched = model.regularizer.current_lambda
            temp_after_sched = model.temperature if hasattr(model, 'temperature') else 0
            
            if batch_idx % 50 == 0:
                if abs(lambda_after_sched - lambda_before_sched) > 1e-6:
                    print(f"   ⚠️⚠️⚠️ SCHEDULER CHANGED LAMBDA!")
                    print(f"   {lambda_before_sched:.6f} → {lambda_after_sched:.6f}")
                if abs(temp_after_sched - temp_before_sched) > 1e-6:
                    print(f"   ⚠️⚠️⚠️ SCHEDULER CHANGED TEMP!")
                    print(f"   {temp_before_sched:.4f} → {temp_after_sched:.4f}")
            
            # Track loss
            train_loss += loss.item()
            
            # Track edge statistics
            if weighted_adj is not None:
                edge_mask = weighted_adj > 0
                if edge_mask.sum() > 0:
                    active_weights = weighted_adj[edge_mask]
                    edge_density = (active_weights > 0.1).float().mean().item()
                    edge_densities.append(edge_density)
                    edge_weights_list.append(active_weights.detach().cpu())
            
            # Store batch debug info
            if batch_idx % 50 == 0:
                batch_debug_info.append({
                    'batch': batch_idx,
                    'edge_density': edge_density if weighted_adj is not None else 0,
                    'logits_entries_before': num_entries_before if has_logits_storage else -1,
                    'logits_entries_after_clear': entries_after_clear,
                    'logits_entries_after_forward': num_entries_after_forward if has_logits_storage else -1,
                    'lambda': lambda_after_sched,
                    'temp': temp_after_sched,
                })
            
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"⚠️ CUDA OOM in batch {batch_idx}. Skipping and clearing cache.")
                torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                raise e
    
    # ============================================
    # 🔍 EPOCH SUMMARY WITH DETAILED ANALYSIS
    # ============================================
    print(f"\n{'='*70}")
    print(f"📊 EPOCH {epoch+1} SUMMARY & ANALYSIS")
    print(f"{'='*70}")
    
    # Check if lambda/temp stayed constant
    final_lambda = model.regularizer.current_lambda
    final_temp = model.temperature if hasattr(model, 'temperature') else 0
    
    print(f"\n🔍 Hyperparameter Stability:")
    print(f"   Lambda: {initial_lambda:.6f} → {final_lambda:.6f}")
    print(f"   Temp: {initial_temp:.4f} → {final_temp:.4f}")
    
    if abs(final_lambda - initial_lambda) > 1e-6:
        print(f"   ⚠️⚠️⚠️ LAMBDA CHANGED DURING EPOCH! This causes per-batch pruning!")
    else:
        print(f"   ✅ Lambda stayed constant")
    
    if abs(final_temp - initial_temp) > 1e-6:
        print(f"   ⚠️⚠️⚠️ TEMPERATURE CHANGED DURING EPOCH!")
    else:
        print(f"   ✅ Temperature stayed constant")
    
    # Logits storage analysis
    if batch_debug_info:
        print(f"\n🔍 Logits Storage Analysis (sampled batches):")
        for info in batch_debug_info:
            print(f"   Batch {info['batch']:4d}: before={info['logits_entries_before']:3d}, "
                  f"after_clear={info['logits_entries_after_clear']:3d}, "
                  f"after_forward={info['logits_entries_after_forward']:3d}")
        
        # Check for accumulation pattern
        after_forward_counts = [info['logits_entries_after_forward'] for info in batch_debug_info if info['logits_entries_after_forward'] >= 0]
        
        if after_forward_counts:
            max_entries = max(after_forward_counts)
            
            if max_entries > 1:
                print(f"\n   ⚠️⚠️⚠️ ACCUMULATION DETECTED!")
                print(f"   Max entries after forward: {max_entries}")
                print(f"   → Forward pass stores MULTIPLE logits per call!")
                print(f"   → Bug is in EdgeScoring or model.forward(), not in clear!")
            elif max_entries == 1:
                print(f"\n   ✅ Normal: Single entry per batch (PATH B used)")
            else:
                print(f"\n   ✅ Empty: No storage used (PATH A - direct computation)")
    
    # Edge density progression analysis
    if edge_densities:
        print(f"\n📈 Edge Density Progression Within Epoch:")
        
        # Sample points throughout the epoch
        num_batches = len(edge_densities)
        sample_indices = [0, num_batches//4, num_batches//2, 3*num_batches//4, num_batches-1]
        
        for i in sample_indices:
            if i < len(edge_densities):
                print(f"   Batch {i:4d}: {edge_densities[i]:.4f}")
        
        # Calculate drop
        total_drop = edge_densities[0] - edge_densities[-1]
        percent_drop = (total_drop / edge_densities[0]) * 100 if edge_densities[0] > 0 else 0
        
        print(f"\n   Total drop: {total_drop:.4f} ({percent_drop:.1f}%)")
        
        # Diagnosis
        if total_drop > 0.10:
            print(f"   ⚠️⚠️⚠️ CATASTROPHIC DROP! (>10%)")
            print(f"   Likely causes:")
            print(f"     1. Logits accumulating (check analysis above)")
            print(f"     2. Lambda/temp changing per batch")
            print(f"     3. Lambda too high relative to classification loss")
        elif total_drop > 0.05:
            print(f"   ⚠️ LARGE DROP within epoch (>5%)")
            print(f"   This should NOT happen with proper clearing!")
        elif total_drop > 0.02:
            print(f"   ⚠️ Moderate drop (2-5%)")
            print(f"   Some change is expected, but monitor closely")
        else:
            print(f"   ✅ Small drop (<2%) - EXPECTED BEHAVIOR")
    
    # Compute final metrics
    train_acc = trainer.get_scores()
    avg_train_loss = train_loss / max(1, len(train_loader))
    avg_cls_loss = cls_loss_sum / max(1, len(train_loader))
    avg_reg_loss = reg_loss_sum / max(1, len(train_loader))
    
    # Gradient statistics
    mean_grad_norm = np.mean(grad_norms) if grad_norms else 0
    grad_variance = np.var(grad_norms) if len(grad_norms) > 1 else 0
    
    # Edge statistics
    mean_edge_density = np.mean(edge_densities) if edge_densities else 0
    if edge_weights_list:
        all_weights = torch.cat(edge_weights_list)
        mean_edge_weight = all_weights.mean().item()
    else:
        mean_edge_weight = 0
    
    print(f"\n📊 Final Epoch Metrics:")
    print(f"   Training Accuracy: {train_acc:.4f}")
    print(f"   Total Loss: {avg_train_loss:.4f}")
    print(f"   Cls Loss: {avg_cls_loss:.4f}")
    print(f"   Reg Loss: {avg_reg_loss:.4f}")
    print(f"   Reg/Cls Ratio: {(avg_reg_loss/(avg_cls_loss+1e-8)):.4f}")
    print(f"   Edge Density: {mean_edge_density:.4f}")
    print(f"   Mean Edge Weight: {mean_edge_weight:.4f}")
    print(f"   Grad Norm: {mean_grad_norm:.4f}")
    print(f"{'='*70}")
    
    # Compile metrics
    metrics = {
        'accuracy': train_acc,
        'loss': avg_train_loss,
        'cls_loss': avg_cls_loss,
        'reg_loss': avg_reg_loss,
        'edge_density': mean_edge_density,
        'mean_edge_weight': mean_edge_weight,
        'grad_norm': mean_grad_norm,
        'grad_variance': grad_variance,
        'current_lambda': final_lambda,
        'temperature': final_temp,
    }
    
    # ARM-specific metrics
    if l0_method == 'arm':
        metrics['arm_loss'] = arm_loss_sum / max(1, len(train_loader))
        if hasattr(model.l0_params, 'baseline'):
            metrics['arm_baseline'] = model.l0_params.baseline.item() if model.l0_params.baseline is not None else 0
    
    return metrics


def validate_epoch(model, val_loader, evaluator, n_features, epoch):
    """
    Run one epoch of validation
    
    Returns:
        dict: Validation metrics
    """
    model.eval()
    val_loss = 0.0
    evaluator.reset_metrics()
    
    # Disable stats printing
    if hasattr(model, 'set_print_stats'):
        model.set_print_stats(False)
    
    # Track edge statistics
    all_edge_weights = []
    
    with torch.no_grad():
        for sample in val_loader:
            try:
                # Standard evaluation
                pred, labels, loss, weighted_adj = evaluator.eval_test(
                    sample, model, n_features=n_features
                )
                val_loss += loss.item()
                
                # Get adjacency for sparsity calculation
                node_feat, labels, adjs, masks = preparefeatureLabel(
                    sample['image'], sample['label'], sample['adj_s'], n_features=n_features
                )
                
                # Calculate edge weights (normalized by original adjacency)
                edge_mask = (adjs > 0).float()
                masked_weights = torch.zeros_like(weighted_adj)
                for i in range(adjs.shape[0]):
                    mask = adjs[i] > 0
                    if mask.sum() > 0:
                        masked_weights[i, mask] = weighted_adj[i, mask] / adjs[i, mask]
                
                all_edge_weights.append(masked_weights.detach().cpu())
                
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"⚠️ CUDA OOM in validation. Skipping and clearing cache.")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise e
    
    # Compute metrics
    val_acc = evaluator.get_scores()
    avg_val_loss = val_loss / max(1, len(val_loader))
    
    # Calculate edge sparsity
    edge_sparsity, edge_density = calculate_edge_sparsity(all_edge_weights)
    
    metrics = {
        'accuracy': val_acc,
        'loss': avg_val_loss,
        'edge_sparsity': edge_sparsity,
        'edge_density': edge_density,
    }
    
    return metrics


def calculate_edge_sparsity(all_edge_weights):
    """
    Calculate edge sparsity metrics from edge weights
    
    Returns:
        edge_sparsity: Percentage of edges > 0.1
        edge_density: Percentage of edges > 0.5
    """
    if all_edge_weights:
        # Concatenate all edge weights
        all_weights = torch.cat([w.flatten() for w in all_edge_weights])
        all_weights = all_weights[all_weights > 0]  # Only consider positive weights
        
        if len(all_weights) > 0:
            # Calculate metrics
            avg_weight = all_weights.mean().item()
            median_weight = all_weights.median().item()
            
            # Count weights above thresholds
            thresholds = [0.01, 0.05, 0.1, 0.2, 0.5]
            sparsity_metrics = {}
            for t in thresholds:
                sparsity_metrics[f">{t}"] = (all_weights > t).float().mean().item() * 100.0
            
            # Main metrics
            edge_sparsity = sparsity_metrics[">0.1"]
            edge_density = sparsity_metrics[">0.5"]
            
            # Print
            print(f"   📊 Edge weight stats - Mean: {avg_weight:.6f}, Median: {median_weight:.6f}")
            print(f"   📊 Edge sparsity - {sparsity_metrics['>0.01']:.1f}% > 0.01, "
                  f"{sparsity_metrics['>0.1']:.1f}% > 0.1, "
                  f"{sparsity_metrics['>0.5']:.1f}% > 0.5")
            
            return edge_sparsity, edge_density
    
    return 0.0, 0.0


def save_pruned_adjacencies(trainer, adj_output_dir):
    """Save pruned adjacency matrices"""
    if hasattr(trainer, 'saved_pruned_adjs') and trainer.saved_pruned_adjs:
        print(f"\n💾 Saving {len(trainer.saved_pruned_adjs)} pruned adjacency matrices...")
        
        os.makedirs(adj_output_dir, exist_ok=True)
        
        for wsi_id, pruned_adj in trainer.saved_pruned_adjs.items():
            # Clean filename
            clean_id = ''.join(c for c in wsi_id if c.isalnum() or c in '._- ')
            
            # Save
            adj_path = os.path.join(adj_output_dir, f"{clean_id}_pruned_adj.pt")
            torch.save({
                'pruned_adj': pruned_adj,
                'original_edges': trainer.original_edges.get(wsi_id, 0),
                'wsi_id': wsi_id
            }, adj_path)
        
        print(f"✅ Pruned adjacencies saved to {adj_output_dir}")
