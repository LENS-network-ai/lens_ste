# training/training_loop.py
"""
Training loop with WandB logging and TARGET DENSITY CONTROL
Supports Hard-Concrete + ARM with adaptive lambda and density loss
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
    Train and evaluate a model with WandB logging and density control
    
    Args:
        model: LENS model with density control
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
    
    # Track density metrics across epochs
    density_history = []
    alpha_history = []
    
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
        density_history.append(train_metrics.get('current_density', 0))
        alpha_history.append(train_metrics.get('avg_alpha', 1.0))
        
        # Log training metrics to WandB
        if use_wandb:
            log_dict = {
                'epoch': epoch,
                # Performance metrics
                'train/accuracy': train_metrics['accuracy'],
                'train/loss': train_metrics['loss'],
                'train/cls_loss': train_metrics.get('cls_loss', 0),
                'train/reg_loss': train_metrics.get('reg_loss', 0),
                'train/edge_density': train_metrics.get('edge_density', 0),
                'train/mean_edge_weight': train_metrics.get('mean_edge_weight', 0),
                'train/grad_norm': train_metrics.get('grad_norm', 0),
                
                # 🆕 DENSITY CONTROL METRICS
                'density/current': train_metrics.get('current_density', 0),
                'density/deviation': train_metrics.get('density_deviation', 0),
                'density/alpha': train_metrics.get('avg_alpha', 1.0),
                
                # 🆕 LAMBDA METRICS
                'hyperparams/lambda_base': train_metrics.get('current_lambda', 0),
                'hyperparams/lambda_eff': train_metrics.get('lambda_eff', 0),
                'hyperparams/lambda_density': train_metrics.get('current_lambda_density', 0),
                'hyperparams/temperature': train_metrics.get('temperature', 0),
                'hyperparams/learning_rate': optimizer.param_groups[0]['lr'],
            }
            
            # ARM-specific metrics
            if l0_method == 'arm' and 'arm_loss' in train_metrics:
                log_dict.update({
                    'train/arm_loss': train_metrics['arm_loss'],
                    'train/baseline': train_metrics.get('arm_baseline', 0),
                })
            
            # Target density tracking
            if hasattr(model.regularizer, 'target_density'):
                log_dict['density/target'] = model.regularizer.target_density
            
            wandb.log(log_dict)
        
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
                'current_density': train_metrics.get('current_density', 0),
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
    
    # 🆕 Final density control summary
    print_density_control_summary(density_history, alpha_history, model.regularizer)
    
    # Final WandB summary
    if use_wandb:
        wandb.run.summary.update({
            'final_train_accuracy': train_accs[-1],
            'final_val_accuracy': val_accs[-1],
            'best_val_accuracy': best_val_acc,
            'best_epoch': best_epoch,
            'final_density': density_history[-1] if density_history else 0,
            'avg_density': np.mean(density_history) if density_history else 0,
            'avg_alpha': np.mean(alpha_history) if alpha_history else 1.0,
        })
        wandb.finish()
    
    results = {
        "train_accs": train_accs,
        "train_losses": train_losses,
        "val_accs": val_accs,
        "val_losses": val_losses,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "best_edge_sparsity": best_edge_sparsity,
        "density_history": density_history,
        "alpha_history": alpha_history,
    }
    
    return results


def train_epoch(epoch, model, train_loader, optimizer, scheduler, trainer,
               n_features, num_epochs, warmup_epochs, analysis_dir, edge_dist_dir,
               l0_method='hard-concrete'):
    """
    Run one epoch of training WITH DENSITY CONTROL
    
    Returns:
        dict: Training metrics including density control metrics
    """
    # ============================================
    # 🔍 EPOCH SETUP
    # ============================================
    model.set_epoch(epoch)
    
    # 🆕 UPDATE ALL SCHEDULES (INCLUDING DENSITY LAMBDA!)
    initial_temp = model.temperature if hasattr(model, 'temperature') else 5.0
    
    if hasattr(model.regularizer, 'update_all_schedules'):
        schedules = model.regularizer.update_all_schedules(
            current_epoch=epoch,
            initial_temp=initial_temp
        )
        
        print(f"\n{'─'*70}")
        print(f"🔍 EPOCH {epoch+1} CONFIGURATION")
        print(f"{'─'*70}")
        print(f"   L0 λ: {schedules['lambda']:.6f}")
        print(f"   Density λ: {schedules['lambda_density']:.6f}")
        print(f"   Temperature: {schedules['temperature']:.4f}")
        
        if hasattr(model.regularizer, 'target_density'):
            print(f"   Target Density: {model.regularizer.target_density*100:.1f}%")
            print(f"   Adaptive Lambda: {model.regularizer.enable_adaptive_lambda}")
        
        # Warmup progress
        if epoch < warmup_epochs:
            progress = 100 * (epoch / warmup_epochs)
            print(f"   🔹 Warmup: {progress:.1f}% complete ({epoch+1}/{warmup_epochs} epochs)")
        else:
            print(f"   🔹 Warmup completed. Full density control active.")
        print(f"{'─'*70}\n")
    
    # Training mode
    model.train()
    train_loss = 0.0
    cls_loss_sum = 0.0
    reg_loss_sum = 0.0
    density_loss_sum = 0.0  # 🆕
    arm_loss_sum = 0.0
    trainer.reset_metrics()
    
    # For statistics
    grad_norms = []
    batch_densities = []  # 🆕 Track density per batch
    edge_weights_list = []
    alpha_values = []  # 🆕 Track adaptive scaling per batch
    lambda_eff_values = []  # 🆕 Track effective lambda
    
    # ============================================
    # 🔍 BATCH LOOP
    # ============================================
    for batch_idx, sample in enumerate(train_loader):
        optimizer.zero_grad()
        
        # Clear logits storage (forced direct clear)
        if hasattr(model, 'regularizer') and hasattr(model.regularizer, 'logits_storage'):
            model.regularizer.logits_storage = {}
        
        # Print stats occasionally
        if hasattr(model, 'set_print_stats'):
            model.set_print_stats(batch_idx % 50 == 0)
        
        # ============================================
        # 🚀 FORWARD PASS
        # ============================================
        try:
            if l0_method == 'hard-concrete':
                # Standard Hard-Concrete training
                pred, labels, loss, weighted_adj = trainer.train(
                    sample, model, n_features=n_features
                )
                
                # Extract loss components
                if hasattr(model, 'stats_tracker') and hasattr(model.stats_tracker, 'cls_loss_history'):
                    if len(model.stats_tracker.cls_loss_history) > 0:
                        cls_loss = model.stats_tracker.cls_loss_history[-1]
                        reg_loss = model.stats_tracker.reg_loss_history[-1]
                        cls_loss_sum += cls_loss
                        reg_loss_sum += reg_loss
                        
                        # 🆕 Extract density loss if available
                        if hasattr(model.stats_tracker, 'density_loss_history'):
                            if len(model.stats_tracker.density_loss_history) > 0:
                                density_loss = model.stats_tracker.density_loss_history[-1]
                                density_loss_sum += density_loss
                
            elif l0_method == 'arm':
                # ARM training
                node_feat, labels, adjs, masks = preparefeatureLabel(
                    sample['image'], sample['label'], sample['adj_s'], n_features=n_features
                )
                
                pred, labels_out, loss_b, weighted_adj, edge_weights_anti = model(
                    node_feat, labels, adjs, masks,
                    return_edge_weights_anti=True
                )
                
                cls_loss_b = loss_b
                
                if hasattr(model.edge_scorer, 'last_logAlpha'):
                    logAlpha = model.edge_scorer.last_logAlpha
                else:
                    logAlpha = None
                
                if edge_weights_anti is not None and logAlpha is not None:
                    pred_anti, cls_loss_anti = model.forward_arm_antithetic(
                        node_feat, labels, adjs, edge_weights_anti, masks
                    )
                    arm_loss = model.get_arm_gradient_loss(
                        cls_loss_b, cls_loss_anti, logAlpha
                    )
                    arm_loss_sum += arm_loss.item() if isinstance(arm_loss, torch.Tensor) else 0
                else:
                    arm_loss = torch.tensor(0.0)
                
                loss = loss_b + arm_loss
                cls_loss_sum += cls_loss_b.item() if isinstance(cls_loss_b, torch.Tensor) else 0
                trainer.update_metrics(pred, labels)
            
            else:
                raise ValueError(f"Unknown l0_method: {l0_method}")
            
            # ============================================
            # 🆕 COMPUTE CURRENT DENSITY & ADAPTIVE METRICS
            # ============================================
            if weighted_adj is not None:
                # Get original adjacency
                node_feat_temp, labels_temp, adjs, masks_temp = preparefeatureLabel(
                    sample['image'], sample['label'], sample['adj_s'], n_features=n_features
                )
                
                # Compute density
                from model.EGL_L0_Reg import compute_density
                current_density = compute_density(weighted_adj, adjs)
                batch_densities.append(current_density.item())
                
                # Get adaptive scaling
                if hasattr(model.regularizer, 'compute_adaptive_lambda'):
                    lambda_eff, alpha = model.regularizer.compute_adaptive_lambda(current_density)
                    alpha_values.append(alpha)
                    lambda_eff_values.append(lambda_eff if isinstance(lambda_eff, float) else lambda_eff.item())
                
                # Print batch stats periodically
                if batch_idx % 50 == 0:
                    print(f"   Batch {batch_idx}: "
                          f"Loss = {loss.item():.4f}, "
                          f"Density = {current_density.item()*100:.2f}%", end="")
                    
                    if alpha_values:
                        print(f", λ_eff = {lambda_eff_values[-1]:.6f}", end="")
                    
                    print()
            
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
            
            # Optimizer step
            optimizer.step()
            scheduler(optimizer, batch_idx, epoch, 0)
            
            # Track loss
            train_loss += loss.item()
            
            # Track edge statistics
            if weighted_adj is not None:
                edge_mask = weighted_adj > 0
                if edge_mask.sum() > 0:
                    active_weights = weighted_adj[edge_mask]
                    edge_weights_list.append(active_weights.detach().cpu())
            
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"⚠️ CUDA OOM in batch {batch_idx}. Skipping and clearing cache.")
                torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                raise e
    
    # ============================================
    # 🆕 EPOCH SUMMARY WITH DENSITY CONTROL METRICS
    # ============================================
    print(f"\n{'='*70}")
    print(f"📊 EPOCH {epoch+1} SUMMARY")
    print(f"{'='*70}")
    
    # Compute final metrics
    train_acc = trainer.get_scores()
    avg_train_loss = train_loss / max(1, len(train_loader))
    avg_cls_loss = cls_loss_sum / max(1, len(train_loader))
    avg_reg_loss = reg_loss_sum / max(1, len(train_loader))
    avg_density_loss = density_loss_sum / max(1, len(train_loader))
    
    # 🆕 Density metrics
    avg_density = np.mean(batch_densities) if batch_densities else 0
    avg_alpha = np.mean(alpha_values) if alpha_values else 1.0
    avg_lambda_eff = np.mean(lambda_eff_values) if lambda_eff_values else 0
    
    # Gradient statistics
    mean_grad_norm = np.mean(grad_norms) if grad_norms else 0
    
    # Edge statistics
    if edge_weights_list:
        all_weights = torch.cat(edge_weights_list)
        mean_edge_weight = all_weights.mean().item()
        edge_sparsity = (all_weights > 0.1).float().mean().item()
    else:
        mean_edge_weight = 0
        edge_sparsity = 0
    
    # Print performance metrics
    print(f"\n📈 Performance Metrics:")
    print(f"   Training Accuracy: {train_acc:.4f}")
    print(f"   Total Loss: {avg_train_loss:.4f}")
    print(f"   - Cls Loss: {avg_cls_loss:.4f}")
    print(f"   - Reg Loss: {avg_reg_loss:.4f}")
    if avg_density_loss > 0:
        print(f"   - Density Loss: {avg_density_loss:.4f}")
    print(f"   Reg/Cls Ratio: {(avg_reg_loss/(avg_cls_loss+1e-8)):.4f}")
    
    # 🆕 Print density control metrics
    print(f"\n🎯 Density Control:")
    print(f"   Current Density: {avg_density*100:.2f}%")
    
    if hasattr(model.regularizer, 'target_density'):
        target = model.regularizer.target_density
        deviation = abs(avg_density - target)
        print(f"   Target Density: {target*100:.2f}%")
        print(f"   Deviation: {deviation*100:.1f}%")
        
        # Adaptive direction indicator
        if avg_alpha > 1.05:
            direction = "increased"
            change = (avg_alpha - 1) * 100
            arrow = "↑"
        elif avg_alpha < 0.95:
            direction = "decreased"
            change = (1 - avg_alpha) * 100
            arrow = "↓"
        else:
            direction = "stable"
            change = 0
            arrow = "→"
        
        if direction != "stable":
            print(f"   → Adaptive: {direction} by {change:.1f}% {arrow}")
        else:
            print(f"   → Adaptive: {direction} {arrow}")
        
        # Status indicator
        if deviation < 0.02:
            print(f"   ✅ Excellent density control!")
        elif deviation < 0.05:
            print(f"   ✓ Good density control")
        elif deviation < 0.10:
            print(f"   ⚠️ Consider tuning lambda_density")
        else:
            print(f"   ⚠️⚠️ Large deviation! Check hyperparameters")
    
    # 🆕 Print regularization info
    print(f"\n🔧 Regularization:")
    if hasattr(model.regularizer, 'current_lambda'):
        print(f"   Base L0 λ: {model.regularizer.current_lambda:.6f}")
        if lambda_eff_values:
            print(f"   Effective λ: {avg_lambda_eff:.6f}")
            print(f"   Avg α: {avg_alpha:.3f}")
    
    if hasattr(model.regularizer, 'current_lambda_density'):
        print(f"   Density λ: {model.regularizer.current_lambda_density:.6f}")
    
    # Print edge statistics
    print(f"\n📊 Edge Statistics:")
    print(f"   Sparsity (>0.1): {edge_sparsity:.4f}")
    print(f"   Mean Edge Weight: {mean_edge_weight:.4f}")
    print(f"   Gradient Norm: {mean_grad_norm:.4f}")
    print(f"{'='*70}\n")
    
    # Compile metrics
    metrics = {
        # Performance
        'accuracy': train_acc,
        'loss': avg_train_loss,
        'cls_loss': avg_cls_loss,
        'reg_loss': avg_reg_loss,
        'density_loss': avg_density_loss,
        
        # 🆕 Density control metrics
        'current_density': avg_density,
        'density_deviation': abs(avg_density - model.regularizer.target_density) if hasattr(model.regularizer, 'target_density') else 0,
        'avg_alpha': avg_alpha,
        'lambda_eff': avg_lambda_eff,
        
        # Edge metrics
        'edge_density': edge_sparsity,
        'mean_edge_weight': mean_edge_weight,
        
        # Training metrics
        'grad_norm': mean_grad_norm,
        
        # Hyperparameters
        'current_lambda': model.regularizer.current_lambda if hasattr(model.regularizer, 'current_lambda') else 0,
        'current_lambda_density': model.regularizer.current_lambda_density if hasattr(model.regularizer, 'current_lambda_density') else 0,
        'temperature': model.temperature if hasattr(model, 'temperature') else 0,
    }
    
    # ARM-specific
    if l0_method == 'arm':
        metrics['arm_loss'] = arm_loss_sum / max(1, len(train_loader))
        if hasattr(model, 'l0_params') and hasattr(model.l0_params, 'baseline'):
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


def print_density_control_summary(density_history, alpha_history, regularizer):
    """
    Print comprehensive density control summary across all epochs
    
    Args:
        density_history: List of density values per epoch
        alpha_history: List of alpha values per epoch
        regularizer: EGLassoRegularization instance
    """
    if not density_history:
        return
    
    print(f"\n{'='*70}")
    print(f"🎯 DENSITY CONTROL SUMMARY")
    print(f"{'='*70}")
    
    # Calculate statistics
    avg_density = np.mean(density_history)
    final_density = density_history[-1]
    min_density = np.min(density_history)
    max_density = np.max(density_history)
    
    if hasattr(regularizer, 'target_density'):
        target = regularizer.target_density
        final_deviation = abs(final_density - target)
        avg_deviation = np.mean([abs(d - target) for d in density_history])
        
        print(f"\n📊 Density Statistics:")
        print(f"   Target: {target*100:.2f}%")
        print(f"   Final: {final_density*100:.2f}%")
        print(f"   Average: {avg_density*100:.2f}%")
        print(f"   Range: [{min_density*100:.2f}%, {max_density*100:.2f}%]")
        print(f"   Final Deviation: {final_deviation*100:.2f}%")
        print(f"   Avg Deviation: {avg_deviation*100:.2f}%")
    
    if alpha_history:
        avg_alpha = np.mean(alpha_history)
        print(f"\n🔧 Adaptive Lambda Statistics:")
        print(f"   Average α: {avg_alpha:.3f}")
        print(f"   Final α: {alpha_history[-1]:.3f}")
        print(f"   Range: [{np.min(alpha_history):.3f}, {np.max(alpha_history):.3f}]")
    
    # Assessment
    print(f"\n✅ Assessment:")
    if hasattr(regularizer, 'target_density'):
        if final_deviation < 0.02:
            print(f"   🎉 Excellent! Density within 2% of target")
        elif final_deviation < 0.05:
            print(f"   ✓ Good density control achieved")
        elif final_deviation < 0.10:
            print(f"   ⚠️ Moderate control - consider tuning lambda_density")
        else:
            print(f"   ⚠️⚠️ Poor control - review hyperparameters")
            
            # Provide specific guidance
            if final_density < target:
                print(f"   → Density too low: Decrease base lambda or increase lambda_density")
            else:
                print(f"   → Density too high: Increase base lambda or decrease lambda_density")
    
    print(f"{'='*70}\n")


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
