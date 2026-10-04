#!/usr/bin/env python3
"""
Plot training metrics from SALMONN training logs
Usage: python plot_training_metrics.py <log_file_or_directory>
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def parse_log_file(log_path):
    """Parse training log file and extract metrics"""
    train_losses = []
    train_lrs = []
    train_iterations = []  # Global iteration numbers
    valid_losses = []
    valid_metrics = []
    valid_reason_accs = [] # New: Store reasoning accuracy
    valid_reason_per_class = [] # New: Store per-class reasoning accuracy
    epochs = []
    
    # Track iters_per_epoch to calculate global iterations
    iters_per_epoch = None
    
    with open(log_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                
                # Training metrics
                if 'train_loss' in data:
                    train_losses.append(float(data['train_loss']))
                    if 'train_lr' in data:
                        train_lrs.append(float(data['train_lr']))
                    
                    train_iterations.append(50 * len(train_losses) - 1)
                
                # Validation metrics
                if 'valid_loss' in data:
                    valid_losses.append(float(data['valid_loss']))
                    if 'valid_agg_metrics' in data:
                        valid_metrics.append(float(data['valid_agg_metrics']))
                    if 'valid_reason_acc' in data: # New: Parse reasoning accuracy
                        valid_reason_accs.append(float(data['valid_reason_acc']))
                    if 'valid_reason_per_class' in data: # New: Parse per-class reasoning accuracy
                        valid_reason_per_class.append(data['valid_reason_per_class'])
                    if 'valid_best_epoch' in data:
                        epochs.append(int(data['valid_best_epoch']))
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                # Skip malformed lines
                continue
    
    return {
        'train_losses': train_losses,
        'train_lrs': train_lrs,
        'train_iterations': train_iterations,
        'valid_losses': valid_losses,
        'valid_metrics': valid_metrics,
        'valid_reason_accs': valid_reason_accs, # New
        'valid_reason_per_class': valid_reason_per_class, # New
        'epochs': epochs
    }


def plot_metrics(metrics, output_dir):
    """Create plots for training metrics"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Training Loss
    if metrics['train_losses']:
        x_vals = metrics['train_iterations'] if metrics['train_iterations'] else range(len(metrics['train_losses']))
        
        # Calculate smoothed line if enough data
        smoothed = None
        window = None
        if len(metrics['train_losses']) > 20:
            window = min(20, len(metrics['train_losses']) // 10)
            smoothed = np.convolve(metrics['train_losses'], 
                                   np.ones(window)/window, mode='valid')
        
        # Plot 1a: Training Loss (Linear Scale)
        plt.figure(figsize=(12, 6))
        plt.plot(x_vals, metrics['train_losses'], linewidth=1, alpha=0.4, label='Raw', color='blue')
        if smoothed is not None:
            plt.plot(x_vals[window-1:], smoothed, linewidth=2.5, label=f'Smoothed (window={window})', color='blue')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Training Loss', fontsize=12)
        plt.title('Training Loss Over Time', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'training_loss.png', dpi=150)
        print(f"✓ Saved: {output_dir / 'training_loss.png'}")
        plt.close()
        
        # Plot 1b: Training Loss (Log Scale)
        plt.figure(figsize=(12, 6))
        plt.plot(x_vals, metrics['train_losses'], linewidth=1, alpha=0.4, label='Raw', color='blue')
        if smoothed is not None:
            plt.plot(x_vals[window-1:], smoothed, linewidth=2.5, label=f'Smoothed (window={window})', color='blue')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Training Loss (log scale)', fontsize=12)
        plt.title('Training Loss Over Time (Log Scale)', fontsize=14, fontweight='bold')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3, which='both')
        plt.tight_layout()
        plt.savefig(output_dir / 'training_loss_log.png', dpi=150)
        print(f"✓ Saved: {output_dir / 'training_loss_log.png'}")
        plt.close()
        
        # Plot 1c: Last N iterations (zoomed in)
        if len(metrics['train_losses']) > 20:
            last_n = min(100, len(metrics['train_losses']))
            plt.figure(figsize=(12, 6))
            plt.plot(x_vals[-last_n:], metrics['train_losses'][-last_n:], 
                    linewidth=1.5, marker='o', markersize=3, alpha=0.6, label='Raw', color='blue')
            if smoothed is not None and len(smoothed) >= last_n:
                plt.plot(x_vals[window-1:][-last_n:], smoothed[-last_n:], 
                        linewidth=2.5, label=f'Smoothed (window={window})', color='blue')
            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('Training Loss', fontsize=12)
            plt.title(f'Training Loss (Last {last_n} Iterations)', fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'training_loss_recent.png', dpi=150)
            print(f"✓ Saved: {output_dir / 'training_loss_recent.png'}")
            plt.close()
    
    # Plot 2: Learning Rate
    if metrics['train_lrs']:
        x_vals = metrics['train_iterations'] if metrics['train_iterations'] else range(len(metrics['train_lrs']))
        plt.figure(figsize=(12, 6))
        plt.plot(x_vals, metrics['train_lrs'], linewidth=1.5, color='orange')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'learning_rate.png', dpi=150)
        print(f"✓ Saved: {output_dir / 'learning_rate.png'}")
        plt.close()
    
    # Plot 3: Validation Loss
    if metrics['valid_losses']:
        plt.figure(figsize=(10, 6))
        epochs = list(range(len(metrics['valid_losses'])))
        plt.plot(epochs, metrics['valid_losses'], marker='o', 
                linewidth=2, markersize=8, color='red')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Validation Loss', fontsize=12)
        plt.title('Validation Loss per Epoch', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'validation_loss.png', dpi=150)
        print(f"✓ Saved: {output_dir / 'validation_loss.png'}")
        plt.close()
    
    # Plot 4: Validation Metrics (Accuracy)
    if metrics['valid_metrics']:
        plt.figure(figsize=(10, 6))
        epochs = list(range(len(metrics['valid_metrics'])))
        plt.plot(epochs, metrics['valid_metrics'], marker='s', 
                linewidth=2, markersize=8, color='green')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Validation Accuracy', fontsize=12)
        plt.title('Validation Accuracy per Epoch', fontsize=14, fontweight='bold')
        plt.ylim([0, 1.05])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'validation_accuracy.png', dpi=150)
        print(f"✓ Saved: {output_dir / 'validation_accuracy.png'}")
        plt.close()

    # Plot 4b: Validation Reason Accuracy (New)
    if metrics['valid_reason_accs']:
        plt.figure(figsize=(10, 6))
        epochs = list(range(len(metrics['valid_reason_accs'])))
        plt.plot(epochs, metrics['valid_reason_accs'], marker='^', 
                linewidth=2, markersize=8, color='purple')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Reasoning Balanced Acc', fontsize=12)
        plt.title('Validation Reasoning Accuracy per Epoch', fontsize=14, fontweight='bold')
        plt.ylim([0, 1.05])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'validation_reason_accuracy.png', dpi=150)
        print(f"✓ Saved: {output_dir / 'validation_reason_accuracy.png'}")
        plt.close()
        
    # Plot 4c: Validation Reason Accuracy Per Class (New)
    if metrics.get('valid_reason_per_class'):
        per_class_data = metrics['valid_reason_per_class']
        if per_class_data:
            # Get all unique classes
            all_classes = set()
            for d in per_class_data:
                all_classes.update(d.keys())
            all_classes = sorted(list(all_classes))
            
            if all_classes:
                plt.figure(figsize=(12, 8))
                epochs = list(range(len(per_class_data)))
                
                # Color map
                cmap = plt.get_cmap('tab10')
                
                for i, cls_name in enumerate(all_classes):
                    # Extract values for this class, filling missing with None (broken line) or previous/0
                    # Here we just skip points if missing
                    y_vals = [d.get(cls_name, None) for d in per_class_data]
                    
                    # Filter out Nones for plotting
                    valid_points = [(x, y) for x, y in zip(epochs, y_vals) if y is not None]
                    if valid_points:
                        xs, ys = zip(*valid_points)
                        plt.plot(xs, ys, marker='.', linewidth=1.5, label=cls_name, color=cmap(i % 10))
                
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel('Accuracy', fontsize=12)
                plt.title('Validation Accuracy per Reason Class', fontsize=14, fontweight='bold')
                plt.ylim([0, 1.05])
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_dir / 'validation_reason_per_class.png', dpi=150)
                print(f"✓ Saved: {output_dir / 'validation_reason_per_class.png'}")
        plt.close()
    
    # Plot 5: Combined view
    if metrics['train_losses'] and metrics['valid_losses']:
        x_vals = metrics['train_iterations'] if metrics['train_iterations'] else range(len(metrics['train_losses']))
        
        # Calculate smoothed line if enough data
        smoothed = None
        window = None
        if len(metrics['train_losses']) > 20:
            window = min(20, len(metrics['train_losses']) // 10)
            smoothed = np.convolve(metrics['train_losses'], 
                                   np.ones(window)/window, mode='valid')
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Training loss (LOG SCALE)
        axes[0, 0].plot(x_vals, metrics['train_losses'], linewidth=1, alpha=0.4, color='blue')
        if smoothed is not None:
            axes[0, 0].plot(x_vals[window-1:], smoothed, linewidth=2, color='blue')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Training Loss (log scale)')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(True, alpha=0.3, which='both')
        
        # Learning rate
        if metrics['train_lrs']:
            axes[0, 1].plot(x_vals, metrics['train_lrs'], linewidth=1.5, color='orange')
            axes[0, 1].set_xlabel('Iteration')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].set_title('Learning Rate Schedule')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Validation loss
        epochs = list(range(len(metrics['valid_losses'])))
        axes[1, 0].plot(epochs, metrics['valid_losses'], marker='o', 
                       linewidth=2, markersize=8, color='red')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Validation Loss')
        axes[1, 0].set_title('Validation Loss')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Validation accuracy
        if metrics['valid_metrics']:
            axes[1, 1].plot(epochs, metrics['valid_metrics'], marker='s', 
                           linewidth=2, markersize=8, color='green')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Validation Accuracy')
            axes[1, 1].set_title('Validation Accuracy')
            axes[1, 1].set_ylim([0, 1.05])
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'training_overview.png', dpi=150)
        print(f"✓ Saved: {output_dir / 'training_overview.png'}")
        plt.close()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    if metrics['train_losses']:
        print(f"Training Loss:")
        print(f"  Initial: {metrics['train_losses'][0]:.4f}")
        print(f"  Final:   {metrics['train_losses'][-1]:.4f}")
        print(f"  Min:     {min(metrics['train_losses']):.4f}")
        print(f"  Reduction: {(1 - metrics['train_losses'][-1]/metrics['train_losses'][0])*100:.1f}%")
    
    if metrics['valid_losses']:
        print(f"\nValidation Loss:")
        print(f"  Initial: {metrics['valid_losses'][0]:.4f}")
        print(f"  Final:   {metrics['valid_losses'][-1]:.4f}")
        print(f"  Best:    {min(metrics['valid_losses']):.4f}")
    
    if metrics['valid_metrics']:
        print(f"\nValidation Accuracy:")
        print(f"  Initial: {metrics['valid_metrics'][0]:.4f} ({metrics['valid_metrics'][0]*100:.1f}%)")
        print(f"  Final:   {metrics['valid_metrics'][-1]:.4f} ({metrics['valid_metrics'][-1]*100:.1f}%)")
        print(f"  Best:    {max(metrics['valid_metrics']):.4f} ({max(metrics['valid_metrics'])*100:.1f}%)")
        
    if metrics.get('valid_reason_accs'):
        print(f"\nReasoning Accuracy (Balanced):")
        print(f"  Final:   {metrics['valid_reason_accs'][-1]:.4f}")
        print(f"  Best:    {max(metrics['valid_reason_accs']):.4f}")
        
    if metrics.get('valid_reason_per_class'):
        print(f"\nReasoning Accuracy per Class (Final Epoch):")
        final_per_class = metrics['valid_reason_per_class'][-1]
        for cls_name, acc in sorted(final_per_class.items()):
            print(f"  {cls_name}: {acc:.4f}")

    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Plot SALMONN training metrics')
    parser.add_argument('path', type=str, 
                       help='Path to log.txt file or training output directory')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output directory for plots (default: same as log directory)')
    
    args = parser.parse_args()
    
    # Find log file
    path = Path(args.path)
    if path.is_dir():
        log_file = path / 'log.txt'
    else:
        log_file = path
    
    if not log_file.exists():
        print(f"❌ Error: Log file not found: {log_file}")
        return
    
    print(f"📊 Parsing log file: {log_file}")
    
    # Parse metrics
    metrics = parse_log_file(log_file)
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = log_file.parent / 'plots'
    
    print(f"📈 Creating plots in: {output_dir}")
    
    # Create plots
    plot_metrics(metrics, output_dir)
    
    print(f"\n✅ Done! View plots in: {output_dir}")


if __name__ == '__main__':
    main()

