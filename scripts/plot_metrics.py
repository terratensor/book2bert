#!/usr/bin/env python3
"""
Построение графиков из CSV логов обучения.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_metrics(csv_path: Path, output_dir: Path):
    """Построение графиков из CSV файла"""
    
    df = pd.read_csv(csv_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Loss по эпохам
    axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
    axes[0, 0].plot(df['epoch'], df['val_loss'], label='Val Loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 2. Perplexity
    axes[0, 1].plot(df['epoch'], df['val_perplexity'], label='Val Perplexity', marker='o', color='orange')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Perplexity')
    axes[0, 1].set_title('Validation Perplexity (lower is better)')
    axes[0, 1].grid(True)
    
    # 3. Learning rate
    axes[1, 0].plot(df['epoch'], df['learning_rate'], marker='o', color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].grid(True)
    
    # 4. Best loss progress
    axes[1, 1].plot(df['epoch'], df['best_val_loss'], marker='o', color='red')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Best Val Loss')
    axes[1, 1].set_title('Best Validation Loss Progress')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    # Сохраняем график
    output_path = output_dir / 'training_curves.png'
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")
    
    # Также сохраняем отдельно loss график для презентаций
    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['epoch'], df['train_loss'], label='Train Loss', linewidth=2)
    ax.plot(df['epoch'], df['val_loss'], label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Curves')
    ax.legend()
    ax.grid(True)
    plt.savefig(output_dir / 'loss_curves.png', dpi=150)
    
    plt.show()
    
    # Выводим статистику
    print("\n=== Training Statistics ===")
    print(f"Best validation loss: {df['val_loss'].min():.4f} (epoch {df['val_loss'].idxmin() + 1})")
    print(f"Final validation loss: {df['val_loss'].iloc[-1]:.4f}")
    print(f"Best perplexity: {df['val_perplexity'].min():.2f}")
    print(f"Total epochs: {len(df)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True, help='Path to model directory with CSV logs')
    args = parser.parse_args()
    
    logdir = Path(args.logdir)
    csv_path = logdir / 'csv' / 'metrics.csv'
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return
    
    plot_metrics(csv_path, logdir)


if __name__ == "__main__":
    main()