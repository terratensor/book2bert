#!/usr/bin/env python3
"""
Визуализация attention patterns для v3 модели.
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "training"))

from model import BERT
from tokenizers import BertWordPieceTokenizer

def load_model(model_dir, device="cuda"):
    tokenizer = BertWordPieceTokenizer(
        str(Path(model_dir).parent.parent / "processed" / "tokenizer_militera_v3" / "vocab.txt"),
        lowercase=False
    )
    
    bert = BERT(
        vocab_size=50000,
        hidden_size=384,
        num_layers=6,
        num_heads=12,
        intermediate_size=1536,
        max_position=512,
        dropout=0.1
    )
    
    checkpoint_path = Path(model_dir) / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    bert.load_state_dict(checkpoint['model_state_dict'], strict=False)
    bert.to(device)
    bert.eval()
    
    return bert, tokenizer

def plot_attention_head(attn, tokens, layer, head, save_path=None):
    plt.figure(figsize=(12, 10))
    sns.heatmap(attn, xticklabels=tokens, yticklabels=tokens,
                cmap='viridis', square=True, cbar_kws={'label': 'Attention weight'})
    plt.title(f'Layer {layer}, Head {head}')
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--text', type=str, default="Философия есть учение о всеобщем.")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading model from: {args.model_dir}")
    
    bert, tokenizer = load_model(args.model_dir, device)
    
    print(f"Analyzing text: {args.text}")
    
    encoded = tokenizer.encode(args.text)
    input_ids = torch.tensor([encoded.ids]).to(device)
    tokens = encoded.tokens
    
    with torch.no_grad():
        _, attention_weights = bert(input_ids)
    
    print(f"Number of layers: {len(attention_weights)}")
    print(f"Number of tokens: {len(tokens)}")
    print(f"Tokens: {tokens}")
    
    output_dir = Path("data/analysis/attention")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Визуализируем несколько heads на разных слоях
    layers_to_plot = [0, 2, 4, 5]
    heads_to_plot = [0, 3, 6, 9, 11]
    
    for layer in layers_to_plot:
        for head in heads_to_plot:
            if head < attention_weights[layer].shape[1]:
                attn = attention_weights[layer][0, head].cpu().numpy()
                plot_attention_head(attn, tokens, layer, head,
                                   save_path=output_dir / f"layer{layer}_head{head}.png")
    
    print(f"Attention visualizations saved to: {output_dir}")

if __name__ == "__main__":
    main()
