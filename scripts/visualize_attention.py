# scripts/visualize_attention.py
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
from model import BERT
from tokenizers import BertWordPieceTokenizer

def load_model_for_attention(model_path, tokenizer_path, device="cuda"):
    """Загружает модель для извлечения attention weights"""
    tokenizer = BertWordPieceTokenizer(
        str(Path(tokenizer_path) / "vocab.txt"),
        lowercase=False
    )
    
    bert = BERT(
        vocab_size=30000,
        hidden_size=384,
        num_layers=6,
        num_heads=12,
        intermediate_size=1536,
        max_position=512,
        dropout=0.1
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    bert.load_state_dict(checkpoint['model_state_dict'], strict=False)
    bert.to(device)
    bert.eval()
    
    return bert, tokenizer

def get_attention(bert, tokenizer, text, device="cuda"):
    """Получает attention weights для всех слоев"""
    encoded = tokenizer.encode(text)
    input_ids = torch.tensor([encoded.ids]).to(device)
    
    with torch.no_grad():
        _, attention_weights = bert(input_ids)
    
    # attention_weights: list of [batch, num_heads, seq_len, seq_len]
    return attention_weights, encoded.tokens

def plot_attention_head(attn, tokens, layer, head, save_path=None):
    """Визуализирует один attention head"""
    plt.figure(figsize=(12, 10))
    
    # attn: [seq_len, seq_len]
    sns.heatmap(attn, 
                xticklabels=tokens, 
                yticklabels=tokens,
                cmap='viridis',
                square=True,
                cbar_kws={'label': 'Attention weight'})
    
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model_path = "data/models/best_model.pt"
    tokenizer_path = "data/processed/tokenizer"
    
    bert, tokenizer = load_model_for_attention(model_path, tokenizer_path, device)
    
    # Тестовый текст
    text = "Философия есть учение о всеобщем."
    print(f"Analyzing: {text}")
    
    attention_weights, tokens = get_attention(bert, tokenizer, text, device)
    
    print(f"\nNumber of layers: {len(attention_weights)}")
    print(f"Number of tokens: {len(tokens)}")
    print(f"Tokens: {tokens}")
    
    # Визуализируем несколько heads
    output_dir = Path("data/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Layer 0, Heads 0, 3, 6, 9
    for head in [0, 3, 6, 9]:
        attn = attention_weights[0][0, head].cpu().numpy()
        plot_attention_head(attn, tokens, layer=0, head=head, 
                           save_path=output_dir / f"attention_layer0_head{head}.png")
    
    # Layer 3, Heads 0, 3, 6, 9
    for head in [0, 3, 6, 9]:
        attn = attention_weights[3][0, head].cpu().numpy()
        plot_attention_head(attn, tokens, layer=3, head=head,
                           save_path=output_dir / f"attention_layer3_head{head}.png")
    
    # Layer 5 (последний), Heads 0, 3, 6, 9
    for head in [0, 3, 6, 9]:
        attn = attention_weights[5][0, head].cpu().numpy()
        plot_attention_head(attn, tokens, layer=5, head=head,
                           save_path=output_dir / f"attention_layer5_head{head}.png")
    
    print(f"\nAttention visualizations saved to {output_dir}/")

if __name__ == "__main__":
    main()