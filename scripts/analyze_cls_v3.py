#!/usr/bin/env python3
"""
Анализ [CLS] эмбеддингов для v3 модели.
"""

import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
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

def get_cls_embedding(bert, tokenizer, text, device="cuda"):
    encoded = tokenizer.encode(text)
    input_ids = torch.tensor([encoded.ids]).to(device)
    
    with torch.no_grad():
        hidden_states, _ = bert(input_ids)
    
    return hidden_states[0, 0, :].cpu().numpy()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading model from: {args.model_dir}")
    
    bert, tokenizer = load_model(args.model_dir, device)
    
    # Разные тексты для анализа
    texts = [
        "Философия есть учение о всеобщем.",
        "Методология — система принципов и способов организации деятельности.",
        "Время есть форма бытия материи.",
        "Пространство и время объективны.",
        "Материя есть объективная реальность.",
        "Сознание — свойство высокоорганизованной материи.",
        "Диалектика — учение о развитии.",
        "Генерал Шкуро командовал дивизией.",
        "Танковая дивизия прорвала оборону противника.",
        "Командование приняло решение об отступлении."
    ]
    
    embeddings = []
    for text in texts:
        emb = get_cls_embedding(bert, tokenizer, text, device)
        embeddings.append(emb)
    
    embeddings = np.array(embeddings)
    
    # t-SNE визуализация
    tsne = TSNE(n_components=2, random_state=42, perplexity=3)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(14, 10))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=100, alpha=0.7)
    
    for i, text in enumerate(texts):
        plt.annotate(text[:30], (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=8, alpha=0.8)
    
    plt.title("t-SNE visualization of [CLS] embeddings (v3 model)")
    plt.tight_layout()
    
    output_path = Path("data/analysis/cls_embeddings_v3.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()
