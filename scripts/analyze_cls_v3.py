#!/usr/bin/env python3
"""
Анализ [CLS] эмбеддингов для v3 модели.
Сохраняет координаты в CSV и PNG.
"""

import torch
import numpy as np
import csv
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
    
    # Военные тексты (реальные из корпуса)
    texts = [
        "Генерал Шкуро командовал дивизией в трудных условиях.",
        "Танковая дивизия прорвала оборону противника.",
        "Командование приняло решение об отступлении.",
        "Солдаты пережили тяжелую зиму под Сталинградом.",
        "Артиллерия вела огонь по позициям врага.",
        "Разведчики проникли в тыл противника.",
        "Получив подкрепление, батальон перешел в наступление.",
        "Фронтовая разведка доложила о передвижении вражеских колонн."
    ]
    
    print(f"Analyzing {len(texts)} military texts...")
    
    embeddings = []
    for text in texts:
        emb = get_cls_embedding(bert, tokenizer, text, device)
        embeddings.append(emb)
        print(f"  {text[:50]}... → embedding shape {emb.shape}")
    
    embeddings = np.array(embeddings)
    
    # t-SNE визуализация
    tsne = TSNE(n_components=2, random_state=42, perplexity=3)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Сохраняем координаты в CSV
    output_dir = Path("data/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "cls_coordinates_military.csv"
    
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'x', 'y'])
        for i, text in enumerate(texts):
            writer.writerow([text, embeddings_2d[i, 0], embeddings_2d[i, 1]])
    
    print(f"\nCoordinates saved to: {csv_path}")
    
    # Визуализация
    plt.figure(figsize=(14, 10))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=100, alpha=0.7)
    
    for i, text in enumerate(texts):
        plt.annotate(text[:30], (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=8, alpha=0.8)
    
    plt.title("t-SNE visualization of [CLS] embeddings (military texts only)")
    plt.tight_layout()
    
    output_path = output_dir / "cls_embeddings_military.png"
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
