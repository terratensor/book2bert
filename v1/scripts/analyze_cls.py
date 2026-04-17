# scripts/analyze_cls.py
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
from model import BERT
from tokenizers import BertWordPieceTokenizer

def load_bert(model_path, tokenizer_path, device="cuda"):
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

def get_cls_embedding(bert, tokenizer, text, device="cuda"):
    encoded = tokenizer.encode(text)
    input_ids = torch.tensor([encoded.ids]).to(device)
    
    with torch.no_grad():
        hidden_states, _ = bert(input_ids)
    
    # [CLS] — первый токен
    cls_embedding = hidden_states[0, 0, :].cpu().numpy()
    return cls_embedding

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_path = "data/models/best_model.pt"
    tokenizer_path = "data/processed/tokenizer"
    
    bert, tokenizer = load_bert(model_path, tokenizer_path, device)
    
    # Разные тексты для анализа
    texts = [
        "Философия есть учение о всеобщем.",
        "Методология — система принципов и способов организации деятельности.",
        "Время есть форма бытия материи.",
        "Пространство и время объективны.",
        "Материя есть объективная реальность.",
        "Сознание — свойство высокоорганизованной материи.",
        "Диалектика — учение о развитии.",
        "Логика — наука о формах мышления.",
    ]
    
    embeddings = []
    for text in texts:
        emb = get_cls_embedding(bert, tokenizer, text, device)
        embeddings.append(emb)
    
    embeddings = np.array(embeddings)
    
    # t-SNE визуализация
    tsne = TSNE(n_components=2, random_state=42, perplexity=3)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(12, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=100)
    
    for i, text in enumerate(texts):
        plt.annotate(text[:30], (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=8)
    
    plt.title("t-SNE visualization of [CLS] embeddings")
    plt.tight_layout()
    plt.savefig("data/analysis/cls_embeddings.png", dpi=150)
    plt.show()
    
    print("Saved to data/analysis/cls_embeddings.png")

if __name__ == "__main__":
    main()