#!/usr/bin/env python3
"""Тест MLM предсказаний на v3 модели"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "training"))

from model import BERT, BERTForMLM
from tokenizers import BertWordPieceTokenizer

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Пути к v3
    model_path = "data/models/tiny_bert_militera_v2/best_model.pt"
    tokenizer_path = "data/processed/tokenizer_militera_v2"
    
    tokenizer = BertWordPieceTokenizer(
        str(Path(tokenizer_path) / "vocab.txt"),
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
    model = BERTForMLM(bert, vocab_size=50000)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("Model loaded successfully\n")
    
    test_texts = [
        "Философия есть [MASK] о всеобщем.",
        "Мера — это [MASK], в которых объект сохраняет устойчивость.",
        "Триединство: [MASK], информация и мера.",
        "Генерал [MASK] командовал дивизией в трудных условиях."
    ]
    
    print("="*60)
    print("MLM Predictions (v3, after epoch 1)")
    print("="*60 + "\n")
    
    for text in test_texts:
        print(f"Input: {text}")
        
        encoded = tokenizer.encode(text)
        input_ids = torch.tensor([encoded.ids]).to(device)
        
        mask_token_id = tokenizer.token_to_id("[MASK]")
        mask_positions = [i for i, token_id in enumerate(encoded.ids) if token_id == mask_token_id]
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs['logits']
        
        for pos in mask_positions:
            pos_logits = logits[0, pos, :]
            top_k = torch.topk(pos_logits, 10)
            
            tokens = []
            for idx in top_k.indices:
                token = tokenizer.id_to_token(idx.item())
                if token not in ['.', ',', ':', ';', '!', '?', '—', '-', '«', '»', '(', ')', '[', ']', '"', "'"]:
                    tokens.append(token)
            
            print(f"  Top-10: {tokens[:10]}")
        print()

if __name__ == "__main__":
    main()
