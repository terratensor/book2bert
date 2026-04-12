# scripts/our_sentence_transformer.py
"""
Адаптер для использования нашей модели как SentenceTransformer
"""

import torch
import sys
from pathlib import Path
from typing import List, Union
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "training"))

from model import BERT
from tokenizers import BertWordPieceTokenizer


class OurSentenceTransformer:
    """Обёртка для нашей модели, совместимая с SentenceTransformer API"""
    
    def __init__(self, model_path: str, tokenizer_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Загружаем токенизатор
        self.tokenizer = BertWordPieceTokenizer(
            str(Path(tokenizer_path) / "vocab.txt"),
            lowercase=False
        )
        
        # Загружаем модель
        bert = BERT(
            vocab_size=120000,
            hidden_size=512,
            num_layers=12,
            num_heads=8,
            intermediate_size=2048,
            max_position=512,
            dropout=0.1
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        # Извлекаем веса BERT (без MLM головы)
        bert_state = {k.replace('bert.', ''): v for k, v in checkpoint['model_state_dict'].items() 
                     if k.startswith('bert.')}
        bert.load_state_dict(bert_state)
        self.model = bert
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded: {sum(p.numel() for p in self.model.parameters()):,} params")
    
    def encode(self, sentences: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """Получение эмбеддингов (совместимо с SentenceTransformer)"""
        if isinstance(sentences, str):
            sentences = [sentences]
        
        all_embeddings = []
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            batch_embeddings = []
            
            for text in batch:
                encoded = self.tokenizer.encode(text)
                input_ids = torch.tensor([encoded.ids]).to(self.device)
                attention_mask = torch.tensor([encoded.attention_mask]).to(self.device)
                
                with torch.no_grad():
                    hidden_states, _ = self.model(input_ids, attention_mask)
                
                # [CLS] эмбеддинг (как в SentenceTransformer по умолчанию)
                cls_embedding = hidden_states[0, 0, :].cpu().numpy()
                batch_embeddings.append(cls_embedding)
            
            all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings)
    
    def to(self, device: str):
        """Смена устройства"""
        self.device = torch.device(device)
        self.model.to(self.device)
        return self