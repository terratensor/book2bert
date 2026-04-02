# scripts/evaluate_mlm.py
import torch
import sys
from pathlib import Path

# Добавляем путь к training
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))

from model import BERT, BERTForMLM
from tokenizers import BertWordPieceTokenizer

def load_model(model_path, tokenizer_path, device="cuda"):
    # Загружаем токенизатор
    tokenizer = BertWordPieceTokenizer(
        str(Path(tokenizer_path) / "vocab.txt"),
        lowercase=False
    )
    
    # Создаем модель
    bert = BERT(
        vocab_size=30000,
        hidden_size=512,
        num_layers=8,
        num_heads=16,
        intermediate_size=1536,
        max_position=512,
        dropout=0.1
    )
    model = BERTForMLM(bert, vocab_size=30000)
    
    # Загружаем веса
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, tokenizer

def predict_masked(model, tokenizer, text, mask_token="[MASK]", device="cuda"):
    # Токенизируем
    encoded = tokenizer.encode(text)
    input_ids = torch.tensor([encoded.ids]).to(device)
    
    # Находим позицию [MASK]
    mask_id = tokenizer.token_to_id(mask_token)
    mask_positions = (input_ids[0] == mask_id).nonzero(as_tuple=True)[0]
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs['logits']  # [1, seq_len, vocab_size]
    
    # Для каждой маски предсказываем топ-5 токенов
    predictions = []
    for pos in mask_positions:
        pos_logits = logits[0, pos, :]
        top_k = torch.topk(pos_logits, 5)
        
        tokens = []
        for idx in top_k.indices:
            tokens.append(tokenizer.id_to_token(idx.item()))
        predictions.append(tokens)
    
    return predictions

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Пути
    model_path = "data/models/best_model.pt"
    tokenizer_path = "data/processed/tokenizer"
    
    # Загружаем модель
    model, tokenizer = load_model(model_path, tokenizer_path, device)
    
    # Тестовые примеры
    test_texts = [
        "Философия есть [MASK] о всеобщем.",
        "Мера — это [MASK], в которых объект сохраняет устойчивость.",
        "Триединство: [MASK], информация и мера.",
        "Человек, который среди земных невзгод обходится без [MASK], подобен тому, кто шагает с непокрытой головой под проливным дождем."
    ]
    
    print("\n=== MLM Predictions ===\n")
    for text in test_texts:
        print(f"Input: {text}")
        predictions = predict_masked(model, tokenizer, text, device=device)
        for i, preds in enumerate(predictions):
            print(f"  Mask {i+1}: {preds}")
        print()

if __name__ == "__main__":
    main()