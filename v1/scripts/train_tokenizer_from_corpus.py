#!/usr/bin/env python3
"""
Обучение токенизатора из готового корпус-файла.
Сохраняет config.json и stats.json.
"""

import argparse
import json
from pathlib import Path
from tokenizers import BertWordPieceTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, required=True, help='путь к corpus.txt')
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--vocab-size', type=int, default=50000)
    parser.add_argument('--min-frequency', type=int, default=2)
    args = parser.parse_args()
    
    print(f"Training tokenizer on {args.corpus}...")
    
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=False,
        wordpieces_prefix="##"
    )
    
    tokenizer.train(
        files=[args.corpus],
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        limit_alphabet=1000,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_model(str(output_dir))
    
    # Сохраняем config.json
    config = {
        "vocab_size": args.vocab_size,
        "min_frequency": args.min_frequency,
        "special_tokens": ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        "model_type": "bert-wordpiece"
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Сохраняем stats.json
    stats = {
        "vocab_size": len(tokenizer.get_vocab()),
        "corpus_file": args.corpus,
        "min_frequency": args.min_frequency
    }
    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"Tokenizer saved to {args.output_dir}")
    print(f"Vocabulary size: {len(tokenizer.get_vocab())}")

if __name__ == "__main__":
    main()