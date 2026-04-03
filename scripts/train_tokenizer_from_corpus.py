#!/usr/bin/env python3
"""
Обучение токенизатора из готового корпус-файла.
"""

import argparse
from tokenizers import BertWordPieceTokenizer
from pathlib import Path

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
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    tokenizer.save_model(args.output_dir)
    
    print(f"Tokenizer saved to {args.output_dir}")
    print(f"Vocabulary size: {len(tokenizer.get_vocab())}")

if __name__ == "__main__":
    main()