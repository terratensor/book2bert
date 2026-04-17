#!/usr/bin/env python3
"""
Обучение BERT с нуля.
"""

import os
import argparse
import json
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_from_disk
from tqdm import tqdm
import wandb

from v1.training.model import BERT, BERTForMLM, count_parameters


def collate_fn(batch):
    """Сборка батча из примеров"""
    input_ids = torch.tensor([item['input_ids'] for item in batch], dtype=torch.long)
    attention_mask = torch.tensor([item['attention_mask'] for item in batch], dtype=torch.long)
    token_type_ids = torch.tensor([item['token_type_ids'] for item in batch], dtype=torch.long)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
    }


def mask_tokens(input_ids, tokenizer, mlm_probability=0.15):
    """
    Маскирование токенов для MLM.
    input_ids: [batch, seq_len] (на GPU или CPU)
    tokenizer: объект с атрибутами id специальных токенов
    mlm_probability: вероятность маскирования
    """
    # Создаем labels (копируем input_ids)
    labels = input_ids.clone()
    
    # Создаем маску специальных токенов на том же устройстве, что и input_ids
    device = input_ids.device
    special_tokens = torch.tensor(
        [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id],
        device=device
    )
    
    # Определяем, какие позиции являются специальными токенами
    special_tokens_mask = torch.isin(input_ids, special_tokens)
    
    # Вероятностная маска для 15% токенов
    probability_matrix = torch.full(labels.shape, mlm_probability, device=device)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # -100 игнорируется в loss
    
    # 80% из замаскированных -> [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=device)).bool() & masked_indices
    input_ids[indices_replaced] = tokenizer.mask_token_id
    
    # 10% -> случайный токен (из всего словаря)
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device=device)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(tokenizer.vocab_size, labels.shape, dtype=torch.long, device=device)
    input_ids[indices_random] = random_words[indices_random]
    
    # 10% остаются без изменений
    
    return input_ids, labels

def train_epoch(model, dataloader, optimizer, scheduler, device, tokenizer, mlm_probability=0.15):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        # Переносим на устройство
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        
        # Применяем маскирование
        masked_input_ids, labels = mask_tokens(input_ids.clone(), tokenizer, mlm_probability)
        
        # Forward
        outputs = model(
            input_ids=masked_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )
        
        loss = outputs['loss']
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        # Обновляем прогресс-бар
        progress_bar.set_postfix({'loss': loss.item(), 'lr': scheduler.get_last_lr()[0]})
        
        # Логирование
        if wandb.run:
            wandb.log({
                'train_loss': loss.item(),
                'learning_rate': scheduler.get_last_lr()[0]
            })
    
    return total_loss / len(dataloader)


def eval_epoch(model, dataloader, device, tokenizer, mlm_probability=0.15):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            
            masked_input_ids, labels = mask_tokens(input_ids.clone(), tokenizer, mlm_probability)
            
            outputs = model(
                input_ids=masked_input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
            
            total_loss += outputs['loss'].item()
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='training/config/mini_bert.yaml')
    parser.add_argument('--dataset_path', type=str, default='data/processed/dataset')
    parser.add_argument('--output_dir', type=str, default='data/models')
    parser.add_argument('--use_wandb', action='store_true')
    args = parser.parse_args()
    
    # Загрузка конфига
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Загрузка датасета
    train_dataset = load_from_disk(f"{args.dataset_path}/train")
    val_dataset = load_from_disk(f"{args.dataset_path}/val")
    
    # Создаем простой токенизатор для маскирования
    class SimpleTokenizer:
        def __init__(self, vocab_size=30000):
            self.vocab_size = vocab_size
            self.cls_token_id = 2
            self.sep_token_id = 3
            self.pad_token_id = 0
            self.mask_token_id = 4
    
    tokenizer = SimpleTokenizer(config['model']['vocab_size'])
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Создание модели
    bert = BERT(**config['model'])
    model = BERTForMLM(bert, config['model']['vocab_size'])
    model.to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Оптимизатор и scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    total_steps = len(train_loader) * config['training']['num_epochs']
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['training']['learning_rate'],
        total_steps=total_steps,
        pct_start=0.1
    )
    
    # WandB
    if args.use_wandb:
        wandb.init(project="book2bert", config=config)
    
    # Создаем выходную директорию
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Обучение
    best_val_loss = float('inf')
    
    for epoch in range(config['training']['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
        
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            tokenizer, config['data']['mlm_probability']
        )
        val_loss = eval_epoch(
            model, val_loader, device,
            tokenizer, config['data']['mlm_probability']
        )
        
        print(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
        
        if args.use_wandb:
            wandb.log({'epoch': epoch, 'train_loss_epoch': train_loss, 'val_loss': val_loss})
        
        # Сохраняем лучшую модель
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, output_dir / 'best_model.pt')
            print(f"Saved best model with val_loss: {val_loss:.4f}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()