#!/usr/bin/env python3
"""
Обучение BERT-tiny (22M параметров) с нуля.
Поддерживает:
- Mixed precision (fp16)
- Gradient accumulation
- Сохранение чекпоинтов
- Возобновление обучения
"""

import os
import argparse
import yaml
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from datasets import load_from_disk
from tqdm import tqdm

from model import BERT, BERTForMLM, count_parameters


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


class SimpleTokenizer:
    """Простой токенизатор для маскирования"""
    def __init__(self, vocab_size=30000):
        self.vocab_size = vocab_size
        self.cls_token_id = 2
        self.sep_token_id = 3
        self.pad_token_id = 0
        self.mask_token_id = 4


def mask_tokens(input_ids, tokenizer, mlm_probability=0.15):
    """
    Маскирование токенов для MLM.
    input_ids: [batch, seq_len] (на GPU или CPU)
    """
    labels = input_ids.clone()
    device = input_ids.device
    
    special_tokens = torch.tensor(
        [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id],
        device=device
    )
    
    special_tokens_mask = torch.isin(input_ids, special_tokens)
    probability_matrix = torch.full(labels.shape, mlm_probability, device=device)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100
    
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=device)).bool() & masked_indices
    input_ids[indices_replaced] = tokenizer.mask_token_id
    
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device=device)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(tokenizer.vocab_size, labels.shape, dtype=torch.long, device=device)
    input_ids[indices_random] = random_words[indices_random]
    
    return input_ids, labels


def train_epoch(model, dataloader, optimizer, scheduler, scaler, device, tokenizer, 
                mlm_probability, gradient_accumulation_steps, epoch, log_freq=50):
    """Обучение одной эпохи"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/Training")
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        
        masked_input_ids, labels = mask_tokens(input_ids.clone(), tokenizer, mlm_probability)
        
        # with autocast():
        with torch.amp.autocast('cuda'):
            outputs = model(
                input_ids=masked_input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
            loss = outputs['loss']
            loss = loss / gradient_accumulation_steps
        
        scaler.scale(loss).backward()
        total_loss += loss.item() * gradient_accumulation_steps
        
        if (step + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        
        if step % log_freq == 0:
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                'lr': f'{current_lr:.2e}'
            })
    
    return total_loss / len(dataloader)


def eval_epoch(model, dataloader, device, tokenizer, mlm_probability):
    """Валидация одной эпохи"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            
            masked_input_ids, labels = mask_tokens(input_ids.clone(), tokenizer, mlm_probability)
            
            with autocast():
                outputs = model(
                    input_ids=masked_input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )
            
            total_loss += outputs['loss'].item()
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="Train BERT-tiny with mixed precision")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--dataset_path', type=str, default='data/processed/dataset')
    parser.add_argument('--output_dir', type=str, default='data/models')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--use_wandb', action='store_true')
    args = parser.parse_args()
    
    # Загрузка конфига
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Создаем выходную директорию
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = config.get('name', 'tiny_bert')
    output_dir = Path(args.output_dir) / f"{model_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем конфиг
    with open(output_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Загрузка датасета
    print("\nLoading datasets...")
    train_dataset = load_from_disk(f"{args.dataset_path}/train")
    val_dataset = load_from_disk(f"{args.dataset_path}/val")
    print(f"Train examples: {len(train_dataset)}")
    print(f"Val examples: {len(val_dataset)}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    # Создание модели
    print("\nCreating model...")
    bert = BERT(**config['model'])
    model = BERTForMLM(bert, config['model']['vocab_size'])
    model.to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Оптимизатор
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Scheduler
    gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
    effective_batch_size = config['training']['batch_size'] * gradient_accumulation_steps
    steps_per_epoch = len(train_loader) // gradient_accumulation_steps
    total_steps = steps_per_epoch * config['training']['num_epochs']
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['training']['learning_rate'],
        total_steps=total_steps,
        pct_start=0.1
    )
    
    # Mixed precision scaler
    # scaler = GradScaler()
    scaler = torch.amp.GradScaler('cuda')
    
    # Токенизатор для маскирования
    tokenizer = SimpleTokenizer(config['model']['vocab_size'])
    
    # WandB
    if args.use_wandb:
        import wandb
        wandb.init(project="book2bert", name=f"{model_name}_{timestamp}", config=config)
    
    # Возобновление из чекпоинта
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"\nResuming from {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}, best_val_loss: {best_val_loss:.4f}")
    
    # Обучение
    print(f"\nStarting training for {config['training']['num_epochs']} epochs")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Total steps: {total_steps}\n")
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{config['training']['num_epochs']}")
        print(f"{'='*50}")
        
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, tokenizer,
            config['data']['mlm_probability'], gradient_accumulation_steps, epoch
        )
        
        val_loss = eval_epoch(
            model, val_loader, device, tokenizer, config['data']['mlm_probability']
        )
        
        print(f"\nEpoch {epoch + 1} summary:")
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val loss:   {val_loss:.4f}")
        print(f"  Learning rate: {scheduler.get_last_lr()[0]:.2e}")
        
        if args.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': scheduler.get_last_lr()[0]
            })
        
        # Сохраняем лучшую модель
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, output_dir / 'best_model.pt')
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        
        # Сохраняем чекпоинт эпохи
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'val_loss': val_loss,
            'config': config
        }, output_dir / f'checkpoint_epoch_{epoch+1}.pt')
    
    print("\n" + "="*50)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir}")
    print("="*50)


if __name__ == "__main__":
    main()