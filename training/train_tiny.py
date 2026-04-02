#!/usr/bin/env python3
"""
Обучение BERT-tiny (22M параметров) с нуля.
Поддерживает:
- Mixed precision (fp16)
- Gradient accumulation
- Сохранение чекпоинтов
- Возобновление обучения
- Множественное логирование (CSV, TensorBoard, JSON)
"""

import os
import sys
import argparse
import yaml
import json
import csv
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
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
    """Маскирование токенов для MLM"""
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
                mlm_probability, gradient_accumulation_steps, epoch, writer, log_freq=50):
    """Обучение одной эпохи с логированием шагов"""
    model.train()
    total_loss = 0
    step_losses = []
    
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
        step_loss = loss.item() * gradient_accumulation_steps
        total_loss += step_loss
        step_losses.append(step_loss)
        
        if (step + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        
        # Логирование на каждом log_freq шаге
        if step % log_freq == 0:
            current_lr = scheduler.get_last_lr()[0]
            global_step = epoch * len(dataloader) + step
            writer.add_scalar('train/step_loss', step_loss, global_step)
            writer.add_scalar('train/learning_rate', current_lr, global_step)
            
            progress_bar.set_postfix({
                'loss': f'{step_loss:.4f}',
                'lr': f'{current_lr:.2e}'
            })
    
    avg_train_loss = total_loss / len(dataloader)
    return avg_train_loss, step_losses


def eval_epoch(model, dataloader, device, tokenizer, mlm_probability, writer, epoch):
    """Валидация одной эпохи с логированием"""
    model.eval()
    total_loss = 0
    batch_losses = []
    
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
            
            batch_loss = outputs['loss'].item()
            total_loss += batch_loss
            batch_losses.append(batch_loss)
    
    avg_val_loss = total_loss / len(dataloader)
    
    # Логирование валидационных метрик
    writer.add_scalar('val/loss', avg_val_loss, epoch)
    writer.add_scalar('val/perplexity', torch.exp(torch.tensor(avg_val_loss)), epoch)
    
    return avg_val_loss, batch_losses


class TrainingLogger:
    """Класс для управления всеми видами логов"""
    
    def __init__(self, log_dir: Path, config: dict):
        self.log_dir = log_dir
        self.config = config
        
        # Создаем поддиректории
        (log_dir / 'csv').mkdir(parents=True, exist_ok=True)
        (log_dir / 'json').mkdir(parents=True, exist_ok=True)
        (log_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
        
        # CSV файл для метрик по эпохам
        self.csv_path = log_dir / 'csv' / 'metrics.csv'
        self._init_csv()
        
        # JSON файл для полной истории
        self.json_path = log_dir / 'json' / 'training_history.json'
        self.history = {'config': config, 'epochs': []}
        
        # Текстовый лог файл
        self.log_path = log_dir / 'training.log'
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir / 'tensorboard')
    
    def _init_csv(self):
        """Инициализация CSV файла с заголовками"""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 
                'train_loss', 
                'val_loss', 
                'val_perplexity',
                'learning_rate',
                'best_val_loss',
                'time_seconds',
                'gradient_accumulation_steps',
                'effective_batch_size'
            ])
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, 
                  learning_rate: float, best_val_loss: float, elapsed_time: float,
                  gradient_accumulation_steps: int, effective_batch_size: int):
        """Логирование метрик эпохи во все форматы"""
        
        val_perplexity = torch.exp(torch.tensor(val_loss)).item()
        
        # CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, train_loss, val_loss, val_perplexity, 
                learning_rate, best_val_loss, elapsed_time,
                gradient_accumulation_steps, effective_batch_size
            ])
        
        # JSON история
        self.history['epochs'].append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_perplexity': val_perplexity,
            'learning_rate': learning_rate,
            'best_val_loss': best_val_loss,
            'time_seconds': elapsed_time
        })
        
        with open(self.json_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # TensorBoard (уже сделано в eval_epoch и train_epoch, добавляем дополнительно)
        self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
        self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
        self.writer.add_scalar('epoch/val_perplexity', val_perplexity, epoch)
        self.writer.add_scalar('epoch/learning_rate', learning_rate, epoch)
        
        # Текстовый лог
        with open(self.log_path, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Epoch {epoch}\n")
            f.write(f"  Train loss: {train_loss:.6f}\n")
            f.write(f"  Val loss:   {val_loss:.6f}\n")
            f.write(f"  Val perplexity: {val_perplexity:.4f}\n")
            f.write(f"  Learning rate: {learning_rate:.2e}\n")
            f.write(f"  Best val loss: {best_val_loss:.6f}\n")
            f.write(f"  Time: {elapsed_time:.1f} seconds\n")
            f.write(f"{'='*60}\n")
        
        print(f"\n  ✓ Logged to {self.log_dir}")
    
    def log_initial(self, model_params: int, total_steps: int, device_info: str):
        """Логирование начальной информации"""
        with open(self.log_path, 'w') as f:
            f.write(f"Training started: {datetime.now().isoformat()}\n")
            f.write(f"Config: {json.dumps(self.config, indent=2)}\n")
            f.write(f"Model parameters: {model_params:,}\n")
            f.write(f"Total steps: {total_steps}\n")
            f.write(f"Device: {device_info}\n")
            f.write(f"Log directory: {self.log_dir}\n")
            f.write("-" * 60 + "\n")
    
    def close(self):
        """Закрытие всех логгеров"""
        self.writer.close()
        
        # Сохраняем финальную историю
        self.history['completed_at'] = datetime.now().isoformat()
        with open(self.json_path, 'w') as f:
            json.dump(self.history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train BERT-tiny with mixed precision")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--dataset_path', type=str, default='data/processed/dataset')
    parser.add_argument('--output_dir', type=str, default='data/models')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--log_every', type=int, default=50, help='Log every N steps')
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
    
    # Инициализация логгера
    logger = TrainingLogger(output_dir, config)
    
    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_info = f"{torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)" if torch.cuda.is_available() else "CPU"
    print(f"Using device: {device}")
    print(f"Logging to: {output_dir}")
    
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
    
    model_params = count_parameters(model)
    print(f"Model parameters: {model_params:,}")
    
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
    
    # scaler = GradScaler()
    scaler = torch.amp.GradScaler('cuda')
    tokenizer = SimpleTokenizer(config['model']['vocab_size'])
    
    # Логируем начальную информацию
    logger.log_initial(model_params, total_steps, device_info)
    
    # WandB (опционально)
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
        
        # Восстанавливаем историю из логов (опционально)
        if 'history' in checkpoint:
            logger.history = checkpoint['history']
    
    # Обучение
    print(f"\nStarting training for {config['training']['num_epochs']} epochs")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Total steps: {total_steps}")
    print(f"Logs will be saved to: {output_dir}\n")
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        epoch_start_time = datetime.now()
        
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{config['training']['num_epochs']}")
        print(f"{'='*50}")
        
        train_loss, _ = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, tokenizer,
            config['data']['mlm_probability'], gradient_accumulation_steps, epoch,
            logger.writer, args.log_every
        )
        
        val_loss, _ = eval_epoch(
            model, val_loader, device, tokenizer, config['data']['mlm_probability'],
            logger.writer, epoch
        )
        
        epoch_time = (datetime.now() - epoch_start_time).total_seconds()
        current_lr = scheduler.get_last_lr()[0]
        
        # Обновляем best loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Сохраняем лучшую модель
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_loss': val_loss,
                'config': config,
                'history': logger.history
            }, output_dir / 'best_model.pt')
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        
        # Логирование эпохи
        logger.log_epoch(
            epoch=epoch + 1,
            train_loss=train_loss,
            val_loss=val_loss,
            learning_rate=current_lr,
            best_val_loss=best_val_loss,
            elapsed_time=epoch_time,
            gradient_accumulation_steps=gradient_accumulation_steps,
            effective_batch_size=effective_batch_size
        )
        
        # Сохраняем чекпоинт эпохи
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'val_loss': val_loss,
            'config': config,
            'history': logger.history
        }, output_dir / 'checkpoints' / f'checkpoint_epoch_{epoch+1}.pt')
        
        print(f"\nEpoch {epoch + 1} summary:")
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val loss:   {val_loss:.4f}")
        print(f"  Learning rate: {current_lr:.2e}")
        print(f"  Time: {epoch_time:.1f}s")
        
        if args.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_perplexity': torch.exp(torch.tensor(val_loss)).item(),
                'learning_rate': current_lr
            })
    
    logger.close()
    
    print("\n" + "="*50)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model and logs saved to: {output_dir}")
    print(f"  - CSV metrics: {output_dir}/csv/metrics.csv")
    print(f"  - TensorBoard: {output_dir}/tensorboard")
    print(f"  - JSON history: {output_dir}/json/training_history.json")
    print(f"  - Training log: {output_dir}/training.log")
    print("="*50)
    
    # Подсказка для просмотра графиков
    print("\nTo view TensorBoard metrics, run:")
    print(f"  tensorboard --logdir={output_dir}/tensorboard")
    print("\nTo plot metrics from CSV:")
    print(f"  python scripts/plot_metrics.py --logdir {output_dir}")


if __name__ == "__main__":
    main()