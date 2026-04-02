#!/usr/bin/env python3
"""
Обучение BERT-tiny с streaming загрузкой JSONL.
Поддерживает:
- Mixed precision (fp16)
- Gradient accumulation
- Сохранение чекпоинтов КАЖДЫЕ N шагов
- TensorBoard, CSV логирование
- Возобновление обучения с любого чекпоинта
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
from torch.utils.data import DataLoader, IterableDataset
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import BERT, BERTForMLM, count_parameters


class StreamingJSONLDataset(IterableDataset):
    """Streaming датасет для JSONL файлов. Не загружает всё в память."""
    
    def __init__(self, data_dir: str, split: str, max_examples: int = None):
        self.data_dir = Path(data_dir) / split
        self.files = sorted(self.data_dir.glob("*.jsonl"))
        self.max_examples = max_examples
        
    def __iter__(self):
        count = 0
        for filepath in self.files:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
                        count += 1
                        if self.max_examples and count >= self.max_examples:
                            return


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
    def __init__(self, vocab_size=50000):
        self.vocab_size = vocab_size
        self.cls_token_id = 2
        self.sep_token_id = 3
        self.pad_token_id = 0
        self.mask_token_id = 4


def mask_tokens(input_ids, tokenizer, mlm_probability=0.15):
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


class TrainingLogger:
    """Логирование в CSV, TensorBoard, JSON"""
    
    def __init__(self, log_dir: Path, config: dict):
        self.log_dir = log_dir
        self.config = config
        
        # Создаем поддиректории
        (log_dir / 'csv').mkdir(parents=True, exist_ok=True)
        (log_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
        
        # CSV файл для шагов
        self.step_csv_path = log_dir / 'csv' / 'step_metrics.csv'
        self.epoch_csv_path = log_dir / 'csv' / 'epoch_metrics.csv'
        self._init_step_csv()
        self._init_epoch_csv()
        
        # JSON история
        self.json_path = log_dir / 'training_history.json'
        self.history = {'config': config, 'steps': [], 'epochs': []}
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir / 'tensorboard')
        
        # Текстовый лог
        self.log_path = log_dir / 'training.log'
    
    def _init_step_csv(self):
        with open(self.step_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'global_step', 'epoch', 'loss', 'learning_rate', 'grad_norm'])
    
    def _init_epoch_csv(self):
        with open(self.epoch_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_perplexity', 'learning_rate', 'best_val_loss', 'time_seconds'])
    
    def log_step(self, global_step: int, epoch: int, loss: float, lr: float, grad_norm: float):
        with open(self.step_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([global_step, global_step, epoch, loss, lr, grad_norm])
        
        self.writer.add_scalar('train/step_loss', loss, global_step)
        self.writer.add_scalar('train/learning_rate', lr, global_step)
        self.writer.add_scalar('train/grad_norm', grad_norm, global_step)
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, 
                  learning_rate: float, best_val_loss: float, elapsed_time: float):
        
        val_perplexity = torch.exp(torch.tensor(val_loss)).item()
        
        with open(self.epoch_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, val_perplexity, learning_rate, best_val_loss, elapsed_time])
        
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
        
        self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
        self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
        self.writer.add_scalar('epoch/val_perplexity', val_perplexity, epoch)
        
        with open(self.log_path, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Epoch {epoch}\n")
            f.write(f"  Train loss: {train_loss:.6f}\n")
            f.write(f"  Val loss: {val_loss:.6f}\n")
            f.write(f"  Val perplexity: {val_perplexity:.2f}\n")
            f.write(f"  Time: {elapsed_time:.1f}s\n")
            f.write(f"{'='*60}\n")
    
    def close(self):
        self.writer.close()
        self.history['completed_at'] = datetime.now().isoformat()
        with open(self.json_path, 'w') as f:
            json.dump(self.history, f, indent=2)


def save_checkpoint(output_dir, model, optimizer, scheduler, scaler, epoch, global_step, 
                    val_loss, best_val_loss, config, is_best=False):
    """Сохраняет чекпоинт (всегда) и лучшую модель (если is_best)"""
    
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'val_loss': val_loss,
        'best_val_loss': best_val_loss,
        'config': config
    }
    
    # Сохраняем чекпоинт с номером шага
    torch.save(checkpoint, output_dir / 'checkpoints' / f'checkpoint_step_{global_step}.pt')
    
    # Сохраняем последний чекпоинт (для возобновления)
    torch.save(checkpoint, output_dir / 'checkpoints' / 'last_checkpoint.pt')
    
    if is_best:
        torch.save(checkpoint, output_dir / 'best_model.pt')


def train_epoch(model, dataloader, optimizer, scheduler, scaler, device, tokenizer, 
                mlm_probability, gradient_accumulation_steps, epoch, global_step,
                logger, checkpoint_every, output_dir, max_batches=None):
    """Обучение с частыми чекпоинтами"""
    model.train()
    total_loss = 0
    accumulated_loss = 0
    steps_in_epoch = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/Training")
    
    for step, batch in enumerate(progress_bar):
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
            loss = outputs['loss']
            loss = loss / gradient_accumulation_steps
        
        scaler.scale(loss).backward()
        
        step_loss = loss.item() * gradient_accumulation_steps
        total_loss += step_loss
        accumulated_loss += step_loss
        steps_in_epoch += 1
        global_step += 1
        
        if (step + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            
            # Логирование шага
            current_lr = scheduler.get_last_lr()[0]
            logger.log_step(global_step, epoch + 1, step_loss, current_lr, grad_norm)
            
            progress_bar.set_postfix({
                'loss': f'{step_loss:.4f}',
                'lr': f'{current_lr:.2e}'
            })
        
        # Сохраняем чекпоинт каждые checkpoint_every шагов
        if global_step % checkpoint_every == 0:
            save_checkpoint(
                output_dir, model, optimizer, scheduler, scaler,
                epoch, global_step, None, None, None,
                is_best=False
            )
            print(f"\n  ✓ Checkpoint saved at step {global_step}")
        
        if max_batches and steps_in_epoch >= max_batches:
            break
    
    avg_loss = total_loss / steps_in_epoch
    return avg_loss, global_step


def eval_epoch(model, dataloader, device, tokenizer, mlm_probability, max_batches=200):
    """Валидация"""
    model.eval()
    total_loss = 0
    batch_count = 0
    
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
            batch_count += 1
            if batch_count >= max_batches:
                break
    
    return total_loss / batch_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='data/models')
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--checkpoint_every', type=int, default=1000, help='Save checkpoint every N steps')
    parser.add_argument('--max_train_batches', type=int, default=None, help='Limit training batches per epoch')
    parser.add_argument('--max_val_batches', type=int, default=200)
    parser.add_argument('--use_wandb', action='store_true')
    args = parser.parse_args()
    
    # Загрузка конфига
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Создаем выходную директорию
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = config.get('name', 'tiny_bert_streaming')
    output_dir = Path(args.output_dir) / f"{model_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем конфиг
    with open(output_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # Логгер
    logger = TrainingLogger(output_dir, config)
    
    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"Logging to: {output_dir}")
    
    # Загрузка датасета
    print("\nLoading datasets in streaming mode...")
    train_dataset = StreamingJSONLDataset(args.dataset_path, "train", max_examples=args.max_train_batches)
    val_dataset = StreamingJSONLDataset(args.dataset_path, "val")
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], 
                              collate_fn=collate_fn, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], 
                            collate_fn=collate_fn, num_workers=0, pin_memory=True)
    
    # Модель
    print("\nCreating model...")
    bert = BERT(**config['model'])
    model = BERTForMLM(bert, config['model']['vocab_size'])
    model.to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Оптимизатор и scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], 
                            weight_decay=config['training']['weight_decay'])
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['training']['learning_rate'], 
                                               total_steps=config['training']['num_epochs'] * 10000, pct_start=0.1)
    scaler = GradScaler()
    tokenizer = SimpleTokenizer(config['model']['vocab_size'])
    
    # Возобновление
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"\nResuming from {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint.get('global_step', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}, step {global_step}, best_val_loss: {best_val_loss:.4f}")
    
    # Обучение
    print(f"\nStarting training for {config['training']['num_epochs']} epochs")
    print(f"Checkpoint every {args.checkpoint_every} steps")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Log directory: {output_dir}\n")
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        epoch_start = datetime.now()
        
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{config['training']['num_epochs']}")
        print(f"{'='*50}")
        
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, tokenizer,
            config['data']['mlm_probability'], config['training'].get('gradient_accumulation_steps', 1),
            epoch, global_step, logger, args.checkpoint_every, output_dir, args.max_train_batches
        )
        
        val_loss = eval_epoch(model, val_loader, device, tokenizer, 
                               config['data']['mlm_probability'], args.max_val_batches)
        
        epoch_time = (datetime.now() - epoch_start).total_seconds()
        current_lr = scheduler.get_last_lr()[0]
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(output_dir, model, optimizer, scheduler, scaler,
                           epoch, global_step, val_loss, best_val_loss, config, is_best=True)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        
        # Сохраняем чекпоинт в конце эпохи
        save_checkpoint(output_dir, model, optimizer, scheduler, scaler,
                       epoch, global_step, val_loss, best_val_loss, config, is_best=False)
        
        logger.log_epoch(epoch + 1, train_loss, val_loss, current_lr, best_val_loss, epoch_time)
        
        print(f"\nEpoch {epoch + 1} summary:")
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val loss:   {val_loss:.4f}")
        print(f"  Learning rate: {current_lr:.2e}")
        print(f"  Time: {epoch_time:.1f}s")
    
    logger.close()
    
    print("\n" + "="*50)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model and logs saved to: {output_dir}")
    print("="*50)


if __name__ == "__main__":
    main()