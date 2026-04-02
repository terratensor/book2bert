#!/usr/bin/env python3
"""
Обучение BERT-tiny с streaming загрузкой JSONL.
Правильное возобновление: сохраняем epoch, step, total_steps_in_epoch.
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
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import BERT, BERTForMLM, count_parameters


class StreamingJSONLDataset(IterableDataset):
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
    input_ids = torch.tensor([item['input_ids'] for item in batch], dtype=torch.long)
    attention_mask = torch.tensor([item['attention_mask'] for item in batch], dtype=torch.long)
    token_type_ids = torch.tensor([item['token_type_ids'] for item in batch], dtype=torch.long)
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}


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
    def __init__(self, log_dir: Path, config: dict):
        self.log_dir = log_dir
        (log_dir / 'csv').mkdir(parents=True, exist_ok=True)
        (log_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
        
        self.step_csv_path = log_dir / 'csv' / 'step_metrics.csv'
        self.epoch_csv_path = log_dir / 'csv' / 'epoch_metrics.csv'
        
        with open(self.step_csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(['step', 'epoch', 'step_in_epoch', 'loss', 'learning_rate', 'grad_norm'])
        with open(self.epoch_csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'val_perplexity', 'learning_rate', 'best_val_loss', 'time_seconds'])
        
        self.writer = SummaryWriter(log_dir / 'tensorboard')
        self.log_path = log_dir / 'training.log'
        self.history = {'config': config, 'steps': [], 'epochs': []}
    
    def log_step(self, global_step: int, epoch: int, step_in_epoch: int, loss: float, lr: float, grad_norm: float):
        with open(self.step_csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([global_step, epoch, step_in_epoch, loss, lr, grad_norm])
        self.writer.add_scalar('train/step_loss', loss, global_step)
        self.writer.add_scalar('train/learning_rate', lr, global_step)
        self.writer.add_scalar('train/grad_norm', grad_norm, global_step)
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, 
                  learning_rate: float, best_val_loss: float, elapsed_time: float):
        val_perplexity = torch.exp(torch.tensor(val_loss)).item()
        with open(self.epoch_csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, val_perplexity, learning_rate, best_val_loss, elapsed_time])
        
        self.history['epochs'].append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                                       'val_perplexity': val_perplexity, 'learning_rate': learning_rate,
                                       'best_val_loss': best_val_loss, 'time_seconds': elapsed_time})
        with open(self.log_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
        self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
        self.writer.add_scalar('epoch/val_perplexity', val_perplexity, epoch)
        
        with open(self.log_path, 'a') as f:
            f.write(f"\n{'='*60}\nEpoch {epoch}\n  Train loss: {train_loss:.6f}\n  Val loss: {val_loss:.6f}\n")
            f.write(f"  Val perplexity: {val_perplexity:.2f}\n  Time: {elapsed_time:.1f}s\n{'='*60}\n")
    
    def close(self):
        self.writer.close()


def save_checkpoint(path, epoch, global_step, step_in_epoch, model, optimizer, scheduler, scaler, best_val_loss, config):
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'step_in_epoch': step_in_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_val_loss': best_val_loss if best_val_loss != float('inf') else None,
        'config': config
    }
    torch.save(checkpoint, path)


def train_epoch(model, dataloader, optimizer, scheduler, scaler, device, tokenizer,
                mlm_probability, grad_accum_steps, epoch, start_step, global_step, logger, 
                output_dir, config, best_val_loss, checkpoint_freq=10000, last_checkpoint_freq=1000):
    model.train()
    total_loss = 0
    steps_in_epoch = start_step
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/Training", initial=start_step)
    
    # Пропускаем уже обработанные шаги при возобновлении
    data_iter = iter(dataloader)
    for _ in range(start_step):
        try:
            next(data_iter)
        except StopIteration:
            break
    
    for step, batch in enumerate(data_iter, start=start_step):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        
        masked_input_ids, labels = mask_tokens(input_ids.clone(), tokenizer, mlm_probability)
        
        with autocast('cuda'):
            outputs = model(input_ids=masked_input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids, labels=labels)
            loss = outputs['loss'] / grad_accum_steps
        
        scaler.scale(loss).backward()
        
        step_loss = loss.item() * grad_accum_steps
        total_loss += step_loss
        steps_in_epoch += 1
        global_step += 1
        
        if (step + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            
            current_lr = scheduler.get_last_lr()[0]
            logger.log_step(global_step, epoch + 1, steps_in_epoch, step_loss, current_lr, grad_norm)
            progress_bar.set_postfix({'loss': f'{step_loss:.4f}', 'lr': f'{current_lr:.2e}'})
            progress_bar.update(1)
            
            # Last checkpoint (часто)
            if global_step % last_checkpoint_freq == 0:
                save_checkpoint(output_dir / 'checkpoints' / 'last_checkpoint.pt',
                               epoch, global_step, steps_in_epoch, model, optimizer, scheduler, scaler,
                               best_val_loss, config)
            
            # Полный чекпоинт (редко)
            if global_step % checkpoint_freq == 0:
                cp_path = output_dir / 'checkpoints' / f'checkpoint_step_{global_step}.pt'
                save_checkpoint(cp_path, epoch, global_step, steps_in_epoch, model, optimizer, scheduler, scaler,
                               best_val_loss, config)
                old = sorted(output_dir.glob('checkpoints/checkpoint_step_*.pt'))
                for p in old[:-5]:
                    p.unlink()
                print(f"\n  ✓ Checkpoint saved at step {global_step}")
    
    avg_loss = total_loss / (steps_in_epoch - start_step) if steps_in_epoch > start_step else 0
    return avg_loss, global_step, steps_in_epoch


def eval_epoch(model, dataloader, device, tokenizer, mlm_probability, max_batches=200):
    model.eval()
    total_loss = 0
    batch_count = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            masked_input_ids, labels = mask_tokens(input_ids.clone(), tokenizer, mlm_probability)
            with autocast('cuda'):
                outputs = model(input_ids=masked_input_ids, attention_mask=attention_mask,
                               token_type_ids=token_type_ids, labels=labels)
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
    parser.add_argument('--checkpoint_every', type=int, default=10000)
    parser.add_argument('--last_checkpoint_every', type=int, default=1000)
    parser.add_argument('--max_val_batches', type=int, default=200)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{config.get('name', 'tiny_bert_streaming')}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    logger = TrainingLogger(output_dir, config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print("\nLoading datasets...")
    train_dataset = StreamingJSONLDataset(args.dataset_path, "train")
    val_dataset = StreamingJSONLDataset(args.dataset_path, "val")
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'],
                              collate_fn=collate_fn, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'],
                            collate_fn=collate_fn, num_workers=0, pin_memory=True)
    
    bert = BERT(**config['model'])
    model = BERTForMLM(bert, config['model']['vocab_size'])
    model.to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'],
                            weight_decay=config['training']['weight_decay'])
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['training']['learning_rate'],
                                               total_steps=config['training']['num_epochs'] * 10000, pct_start=0.1)
    scaler = GradScaler('cuda')
    tokenizer = SimpleTokenizer(config['model']['vocab_size'])
    
    start_epoch = 0
    start_step = 0
    global_step = 0
    best_val_loss = float('inf')
    
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"\nResuming from {args.resume_from}")
        chk = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(chk['model_state_dict'])
        optimizer.load_state_dict(chk['optimizer_state_dict'])
        scheduler.load_state_dict(chk['scheduler_state_dict'])
        scaler.load_state_dict(chk['scaler_state_dict'])
        start_epoch = chk['epoch']
        start_step = chk.get('step_in_epoch', 0)
        global_step = chk.get('global_step', 0)
        best_val_loss = chk.get('best_val_loss')
        if best_val_loss is None:
            best_val_loss = float('inf')
        print(f"Resumed: epoch {start_epoch}, step_in_epoch {start_step}, global_step {global_step}, best_val_loss {best_val_loss:.4f}")
    
    print(f"\nTraining for {config['training']['num_epochs']} epochs")
    print(f"Full checkpoint every {args.checkpoint_every} steps")
    print(f"Last checkpoint every {args.last_checkpoint_every} steps\n")
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        epoch_start = datetime.now()
        step_start = start_step if epoch == start_epoch else 0
        
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{config['training']['num_epochs']}")
        print(f"{'='*50}")
        
        train_loss, global_step, steps_completed = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, tokenizer,
            config['data']['mlm_probability'], config['training'].get('gradient_accumulation_steps', 1),
            epoch, step_start, global_step, logger, output_dir, config, best_val_loss,
            args.checkpoint_every, args.last_checkpoint_every
        )
        
        val_loss = eval_epoch(model, val_loader, device, tokenizer,
                              config['data']['mlm_probability'], args.max_val_batches)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = output_dir / 'best_model.pt'
            save_checkpoint(best_path, epoch, global_step, steps_completed,
                           model, optimizer, scheduler, scaler, best_val_loss, config)
            print(f"  ✓ Best model (val_loss: {val_loss:.4f})")
        
        save_checkpoint(output_dir / 'checkpoints' / f'epoch_{epoch+1}.pt', 
                       epoch, global_step, steps_completed,
                       model, optimizer, scheduler, scaler, best_val_loss, config)
        
        logger.log_epoch(epoch + 1, train_loss, val_loss, scheduler.get_last_lr()[0],
                        best_val_loss, (datetime.now() - epoch_start).total_seconds())
        
        print(f"\nEpoch {epoch + 1} summary:")
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val loss:   {val_loss:.4f}")
        print(f"  Best val loss: {best_val_loss:.4f}")
        print(f"  Time: {(datetime.now() - epoch_start).total_seconds():.1f}s")
    
    logger.close()
    print(f"\n{'='*50}")
    print(f"Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()