# Standard library imports
import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import time
import multiprocessing
import math
import copy
import os

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Local imports
from src.model.music_net import EfficientHarmonicMusicNet
from src.model.tokenizer import MusicTokenizer
from src.data_processing.prepare_dataset import prepare_dataloaders

# Ottimizzazioni per MPS
if hasattr(torch.backends, 'mps'):
    torch.backends.mps.enable_tdz = False  # Disabilita TDZ per migliori performance

# Calcola il numero ottimale di workers
NUM_WORKERS = min(8, multiprocessing.cpu_count())

@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    for data, target in val_loader:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        output = model(data)
        
        # Reshape output and target for loss calculation
        batch_size, seq_len = target.shape[:2]
        output = output.view(batch_size * seq_len * 4, -1)
        target = target.view(-1)
        
        loss = criterion(output, target)
        total_loss += loss.item()
    
    return total_loss / len(val_loader)

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, start_epoch=0, checkpoint_path=None):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using {'MPS' if device.type == 'mps' else 'CPU'} device")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    patience, best_val_loss, epochs_without_improvement = 10, float('inf'), 0
    best_model = None

    # Carica il checkpoint se specificato ed esiste
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"Resuming from epoch {start_epoch} with best validation loss: {best_val_loss:.6f}")
        except (KeyError, ValueError, RuntimeError) as e:
            print(f"Error loading checkpoint: {e}. Starting training from scratch.")
            start_epoch = 0
            best_val_loss = float('inf')
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            batch_start_time = time.time()
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(data)
            output = output.view(-1, output.shape[-1])
            target = target.view(-1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            samples_per_sec = data.shape[0] / (time.time() - batch_start_time)
            train_loss += loss.item()
            print(f"\rEpoch {epoch+1}/{num_epochs} [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.6f} | {samples_per_sec:.1f} samples/s", end="")
        
        train_loss /= len(train_loader)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        print(f"\nEpoch {epoch+1}: Train loss: {train_loss:.6f}, Val loss: {val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, 'checkpoint.pt')
            print("Checkpoint saved at 'checkpoint.pt'")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping triggered")
                break
    
    return best_model

def main(args):
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() and not args.force_cpu else "cpu")
    print(f"Using {'MPS' if device.type == 'mps' else 'CPU'} device")

    # Load data
    train_loader, val_loader = prepare_dataloaders(
        args.dataset,
        args.sequence_length,
        args.batch_size
    )
    
    # Initialize tokenizer to get vocabulary size
    tokenizer = MusicTokenizer()
    vocab_size = len(tokenizer.note_to_id)
    
    # Create model
    model = EfficientHarmonicMusicNet(
        num_notes=args.vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        dropout=0.0
    )

    # Print model complexity
    complexity = model.get_complexity()
    print(f"\nModel complexity:")
    print(f"Total parameters: {complexity['total_parameters']:,}")
    print(f"Memory required: {complexity['total_memory_mb']:.2f} MB")
    
    # Train
    start_time = time.time()
    model = train_model(
        model,
        train_loader,
        val_loader,
        args.num_epochs,
        args.learning_rate,
        checkpoint_path=args.checkpoint
    )
    
    print(f'\nTraining completed in {time.time() - start_time:.2f}s')

    # Save the trained model
    torch.save(model.state_dict(), "model.pt")
    print("Final model saved at 'model.pt'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the efficient harmonic music model')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to the dataset')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--sequence-length', type=int, default=4,
                        help='Sequence length')
    parser.add_argument('--embedding-dim', type=int, default=16,
                        help='Embedding dimension')
    parser.add_argument('--hidden-size', type=int, default=32,
                        help='LSTM hidden size')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--num-epochs', type=int, default=1000,
                        help='Number of epochs')
    parser.add_argument('--force-cpu', action='store_true',
                        help='Force CPU usage even if GPU is available')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--time-limit-hours', type=float, default=24,
                        help='Time limit in hours')
    parser.add_argument('--vocab-size', type=int, default=128,
                        help='MIDI note range')
    
    args = parser.parse_args()
    main(args)