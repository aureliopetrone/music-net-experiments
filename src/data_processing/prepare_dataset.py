# Third-party imports
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class MusicSequenceDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length
        self.num_sequences = (len(data) - sequence_length) // 1  # No overlap between sequences
        
    def __len__(self):
        return self.num_sequences
        
    def __getitem__(self, idx):
        # Get sequence starting at idx
        start_idx = idx
        end_idx = start_idx + self.sequence_length
        
        sequence = self.data[start_idx:end_idx]
        target = self.data[start_idx+1:end_idx+1]  # Target is next token for each position
        
        return sequence, target

def prepare_dataloaders(dataset_path, sequence_length, batch_size):
    """
    Prepare train and validation dataloaders.
    """
    print(f"Loading dataset from {dataset_path}")
    data = torch.load(dataset_path)
    print(f"Dataset size: {len(data)} timesteps")
    
    # Create dataset
    dataset = MusicSequenceDataset(data, sequence_length)
    
    # Split into train and validation (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Created dataloaders with sequence length {sequence_length} and batch size {batch_size}")
    print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepara il dataset per il training')
    parser.add_argument('--dataset', default='output/music_dataset.pt',
                      help='Percorso del dataset PyTorch')
    parser.add_argument('--sequence-length', type=int, default=32,
                      help='Lunghezza delle sequenze per il training')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Dimensione del batch')
    
    args = parser.parse_args()
    
    # Crea i DataLoader
    train_loader, val_loader = prepare_dataloaders(
        args.dataset,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size
    )
    
    # Stampa informazioni sui loader
    print(f"\nDataLoader creati:")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Mostra la shape di un batch di esempio
    for batch, target in train_loader:
        print(f"\nShape di un batch:")
        print(f"Input: {batch.shape}")
        print(f"Target: {target.shape}")
        break 