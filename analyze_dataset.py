import torch
from src.model import MusicTokenizer

# Carica il dataset
data = torch.load('output/music_dataset.pt')

# Stampa informazioni
print(f'Shape: {data.shape}')
print(f'Type: {data.dtype}')
print(f'Min: {data.min()}')
print(f'Max: {data.max()}')

# Carica il tokenizer
tokenizer = MusicTokenizer()

# Mostra alcuni esempi di note
if len(data.shape) == 2:  # Se il dataset è 2D (timesteps, channels)
    for i in range(min(5, len(data))):
        notes = [tokenizer.id_to_note[idx.item()] for idx in data[i]]
        print(f'Timestep {i}: {notes}') 