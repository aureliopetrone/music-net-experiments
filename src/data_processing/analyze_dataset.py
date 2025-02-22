import torch
from src.model.tokenizer import MusicTokenizer
import numpy as np
from collections import Counter

def analyze_dataset(dataset_path):
    """Analizza il dataset musicale"""
    # Carica il dataset e il tokenizer
    print("Caricamento dataset...")
    data = torch.load(dataset_path)
    tokenizer = MusicTokenizer()
    
    # Informazioni di base sul dataset
    print("\nInformazioni di base:")
    print(f"Shape del dataset: {data.shape}")
    print(f"Tipo di dati: {data.dtype}")
    print(f"Numero di timestep: {data.shape[0]}")
    print(f"Numero di canali: {data.shape[1]}")
    
    # Analisi delle note
    print("\nAnalisi delle note:")
    all_notes = []
    for timestep in data:
        notes = [tokenizer.id_to_note[idx.item()] for idx in timestep if tokenizer.id_to_note[idx.item()] != 'O']
        all_notes.extend(notes)
    
    note_counts = Counter(all_notes)
    total_notes = len(all_notes)
    
    print(f"Numero totale di note (escluse le pause): {total_notes}")
    print(f"Note uniche utilizzate: {len(note_counts)}")
    
    # Mostra le 10 note più comuni
    print("\nLe 10 note più comuni:")
    for note, count in note_counts.most_common(10):
        percentage = (count / total_notes) * 100
        print(f"{note}: {count} volte ({percentage:.1f}%)")
    
    # Analisi degli accordi
    print("\nAnalisi degli accordi:")
    chord_sizes = []
    for timestep in data:
        notes = [note for note in timestep if tokenizer.id_to_note[note.item()] != 'O']
        if len(notes) > 0:  # Conta solo i timestep con almeno una nota
            chord_sizes.append(len(notes))
    
    avg_chord_size = np.mean(chord_sizes)
    max_chord_size = np.max(chord_sizes)
    
    print(f"Dimensione media degli accordi: {avg_chord_size:.2f} note")
    print(f"Dimensione massima degli accordi: {max_chord_size} note")
    
    # Distribuzione delle dimensioni degli accordi
    chord_size_dist = Counter(chord_sizes)
    print("\nDistribuzione delle dimensioni degli accordi:")
    for size in sorted(chord_size_dist.keys()):
        count = chord_size_dist[size]
        percentage = (count / len(chord_sizes)) * 100
        print(f"{size} note: {count} volte ({percentage:.1f}%)")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analizza il dataset musicale')
    parser.add_argument('--dataset', default='output/music_dataset.pt',
                      help='Percorso del dataset PyTorch')
    
    args = parser.parse_args()
    analyze_dataset(args.dataset) 