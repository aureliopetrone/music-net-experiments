import torch
import pretty_midi
from src.model import MusicTokenizer
from pathlib import Path

def save_notes_to_text(data_tensor, tokenizer, output_path):
    """Salva le sequenze di note in un file di testo"""
    with open(str(output_path), 'w', encoding='utf-8') as f:
        f.write("Sequenza di note (formato: [canale1, canale2, canale3, canale4])\n")
        f.write("-" * 60 + "\n\n")
        
        for time_idx, timestep in enumerate(data_tensor):
            notes = [tokenizer.id_to_note[idx.item()] for idx in timestep]
            f.write(f"Timestep {time_idx}: {notes}\n")
    
    print(f"Sequenze di note salvate in: {output_path}")

def convert_to_midi(data_tensor, output_path, time_step=0.25):
    """Converte un tensore di note in un file MIDI"""
    # Inizializza il tokenizer
    tokenizer = MusicTokenizer()
    
    # Crea l'oggetto MIDI
    midi = pretty_midi.PrettyMIDI()
    
    # Crea uno strumento per ogni canale
    instruments = []
    for i in range(4):  # 4 canali
        instrument = pretty_midi.Instrument(program=0)  # Piano
        instruments.append(instrument)
        midi.instruments.append(instrument)
    
    # Converti ogni timestep in note MIDI
    for time_idx, timestep in enumerate(data_tensor):
        start_time = time_idx * time_step
        
        # Per ogni nota nel timestep
        for channel, note_idx in enumerate(timestep):
            note_name = tokenizer.id_to_note[note_idx.item()]
            if note_name != 'O':  # Salta le pause
                # Estrai informazioni dalla nota
                pitch = note_name[0]  # Prima lettera (es. 'C' da 'C4')
                octave = int(note_name[1])  # Numero dopo la lettera
                sharp = '#' in note_name  # Controlla se è diesis
                
                # Converti in numero MIDI
                base_notes = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
                midi_number = base_notes[pitch]
                if sharp:
                    midi_number += 1
                midi_number += (octave + 1) * 12
                
                # Crea la nota MIDI
                note = pretty_midi.Note(
                    velocity=100,  # Volume
                    pitch=midi_number,
                    start=start_time,
                    end=start_time + time_step
                )
                instruments[channel].notes.append(note)
    
    # Salva il file MIDI
    midi.write(str(output_path))  # Converti il Path in stringa
    print(f"File MIDI salvato in: {output_path}")

if __name__ == "__main__":
    # Carica il dataset
    data = torch.load('output/music_dataset.pt')
    
    # Crea la directory di output se non esiste
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    # Converti in MIDI
    midi_path = output_dir / 'converted_dataset.mid'
    convert_to_midi(data, midi_path)
    
    # Salva le note in formato testuale
    tokenizer = MusicTokenizer()
    text_path = output_dir / 'converted_dataset_notes.txt'
    save_notes_to_text(data, tokenizer, text_path) 