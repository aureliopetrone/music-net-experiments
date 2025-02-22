import argparse
from pathlib import Path
import midiutil

def note_to_midi_number(note):
    """
    Converte una nota in formato stringa (es. 'C4', 'C#4', 'Cb4', 'C4#', 'C4b') nel suo numero MIDI.
    """
    if note == 'O':  # Pausa
        return None
    
    # Estrai nota
    pitch = note[0]  # Prima lettera è sempre la nota
    
    # Cerca alterazioni e ottava
    accidental = ''
    rest = note[1:]  # Tutto dopo la nota
    
    # Gestisci i vari formati possibili
    if '#' in rest:
        accidental = '#'
        rest = rest.replace('#', '')
    elif 'b' in rest:
        accidental = 'b'
        rest = rest.replace('b', '')
    
    # L'ottava è ciò che rimane
    octave = int(rest)
    
    # Mappa base delle note
    base_notes = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    
    # Calcola il numero MIDI
    midi_number = base_notes[pitch]
    
    # Applica le alterazioni
    if accidental == '#':
        midi_number += 1
    elif accidental == 'b':
        midi_number -= 1
    
    # Aggiungi l'ottava e normalizza a MIDI (C4 = 60)
    midi_number += (octave + 1) * 12
    
    return midi_number

def sequence_to_midi(input_file, output_file, tempo=120):
    """
    Converte una sequenza di note in un file MIDI.
    
    Args:
        input_file: File di testo con le note (una riga per timestep, note separate da virgole)
        output_file: Dove salvare il file MIDI
        tempo: Tempo in BPM
    """
    # Leggi la sequenza
    with open(input_file, 'r') as f:
        sequence = [line.strip().split(',') for line in f]
    
    # Crea il file MIDI
    midi = midiutil.MIDIFile(4)  # 4 tracce, una per canale
    
    # Imposta il tempo
    for track in range(4):
        midi.addTempo(track, 0, tempo)
    
    # Converti le note in eventi MIDI
    for time, chord in enumerate(sequence):
        for track, note in enumerate(chord):
            midi_number = note_to_midi_number(note)
            if midi_number is not None:  # Salta le pause
                # Aggiungi la nota (track, pitch, time, duration, volume)
                midi.addNote(track, 0, midi_number, time, 1, 100)
    
    # Salva il file MIDI
    with open(output_file, 'wb') as f:
        midi.writeFile(f)

def main(args):
    # Assicurati che la directory di output esista
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Converti la sequenza in MIDI
    sequence_to_midi(args.input, args.output, args.tempo)
    print(f'File MIDI salvato in: {args.output}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert note sequence to MIDI')
    parser.add_argument('--input', type=str, required=True,
                      help='Input text file with note sequence')
    parser.add_argument('--output', type=str, required=True,
                      help='Output MIDI file path')
    parser.add_argument('--tempo', type=int, default=120,
                      help='Tempo in BPM')
    
    args = parser.parse_args()
    main(args) 