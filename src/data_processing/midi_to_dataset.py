# Standard library imports
import os
from pathlib import Path

# Third-party imports
import numpy as np
import pretty_midi
import torch

# Local imports
from src.model import MusicTokenizer

class MidiConverter:
    def __init__(self, channels=4, time_step=0.25, max_vocab_size=128):  # time_step = quarter note
        self.tokenizer = MusicTokenizer(max_vocab_size=max_vocab_size)
        self.channels = channels
        self.time_step = time_step
    
    def _note_to_pitch_name(self, note_number):
        """Convert MIDI note number to pitch name (e.g., 60 -> 'C4')"""
        NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (note_number // 12) - 1
        note = NOTES[note_number % 12]
        # Format note to match vocab.txt format (all sharps after octave number)
        if '#' in note:
            base_note = note[0]
            return f"{base_note}{octave}#"  # Format for all sharp notes (e.g., C4#)
        return f"{note}{octave}"
    
    def convert_midi_file(self, midi_path):
        """Convert a single MIDI file to our model's format"""
        try:
            # Convert Path object to string
            midi_path_str = str(midi_path)
            midi_data = pretty_midi.PrettyMIDI(midi_path_str)
            
            # Get the total duration in beats
            total_beats = int(midi_data.get_end_time() / self.time_step)
            
            # Initialize empty sequence
            sequence = []
            
            # Process each beat
            for beat in range(total_beats):
                start_time = beat * self.time_step
                end_time = (beat + 1) * self.time_step
                
                # Find all notes that are active in this time step
                active_notes = []
                for instrument in midi_data.instruments:
                    if instrument.is_drum:  # Skip drum tracks
                        continue
                    for note in instrument.notes:
                        if note.start < end_time and note.end > start_time:
                            active_notes.append(note.pitch)
                
                # Convert MIDI note numbers to pitch names
                if active_notes:
                    # Sort and take up to channels notes
                    active_notes = sorted(set(active_notes))[:self.channels]
                    chord = [self._note_to_pitch_name(note) for note in active_notes]
                    # Pad with 'O' if fewer than channels notes
                    chord.extend(['O'] * (self.channels - len(chord)))
                else:
                    # If no notes are active, represent as a rest
                    chord = ['O'] * self.channels
                
                sequence.append(chord)
            
            # Convert to tensor using tokenizer
            tokens = torch.tensor([
                [self.tokenizer.note_to_id[note] for note in bar]
                for bar in sequence
            ], dtype=torch.long)
            
            return tokens
            
        except Exception as e:
            print(f"Error processing {midi_path}: {str(e)}")
            if isinstance(e, KeyError):
                print("Note not found in vocabulary. Available notes in vocabulary:", sorted(self.tokenizer.note_to_id.keys()))
                print("Problematic note:", e.args[0])  # Print the specific note that caused the error
            return None

def process_midi_directory(input_dir, output_dir):
    """Process all MIDI files in a directory and save the converted data"""
    converter = MidiConverter(max_vocab_size=128)  # Limiting vocabulary size to 128
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each MIDI file
    all_sequences = []
    midi_files = list(Path(input_dir).glob('**/*.mid')) + \
                 list(Path(input_dir).glob('**/*.midi')) + \
                 list(Path(input_dir).glob('**/*.kar'))  # Aggiunto supporto per file .kar
    
    for midi_path in midi_files:
        print(f"Processing {midi_path}")
        sequence = converter.convert_midi_file(midi_path)
        if sequence is not None:
            all_sequences.append(sequence)
    
    if all_sequences:
        # Combine all sequences and save
        combined_data = torch.cat(all_sequences, dim=0)
        output_path = os.path.join(output_dir, 'music_dataset.pt')
        torch.save(combined_data, output_path)
        print(f"\nDataset saved to {output_path}")
        print(f"Total sequences: {len(all_sequences)}")
        print(f"Total timesteps: {combined_data.size(0)}")
    else:
        print("No valid sequences were processed.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert MIDI files to tokenized dataset')
    parser.add_argument('input_dir', help='Directory containing MIDI files')
    parser.add_argument('output_dir', help='Directory to save the processed dataset')
    
    args = parser.parse_args()
    process_midi_directory(args.input_dir, args.output_dir) 