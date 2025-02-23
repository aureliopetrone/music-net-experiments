# Standard library imports
import os
from pathlib import Path

# Third-party imports
import torch
import pretty_midi

# Local imports
from src.model import MusicTokenizer

class DatasetToMidiConverter:
    def __init__(self, channels=4, time_step=0.25, max_vocab_size=128):
        self.tokenizer = MusicTokenizer(max_vocab_size=max_vocab_size)
        self.channels = channels
        self.time_step = time_step
        self.id_to_note = {v: k for k, v in self.tokenizer.note_to_id.items()}  # Reverse mapping
        
    def _pitch_name_to_midi_number(self, pitch_name):
        """Convert pitch name (e.g., 'C4' or 'C4#') to MIDI note number"""
        if pitch_name == 'O':  # Rest
            return None
            
        # Handle sharp notes (e.g., 'C4#')
        if '#' in pitch_name:
            note = pitch_name[0]
            octave = int(pitch_name[2:-1])
            sharp = True
        else:
            note = pitch_name[0]
            octave = int(pitch_name[1:])
            sharp = False
            
        NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        note_index = NOTES.index(note)
        if sharp:
            note_index += 1
        return (octave + 1) * 12 + note_index
    
    def convert_to_midi(self, tokens, output_path):
        """Convert tokenized tensor back to MIDI file"""
        try:
            # Create PrettyMIDI object
            midi = pretty_midi.PrettyMIDI()
            instrument = pretty_midi.Instrument(program=0)  # Piano by default
            
            # Convert tokens to notes
            tokens = tokens.numpy() if torch.is_tensor(tokens) else tokens
            
            # Process each timestep
            for timestep, chord_tokens in enumerate(tokens):
                start_time = timestep * self.time_step
                
                # Convert token IDs to pitch names
                chord = [self.id_to_note[token] for token in chord_tokens]
                
                # Process each note in the chord
                for pitch_name in chord:
                    midi_number = self._pitch_name_to_midi_number(pitch_name)
                    if midi_number is not None:  # Skip rests ('O')
                        note = pretty_midi.Note(
                            velocity=100,  # Default velocity
                            pitch=midi_number,
                            start=start_time,
                            end=start_time + self.time_step
                        )
                        instrument.notes.append(note)
            
            # Add instrument to MIDI file
            midi.instruments.append(instrument)
            
            # Save MIDI file
            midi.write(output_path)
            print(f"MIDI file saved to {output_path}")
            return True
            
        except Exception as e:
            print(f"Error converting to MIDI: {str(e)}")
            return False

def process_dataset_to_midi(dataset_path, output_dir):
    """Process a dataset file and convert it to MIDI"""
    converter = DatasetToMidiConverter()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the dataset
    try:
        dataset = torch.load(dataset_path)
        
        # Generate output filename
        dataset_name = Path(dataset_path).stem
        output_path = os.path.join(output_dir, f"{dataset_name}_converted.mid")
        
        # Convert to MIDI
        success = converter.convert_to_midi(dataset, output_path)
        if success:
            print(f"Successfully processed {dataset_path}")
        else:
            print(f"Failed to process {dataset_path}")
            
    except Exception as e:
        print(f"Error loading dataset {dataset_path}: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert tokenized dataset to MIDI files')
    parser.add_argument('dataset_path', help='Path to the .pt dataset file')
    parser.add_argument('output_dir', help='Directory to save the MIDI files')
    
    args = parser.parse_args()
    process_dataset_to_midi(args.dataset_path, args.output_dir)