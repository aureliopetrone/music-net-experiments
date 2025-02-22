# Third-party imports
import torch

def create_c_major_dataset():
    """
    Create a test dataset with a simple four-chord progression: C -> F -> G -> C
    Each chord is represented in 4-part harmony
    """
    # Define the chords in 4-part harmony
    chords = {
        'C1': [60, 64, 67, 72],  # C4, E4, G4, C5
        'F': [53, 57, 60, 65],   # F3, A3, C4, F4
        'G': [55, 59, 62, 67],   # G3, B3, D4, G4
        'C2': [60, 64, 67, 72],  # C4, E4, G4, C5 (final C)
    }
    
    # Create a sequence of 200 timesteps with the progression C -> F -> G -> C
    sequence = []
    for _ in range(50):  # 50 repetitions of the 4-chord progression
        sequence.append(chords['C1'])
        sequence.append(chords['F'])
        sequence.append(chords['G'])
        sequence.append(chords['C2'])
    
    # Convert to tensor
    data = torch.tensor(sequence, dtype=torch.long)
    
    # Save dataset
    print(f"Creating dataset with {len(data)} timesteps")
    print("Chord progression: C -> F -> G -> C (repeated)")
    torch.save(data, 'output/test_dataset.pt')

if __name__ == '__main__':
    create_c_major_dataset() 