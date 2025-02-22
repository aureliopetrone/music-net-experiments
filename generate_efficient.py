import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from src.model.music_net import EfficientHarmonicMusicNet
from src.model.tokenizer import MusicTokenizer

def sample_from_logits(logits, temperature=1.0):
    if temperature == 0:
        return torch.argmax(logits, dim=-1)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

def generate_music(model, tokenizer, device, seed_sequence=None, num_steps=64, temperature=0.8, sequence_length=32, show_progress=True):
    model.eval()
    
    if seed_sequence is None:
        notes = ['C4', 'E4', 'G4', 'C5']
        indices = [tokenizer.note_to_id.get(note, 0) for note in notes]
        seed_sequence = torch.tensor([indices], dtype=torch.long, device=device).unsqueeze(1)
    
    generated_sequence = seed_sequence.clone()
    
    with torch.no_grad():
        for step in range(num_steps):
            if show_progress:
                print(f'Generating step {step + 1}/{num_steps}', end='\r')
            
            input_sequence = generated_sequence[:, -sequence_length:, :] if generated_sequence.size(1) > sequence_length else generated_sequence
            output = model(input_sequence)
            last_output = output[:, -1, :, :]
            
            new_notes = torch.zeros((1, 1, 4), dtype=torch.long, device=device)
            for channel in range(4):
                channel_logits = last_output[0, channel]
                new_notes[0, 0, channel] = sample_from_logits(channel_logits, temperature)
            
            generated_sequence = torch.cat([generated_sequence, new_notes], dim=1)
    
    if show_progress:
        print('\nGeneration completed!')
    
    notes_sequence = []
    for step in range(generated_sequence.size(1)):
        step_notes = [tokenizer.id_to_note[generated_sequence[0, step, channel].item()] for channel in range(4)]
        notes_sequence.append(step_notes)
    
    return notes_sequence

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print(f"Using {device}")

    tokenizer = MusicTokenizer(max_vocab_size=128)
    vocab_size = len(tokenizer.note_to_id)
    print(f"Using vocabulary size: {vocab_size}")

    model = EfficientHarmonicMusicNet(
        num_notes=vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        dropout=0.0
    ).to(device)

    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    generated_sequence = generate_music(
        model,
        tokenizer,
        device,
        num_steps=args.num_steps,
        temperature=args.temperature,
        sequence_length=32  # Match training
    )
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for step_notes in generated_sequence:
            f.write(','.join(step_notes) + '\n')
    
    print(f'\nSequence saved to {output_path}')
    print(f'Sequence length: {len(generated_sequence)} steps')
    print(f'First 5 steps example:')
    for i, step_notes in enumerate(generated_sequence[:5]):
        print(f'Step {i + 1}: {step_notes}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate music with the trained model')
    parser.add_argument('--model-path', type=str, default='model.pt')
    parser.add_argument('--output', type=str, default='output/generated_sequence.txt')
    parser.add_argument('--num-steps', type=int, default=64)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--embedding-dim', type=int, default=32)
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--force-cpu', action='store_true')
    
    args = parser.parse_args()
    main(args)