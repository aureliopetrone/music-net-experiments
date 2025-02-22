# Efficient Harmonic Music Generator

A deep learning model for generating harmonic music sequences using an efficient architecture. This project uses PyTorch to train a neural network on MIDI files to generate new musical pieces while maintaining harmonic consistency.

## Features

- Efficient neural network architecture optimized for music generation
- MIDI file processing and tokenization
- Training with support for MPS (Apple Silicon) acceleration
- Checkpoint saving and loading for interrupted training
- Configurable model parameters

## Requirements

- Python 3.8+
- PyTorch
- Additional dependencies in `requirements.txt`

## Installation

```bash
# Clone the repository
git clone https://github.com/[your-username]/music-net-experiments.git
cd music-net-experiments

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train_efficient.py --dataset path/to/dataset \
                         --batch-size 16 \
                         --sequence-length 4 \
                         --embedding-dim 16 \
                         --hidden-size 32 \
                         --learning-rate 0.1 \
                         --num-epochs 1000
```

### Generation

```bash
python generate_efficient.py --model-path path/to/model.pt \
                           --output output.mid \
                           --length 1000
```

## Project Structure

- `src/`: Source code
  - `model/`: Neural network architecture
  - `data_processing/`: MIDI processing and dataset preparation
- `train_efficient.py`: Training script
- `generate_efficient.py`: Generation script
- `requirements.txt`: Python dependencies

## License

MIT License 