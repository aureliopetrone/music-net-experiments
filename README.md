# Music Generation with Neural Networks

This project implements a neural network-based music generation system using PyTorch. It can process MIDI files, train a model on them, and generate new musical pieces.

## Project Structure

```
.
├── src/
│   ├── data_processing/    # Data preparation and processing scripts
│   ├── model/             # Neural network model definitions
│   ├── utils/             # Utility functions and helpers
│   └── visualization/     # Music visualization and playback tools
├── data/                  # Directory for training data
├── checkpoints/           # Model checkpoints
├── generated/             # Generated music output
└── output/               # Other output files
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation
1. Place your MIDI files in the `data` directory
2. Run the data preparation script:
```bash
python src/data_processing/prepare_dataset.py
```

### Training
To train the model:
```bash
python src/train.py
```

### Generation
To generate new music:
```bash
python src/generate_music.py
```

### Visualization
To visualize the music:
```bash
python src/visualization/visualize_music.py
```

## Dependencies
- PyTorch: Deep learning framework
- NumPy: Numerical computations
- pretty_midi: MIDI file processing
- pygame: Audio playback

## License
MIT License 