# Music-Net Experiments

Progetto di generazione musicale basato su reti neurali utilizzando PyTorch con supporto CUDA.

## Requisiti

- Python 3.8+
- NVIDIA GPU con supporto CUDA
- CUDA Toolkit 11.8
- Driver NVIDIA aggiornati
- timidity (per la conversione audio)
- ffmpeg (per la conversione audio)

## Installazione

1. Clona il repository:
```bash
git clone https://github.com/tuousername/music-net-experiments.git
cd music-net-experiments
```

2. Installa le dipendenze:
```bash
pip install -r requirements.txt
```

3. Rendi eseguibili gli script:
```bash
chmod +x train_efficient.sh generate_efficient.sh
```

## Training del Modello

### Preparazione Dataset
Prima di iniziare il training, prepara il dataset convertendo i file MIDI:

```bash
python src/data_processing/midi_to_dataset.py data output
```

### Avvio Training
Per avviare il training del modello, usa lo script `train_efficient.sh`:

```bash
# Configurazione base
./train_efficient.sh

# Configurazione personalizzata
DATASET="output/music_dataset.pt" \
NUM_EPOCHS=10000 \
BATCH_SIZE=64 \
EMBEDDING_DIM=32 \
HIDDEN_SIZE=64 \
LEARNING_RATE=1.0 \
SEQUENCE_LENGTH=32 \
TIME_LIMIT_HOURS=24 \
./train_efficient.sh
```

Parametri configurabili:
- `DATASET`: percorso al dataset (default: "output/music_dataset.pt")
- `NUM_EPOCHS`: numero di epoche (default: 10000)
- `BATCH_SIZE`: dimensione del batch (default: 64)
- `EMBEDDING_DIM`: dimensione dell'embedding (default: 32)
- `HIDDEN_SIZE`: dimensione hidden layer LSTM (default: 64)
- `LEARNING_RATE`: learning rate (default: 1.0)
- `SEQUENCE_LENGTH`: lunghezza delle sequenze (default: 32)
- `TIME_LIMIT_HOURS`: limite di tempo in ore (default: 24)
- `FORCE_CPU`: forza l'uso della CPU anche se CUDA è disponibile (default: false)

Lo script cercherà automaticamente l'ultimo checkpoint in `checkpoints/last/last_model.pt` per riprendere il training.

## Generazione Musica

Per generare nuova musica usando il modello addestrato, usa lo script `generate_efficient.sh`:

```bash
# Configurazione base
./generate_efficient.sh

# Configurazione personalizzata
MODEL_PATH="checkpoints/best/best_model.pt" \
OUTPUT="output/generated_sequence.txt" \
NUM_STEPS=256 \
TEMPERATURE=0.8 \
EMBEDDING_DIM=32 \
HIDDEN_SIZE=64 \
./generate_efficient.sh
```

Parametri configurabili:
- `MODEL_PATH`: percorso del modello addestrato (default: "checkpoints/best/best_model.pt")
- `OUTPUT`: percorso output file generato (default: "output/generated_sequence.txt")
- `NUM_STEPS`: numero di step di generazione (default: 256)
- `TEMPERATURE`: temperatura di sampling (default: 0.8)
- `EMBEDDING_DIM`: dimensione embedding (default: 32)
- `HIDDEN_SIZE`: dimensione hidden layer LSTM (default: 64)
- `FORCE_CPU`: forza l'uso della CPU anche se CUDA è disponibile (default: false)

Lo script genererà automaticamente:
1. La sequenza di note in formato testuale (`OUTPUT`)
2. Il file MIDI corrispondente (`OUTPUT.mid`)
3. File audio in formato OGG (`OUTPUT.ogg`)
4. File audio in formato MP3 (`OUTPUT.mp3`)

## Monitoraggio Training

Durante il training, puoi monitorare:
- Loss di training e validazione
- Learning rate
- Velocità di training (samples/sec)
- Utilizzo memoria GPU

I checkpoint vengono salvati automaticamente quando la loss di validazione migliora.

## Note

- Il training su GPU NVIDIA con CUDA è significativamente più veloce rispetto alla CPU
- La generazione può essere eseguita anche su CPU per testing rapido
- Usa temperature più basse (0.5-0.7) per output più "sicuri" e più alte (0.8-1.0) per output più creativi
- Gli script verificheranno automaticamente la presenza di timidity e ffmpeg necessari per la conversione audio
