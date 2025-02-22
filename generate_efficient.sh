#!/bin/bash

# Controlla se timidity è installato
if ! command -v timidity &> /dev/null; then
    echo "timidity non trovato. Installazione in corso..."
    brew install timidity
fi

# Controlla se ffmpeg è installato
if ! command -v ffmpeg &> /dev/null; then
    echo "ffmpeg non trovato. Installazione in corso..."
    brew install ffmpeg
fi

# Hyperparametri configurabili con valori di default
MODEL_PATH=${MODEL_PATH:-"checkpoints/best/best_model.pt"}
OUTPUT=${OUTPUT:-"output/generated_sequence.txt"}
NUM_STEPS=${NUM_STEPS:-256}
TEMPERATURE=${TEMPERATURE:-0.8}
EMBEDDING_DIM=${EMBEDDING_DIM:-32}
HIDDEN_SIZE=${HIDDEN_SIZE:-64}
FORCE_CPU=${FORCE_CPU:-false}

# Stampa la configurazione
echo "Configurazione:"
echo "MODEL_PATH: $MODEL_PATH"
echo "OUTPUT: $OUTPUT"
echo "NUM_STEPS: $NUM_STEPS"
echo "TEMPERATURE: $TEMPERATURE"
echo "EMBEDDING_DIM: $EMBEDDING_DIM"
echo "HIDDEN_SIZE: $HIDDEN_SIZE"
echo "FORCE_CPU: $FORCE_CPU"
echo

# Costruisci il comando
CMD="python3 generate_efficient.py \
    --model-path $MODEL_PATH \
    --output $OUTPUT \
    --num-steps $NUM_STEPS \
    --temperature $TEMPERATURE \
    --embedding-dim $EMBEDDING_DIM \
    --hidden-size $HIDDEN_SIZE"

# Aggiungi opzioni condizionali
if [ "$FORCE_CPU" = true ]; then
    CMD="$CMD --force-cpu"
fi

# Esegui il comando
echo "Esecuzione comando:"
echo "$CMD"
echo
eval "$CMD"

# Se la generazione è riuscita, converti in MIDI e OGG
if [ $? -eq 0 ]; then
    echo
    echo "Conversione in MIDI..."
    python3 src/data_processing/sequence_to_midi.py \
        --input $OUTPUT \
        --output "${OUTPUT%.*}.mid"
    
    if [ $? -eq 0 ]; then
        echo
        echo "Conversione in OGG..."
        timidity "${OUTPUT%.*}.mid" -Ov -o "${OUTPUT%.*}.ogg"
        echo "File OGG salvato in: ${OUTPUT%.*}.ogg"
        
        echo
        echo "Conversione in MP3..."
        ffmpeg -i "${OUTPUT%.*}.ogg" -codec:a libmp3lame -qscale:a 2 "${OUTPUT%.*}.mp3" -y
        echo "File MP3 salvato in: ${OUTPUT%.*}.mp3"
    fi
fi 