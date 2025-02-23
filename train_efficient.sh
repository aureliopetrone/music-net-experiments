#!/bin/bash

# Hyperparametri configurabili con valori ottimizzati
DATASET=${DATASET:-"output/music_dataset.pt"}
NUM_EPOCHS=${NUM_EPOCHS:-200}  # Ridotto con early stopping consigliato
BATCH_SIZE=${BATCH_SIZE:-128}  # Aumentato per sfruttare la GPU
EMBEDDING_DIM=${EMBEDDING_DIM:-64}  # Compromesso tra capacità e leggerezza
HIDDEN_SIZE=${HIDDEN_SIZE:-128}  # Aumentato per maggior capacità
LEARNING_RATE=${LEARNING_RATE:-0.001}  # Valore stabile e comune per LSTM
SEQUENCE_LENGTH=${SEQUENCE_LENGTH:-64}  # Per dipendenze più lunghe
FORCE_CPU=${FORCE_CPU:-false}
TIME_LIMIT_HOURS=${TIME_LIMIT_HOURS:-12}  # Ridotto, sufficiente con GPU

# Controlla se esiste l'ultimo checkpoint
LAST_CHECKPOINT="checkpoints/last/last_model.pt"
if [ -f "$LAST_CHECKPOINT" ]; then
    echo "Trovato ultimo checkpoint: $LAST_CHECKPOINT"
    CHECKPOINT=$LAST_CHECKPOINT
else
    echo "Nessun checkpoint trovato, partendo da zero"
    CHECKPOINT=""
fi

# Stampa la configurazione
echo "Configurazione:"
echo "DATASET: $DATASET"
echo "NUM_EPOCHS: $NUM_EPOCHS"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "EMBEDDING_DIM: $EMBEDDING_DIM"
echo "HIDDEN_SIZE: $HIDDEN_SIZE"
echo "LEARNING_RATE: $LEARNING_RATE"
echo "SEQUENCE_LENGTH: $SEQUENCE_LENGTH"
echo "TIME_LIMIT_HOURS: $TIME_LIMIT_HOURS"
echo "CHECKPOINT: $CHECKPOINT"
echo "FORCE_CPU: $FORCE_CPU"
echo

# Costruisci il comando
CMD="python train_efficient.py \
    --dataset $DATASET \
    --num-epochs $NUM_EPOCHS \
    --batch-size $BATCH_SIZE \
    --embedding-dim $EMBEDDING_DIM \
    --hidden-size $HIDDEN_SIZE \
    --learning-rate $LEARNING_RATE \
    --sequence-length $SEQUENCE_LENGTH \
    --time-limit-hours $TIME_LIMIT_HOURS"

# Aggiungi opzioni condizionali
if [ "$FORCE_CPU" = true ]; then
    CMD="$CMD --force-cpu"
fi

if [ ! -z "$CHECKPOINT" ]; then
    CMD="$CMD --checkpoint $CHECKPOINT"
fi

# Esegui il comando
echo "Esecuzione comando:"
echo "$CMD"
echo
eval "$CMD"