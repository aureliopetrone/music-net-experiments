# train.ps1

param (
    [switch]$ResetHyperparams,  # Opzione --reset-hyperparams come switch
    [switch]$Help  # Opzione per mostrare l'aiuto
)

# Mostra l'aiuto se richiesto
if ($Help) {
    Write-Host "Uso: .\train_efficient.ps1 [-ResetHyperparams] [-Help]"
    Write-Host ""
    Write-Host "Opzioni:"
    Write-Host "  -ResetHyperparams    Resetta tutti gli iperparametri ai valori predefiniti"
    Write-Host "  -Help                Mostra questo messaggio di aiuto"
    Write-Host ""
    Write-Host "Esempio:"
    Write-Host "  .\train_efficient.ps1 -ResetHyperparams"
    exit 0
}

# Hyperparametri configurabili con valori di default
if ($ResetHyperparams) {
    Write-Host "Resettando gli iperparametri ai valori predefiniti..."
    # Forza i valori predefiniti se -ResetHyperparams è specificato
    $env:DATASET = "output/music_dataset.pt"
    $env:NUM_EPOCHS = 200
    $env:BATCH_SIZE = 512
    $env:EMBEDDING_DIM = 128
    $env:HIDDEN_SIZE = 256
    $env:LEARNING_RATE = "0.001"  # Stringa con punto decimale
    $env:SEQUENCE_LENGTH = 128
    $env:FORCE_CPU = "false"
    $env:TIME_LIMIT_HOURS = 12
} else {
    # Usa variabili d'ambiente se presenti, altrimenti valori di default
    $env:DATASET = if ($env:DATASET) { $env:DATASET } else { "output/music_dataset.pt" }
    $env:NUM_EPOCHS = if ($env:NUM_EPOCHS) { $env:NUM_EPOCHS } else { 200 }
    $env:BATCH_SIZE = if ($env:BATCH_SIZE) { $env:BATCH_SIZE } else { 512 }
    $env:EMBEDDING_DIM = if ($env:EMBEDDING_DIM) { $env:EMBEDDING_DIM } else { 128 }
    $env:HIDDEN_SIZE = if ($env:HIDDEN_SIZE) { $env:HIDDEN_SIZE } else { 256 }
    $env:LEARNING_RATE = if ($env:LEARNING_RATE) { $env:LEARNING_RATE.ToString().Replace(",", ".") } else { "0.001" }
    $env:SEQUENCE_LENGTH = if ($env:SEQUENCE_LENGTH) { $env:SEQUENCE_LENGTH } else { 128 }
    $env:FORCE_CPU = if ($env:FORCE_CPU) { $env:FORCE_CPU } else { "false" }
    $env:TIME_LIMIT_HOURS = if ($env:TIME_LIMIT_HOURS) { $env:TIME_LIMIT_HOURS } else { 12 }
}

# Controlla se esiste l'ultimo checkpoint
$LAST_CHECKPOINT = "checkpoints/last/last_model.pt"
if (Test-Path $LAST_CHECKPOINT) {
    Write-Host "Trovato ultimo checkpoint: $LAST_CHECKPOINT"
    $CHECKPOINT = $LAST_CHECKPOINT
} else {
    Write-Host "Nessun checkpoint trovato, partendo da zero"
    $CHECKPOINT = ""
}

# Stampa la configurazione
Write-Host "Configurazione:"
Write-Host "DATASET: $env:DATASET"
Write-Host "NUM_EPOCHS: $env:NUM_EPOCHS"
Write-Host "BATCH_SIZE: $env:BATCH_SIZE"
Write-Host "EMBEDDING_DIM: $env:EMBEDDING_DIM"
Write-Host "HIDDEN_SIZE: $env:HIDDEN_SIZE"
Write-Host "LEARNING_RATE: $env:LEARNING_RATE"
Write-Host "SEQUENCE_LENGTH: $env:SEQUENCE_LENGTH"
Write-Host "TIME_LIMIT_HOURS: $env:TIME_LIMIT_HOURS"
Write-Host "CHECKPOINT: $CHECKPOINT"
Write-Host "FORCE_CPU: $env:FORCE_CPU"
if ($ResetHyperparams) {
    Write-Host "Nota: Hyperparametri resettati ai valori predefiniti con -ResetHyperparams"
}
Write-Host ""

# Costruisci il comando
$CMD = "python train_efficient.py " +
    "--dataset $env:DATASET " +
    "--num-epochs $env:NUM_EPOCHS " +
    "--batch-size $env:BATCH_SIZE " +
    "--embedding-dim $env:EMBEDDING_DIM " +
    "--hidden-size $env:HIDDEN_SIZE " +
    "--learning-rate $env:LEARNING_RATE " +  # Passa il valore come stringa con punto
    "--sequence-length $env:SEQUENCE_LENGTH " +
    "--time-limit-hours $env:TIME_LIMIT_HOURS"

# Aggiungi opzioni condizionali
if ($env:FORCE_CPU -eq "true") {
    $CMD += " --force-cpu"
}

if ($CHECKPOINT) {
    $CMD += " --checkpoint $CHECKPOINT"
}

# Esegui il comando
Write-Host "Esecuzione comando:"
Write-Host $CMD
Write-Host ""
Invoke-Expression $CMD