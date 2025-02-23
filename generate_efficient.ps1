# Attiva il virtual environment
.\venv\Scripts\Activate.ps1

# Hyperparametri configurabili con valori di default
$MODEL_PATH = if ($env:MODEL_PATH) { $env:MODEL_PATH } else { "checkpoint.pt" }
$OUTPUT = if ($env:OUTPUT) { $env:OUTPUT } else { "output/generated_sequence.txt" }
$NUM_STEPS = if ($env:NUM_STEPS) { $env:NUM_STEPS } else { "256" }
$TEMPERATURE = if ($env:TEMPERATURE) { $env:TEMPERATURE } else { "0.8" }
$EMBEDDING_DIM = if ($env:EMBEDDING_DIM) { $env:EMBEDDING_DIM } else { "128" }
$HIDDEN_SIZE = if ($env:HIDDEN_SIZE) { $env:HIDDEN_SIZE } else { "256" }
$FORCE_CPU = if ($env:FORCE_CPU) { $env:FORCE_CPU } else { "false" }
$SEED_FILE = if ($env:SEED_FILE) { $env:SEED_FILE } else { "output/generated_sequence.txt" }
$SEED_LINE = if ($env:SEED_LINE) { $env:SEED_LINE } else { "1" }

# Stampa la configurazione
Write-Host "Configurazione:"
Write-Host "MODEL_PATH: $MODEL_PATH"
Write-Host "OUTPUT: $OUTPUT"
Write-Host "NUM_STEPS: $NUM_STEPS"
Write-Host "TEMPERATURE: $TEMPERATURE"
Write-Host "EMBEDDING_DIM: $EMBEDDING_DIM"
Write-Host "HIDDEN_SIZE: $HIDDEN_SIZE"
Write-Host "FORCE_CPU: $FORCE_CPU"
Write-Host "SEED_FILE: $SEED_FILE"
Write-Host "SEED_LINE: $SEED_LINE"
Write-Host ""

# Costruisci il comando
$CMD = "python generate_efficient.py " + `
    "--model-path `"$MODEL_PATH`" " + `
    "--output `"$OUTPUT`" " + `
    "--num-steps $NUM_STEPS " + `
    "--temperature $TEMPERATURE " + `
    "--embedding-dim $EMBEDDING_DIM " + `
    "--hidden-size $HIDDEN_SIZE" + `
    " --seed-file `"$SEED_FILE`"" + `
    " --seed-line $SEED_LINE"

# Aggiungi opzioni condizionali
if ($FORCE_CPU -eq "true") {
    $CMD = "$CMD --force-cpu"
}

# Esegui il comando
Write-Host "Esecuzione comando:"
Write-Host $CMD
Write-Host ""
Invoke-Expression $CMD

# Se la generazione è riuscita, converti in MIDI e audio
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Conversione in MIDI..."
    $midiFile = $OUTPUT -replace '\.[^.]+$', '.mid'
    $wavFile = $OUTPUT -replace '\.[^.]+$', '.wav'
    $mp3File = $OUTPUT -replace '\.[^.]+$', '.mp3'
    python src/data_processing/sequence_to_midi.py --input $OUTPUT --output $midiFile
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "File MIDI salvato in: $midiFile"
        Write-Host ""
        
        # Scarica la soundfont se non esiste
        if (-not (Test-Path "output/default.sf2")) {
            Write-Host "Download soundfont..."
            Invoke-WebRequest -Uri "https://github.com/gleitz/midi-js-soundfonts/raw/gh-pages/FluidR3_GM/FluidR3_GM.sf2" -OutFile "output/default.sf2"
        }
        
        Write-Host "Conversione in WAV usando FluidSynth..."
        fluidsynth -ni "output/default.sf2" $midiFile -F $wavFile -r 44100
        
        Write-Host "Conversione in MP3..."
        C:\ffmpeg\bin\ffmpeg -y -i $wavFile -acodec libmp3lame -q:a 0 -ab 320k $mp3File
        Remove-Item $wavFile
        if ($LASTEXITCODE -eq 0) {
            Write-Host "File MP3 salvato in: $mp3File"
        }
    }
} 