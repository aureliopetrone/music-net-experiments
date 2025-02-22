# EfficientHarmonicMusicNet: Un Modello per la Generazione di Musica Polifonica

## Abstract

Questo documento descrive EfficientHarmonicMusicNet, un modello per la generazione di musica polifonica che utilizza embedding di note singole e meccanismi di attention per predire sequenze musicali su più canali paralleli.

## 1. Introduzione

La generazione automatica di musica richiede la modellazione di strutture temporali e relazioni armoniche. Questo lavoro presenta un approccio basato su reti neurali per la generazione di musica polifonica, focalizzandosi sull'efficienza computazionale e la qualità musicale.

## 2. Architettura del Sistema

### 2.1 Overview

Il sistema è composto da tre componenti principali:
1. Layer di embedding per note singole
2. Layer di self-attention multi-testa
3. Feed-forward network per la predizione

### 2.2 Rappresentazione dell'Input

- Sequenze di note MIDI tokenizzate
- 4 canali paralleli per timestep
- Note rappresentate come 'nota+ottava' (es. 'C4', 'G#5')
- Token speciale 'O' per il silenzio

### 2.3 Componenti Principali

#### Note Embedding
- Converte ogni nota in un vettore denso
- Dimensione dell'embedding: 32
- Supporta l'intero range MIDI (128 note)
- Memoria efficiente: O(num_notes) invece di O(num_notes^4)

#### Self-Attention Layer
- 4 teste di attention
- Cattura relazioni temporali e armoniche
- Layer normalization per stabilità
- Batch-first per efficienza computazionale

#### Feed-Forward Network
- Espansione intermedia (4x)
- Attivazione GELU
- Proiezione finale alla dimensione dell'embedding

## 3. Training

### 3.1 Dataset
- Brani MIDI polifonici preprocessati
- Tokenizzazione delle note
- Normalizzazione dei dati

### 3.2 Processo di Training
- Optimizer: AdamW
- Batch size: 32
- Early stopping
- Checkpoint automatici

## 4. Generazione

### 4.1 Processo Generativo
- Generazione autoregressiva nota per nota
- Predizione parallela su 4 canali
- Mantenimento della coerenza armonica

### 4.2 Post-processing
- Conversione in MIDI
- Gestione delle durate
- Normalizzazione delle velocità

## 5. Limitazioni e Sviluppi Futuri

- Estensione del numero di canali
- Miglioramento della coerenza a lungo termine
- Controlli stilistici
- Strutture musicali complesse

## 6. Riferimenti

- PyTorch documentation
- MIDI specification
- Music theory fundamentals 