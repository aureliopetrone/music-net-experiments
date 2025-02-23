# Crea la directory per i soundfonts se non esiste
$soundfontDir = "C:\ProgramData\soundfonts"
if (-not (Test-Path $soundfontDir)) {
    New-Item -ItemType Directory -Path $soundfontDir -Force
}

# URL del soundfont (usando un link diretto a un soundfont comune)
$soundfontUrl = "https://raw.githubusercontent.com/gleitz/midi-js-soundfonts/gh-pages/MusyngKite/acoustic_grand_piano-mp3.sf2"
$soundfontPath = Join-Path $soundfontDir "default.sf2"

# Scarica il soundfont
Write-Host "Downloading soundfont..."
try {
    Invoke-WebRequest -Uri $soundfontUrl -OutFile $soundfontPath -UseBasicParsing
    Write-Host "Soundfont downloaded successfully to $soundfontPath"
} catch {
    Write-Host "Error downloading soundfont: $_"
    exit 1
}

# Verifica che il file esista
if (Test-Path $soundfontPath) {
    Write-Host "Soundfont setup completed successfully!"
} else {
    Write-Host "Error: Soundfont file not found after download"
    exit 1
} 