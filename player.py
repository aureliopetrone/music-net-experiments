import numpy as np
import pygame
from pygame import mixer
import time
from mido import MidiFile, MidiTrack, Message
import tempfile
import os
from src.model.tokenizer import MusicTokenizer


class AudioPlayer:
    def __init__(self, tokenizer, bpm=120):
        self.tokenizer = tokenizer
        self.bpm = bpm
        self.is_playing = False
        pygame.mixer.init()
        self.channel = pygame.mixer.Channel(0)
    
    def note_to_midi(self, note_id):
        if note_id == 0:
            return None
        return 36 + (note_id - 1)

    def tensor_to_midi(self, tensor):
        assert tensor.shape[1] == 4, "Tensor must have shape (X, 4)"
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)
        ticks_per_beat = mid.ticks_per_beat
        seconds_per_beat = 60 / self.bpm
        ticks_per_second = ticks_per_beat / seconds_per_beat
        
        for step in tensor:
            pitch_id, duration_id, volume_id, _ = step
            midi_note = self.note_to_midi(pitch_id)
            duration = (duration_id + 1) / 4
            velocity = min(int(volume_id * 12.7), 127)
            duration_ticks = int(duration * ticks_per_beat)
            
            if midi_note is not None:
                track.append(Message('note_on', note=midi_note, velocity=velocity, time=0))
                track.append(Message('note_off', note=midi_note, velocity=0, time=duration_ticks))
            else:
                track.append(Message('note_off', note=36, velocity=0, time=duration_ticks))
        
        return mid

    def play_tensor(self, tensor):
        """Riproduce il tensore come MIDI con un'interfaccia interattiva."""
        mid = self.tensor_to_midi(tensor)
        
        # Crea un file temporaneo manualmente
        temp_file = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
        temp_file_name = temp_file.name
        mid.save(temp_file_name)
        temp_file.close()  # Chiudi il file esplicitamente
        
        # Verifica che il file esista
        if not os.path.exists(temp_file_name):
            raise FileNotFoundError(f"Could not find MIDI file: {temp_file_name}")
        
        try:
            pygame.mixer.music.load(temp_file_name)
            self.is_playing = True
            pygame.mixer.music.play()
            
            # Pygame interactive loop
            pygame.init()
            screen = pygame.display.set_mode((400, 200))
            pygame.display.set_caption("MIDI Player")
            font = pygame.font.Font(None, 36)
            
            while self.is_playing:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.stop()
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            if pygame.mixer.music.get_busy():
                                pygame.mixer.music.pause()
                            else:
                                pygame.mixer.music.unpause()
                        elif event.key == pygame.K_s:
                            self.stop()
                
                screen.fill((255, 255, 255))
                status = "Playing" if pygame.mixer.music.get_busy() else "Paused"
                text = font.render(f"Status: {status} (Space: Pause/Play, S: Stop)", True, (0, 0, 0))
                screen.blit(text, (10, 80))
                pygame.display.flip()
        finally:
            # Pulizia
            pygame.mixer.music.stop()
            pygame.quit()
            if os.path.exists(temp_file_name):
                os.remove(temp_file_name)

    def stop(self):
        pygame.mixer.music.stop()
        self.is_playing = False

if __name__ == "__main__":
    vocab = {'O': 0, 'C1': 1, 'D1': 2, 'E1': 3, 'F1': 4, 'G1': 5}
    tokenizer = MusicTokenizer(vocab=vocab)
    
    tensor = np.array([
        [1, 1, 5, 0],  # C1, half note, medium volume
        [0, 0, 0, 0],  # Pause, quarter note
        [3, 2, 8, 0],  # E1, three-quarter note, loud
        [5, 1, 3, 0],  # G1, half note, soft
    ])
    
    player = AudioPlayer(tokenizer)
    player.play_tensor(tensor)