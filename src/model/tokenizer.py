import ast


class MusicTokenizer:
    """Tokenizer per sequenze di note musicali.

    Questo tokenizer mappa i nomi delle note a token numerici e viceversa.
    Il token 0 è riservato alla O ('O').
    """
    def __init__(self, vocab=None, vocab_file='vocab.txt', max_vocab_size=None):
        if vocab is None:
            self.note_to_id = self._load_vocab(vocab_file, max_vocab_size)
        else:
            self.note_to_id = vocab
        # Crea la mappa inversa per il decoding
        self.id_to_note = {v: k for k, v in self.note_to_id.items()}
        
    def _load_vocab(self, vocab_file, max_vocab_size=None):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        # Se il file inizia con 'vocab', proviamo a interpretarlo come una espressione Python
        if content.startswith("vocab"):
            try:
                _, data = content.split('=', 1)
                data = data.strip()
                vocab_data = ast.literal_eval(data)
            except Exception as e:
                raise ValueError("Errore nel parsing del vocabolario dal file: " + str(e))
            if isinstance(vocab_data, (list, tuple)):
                tokens = list(vocab_data)
            elif isinstance(vocab_data, set):
                tokens = sorted(list(vocab_data), key=lambda x: x)  # ordinamento alfabetico come fallback
            elif isinstance(vocab_data, dict):
                return vocab_data
            else:
                raise ValueError("Formato vocabolario non riconosciuto")
        else:
            # Assume che ogni linea contenga un token
            tokens = [line.strip() for line in content.splitlines() if line.strip() != '']
        # Assicuriamoci che 'O' sia al primo posto (indice 0)
        if 'O' in tokens:
            tokens.remove('O')
            tokens = ['O'] + tokens
        else:
            tokens = ['O'] + tokens
        # Limit vocabulary size if specified
        if max_vocab_size is not None:
            tokens = tokens[:max_vocab_size]
        note_to_id = {note: idx for idx, note in enumerate(tokens)}
        return note_to_id
        
    def encode(self, note_sequence):
        """Codifica una sequenza di note (lista di stringhe) in un elenco di token interi."""
        return [self.note_to_id.get(note, 0) for note in note_sequence]
    
    def decode(self, token_sequence):
        """Decodifica una sequenza di token (lista di interi) in una sequenza di note (stringhe)."""
        return [self.id_to_note.get(token, 'O') for token in token_sequence]


if __name__ == '__main__':
    # Esempio di utilizzo
    tokenizer = MusicTokenizer()
    sample_notes = ['C1', 'E1', 'G1', 'O', 'F7']
    tokens = tokenizer.encode(sample_notes)
    print('Note:', sample_notes)
    print('Tokenizzati:', tokens)
    print('Decodificati:', tokenizer.decode(tokens)) 