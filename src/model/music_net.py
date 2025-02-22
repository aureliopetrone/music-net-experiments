# Third-party imports
import torch
import torch.nn as nn
import math

class EfficientHarmonicMusicNet(nn.Module):
    """
    Simplified model for harmonic music generation.
    """
    def __init__(self, num_notes, embedding_dim=16, hidden_size=32, dropout=0.2):
        super().__init__()
        self.num_notes = num_notes
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        
        # Embedding layers for each channel
        self.embedding1 = nn.Embedding(num_notes, embedding_dim)
        self.embedding2 = nn.Embedding(num_notes, embedding_dim)
        self.embedding3 = nn.Embedding(num_notes, embedding_dim)
        self.embedding4 = nn.Embedding(num_notes, embedding_dim)
        
        # Single LSTM layer
        self.lstm = nn.LSTM(
            4 * embedding_dim,
            hidden_size,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # Output layer
        self.output = nn.Linear(2 * hidden_size, 4 * num_notes)

    def forward(self, x):
        # Handle single batch case
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            
        batch_size, seq_length, _ = x.shape
        
        # Split input into separate channels
        x1, x2, x3, x4 = x.split(1, dim=2)
        
        # Get embeddings for each channel
        embed1 = self.embedding1(x1.squeeze(-1))
        embed2 = self.embedding2(x2.squeeze(-1))
        embed3 = self.embedding3(x3.squeeze(-1))
        embed4 = self.embedding4(x4.squeeze(-1))
        
        # Concatenate embeddings
        concatenated = torch.cat((embed1, embed2, embed3, embed4), dim=2)
        
        # LSTM layer
        lstm_out, _ = self.lstm(concatenated)
        
        # Output layer
        logits = self.output(lstm_out)
        
        # Reshape output for each channel
        logits = logits.view(batch_size, seq_length, 4, self.num_notes)
        
        return logits

    def get_complexity(self):
        """
        Calculate model complexity in terms of parameters and memory usage.
        """
        # Embedding layers
        emb_params = 4 * (self.num_notes * self.embedding_dim)
        emb_memory = emb_params * 4
        
        # LSTM parameters
        input_size = 4 * self.embedding_dim
        lstm_params = 4 * (input_size * self.hidden_size + self.hidden_size * self.hidden_size)
        lstm_params += 4 * self.hidden_size * 2  # bias parameters
        lstm_memory = lstm_params * 4
        
        # Output layer
        output_params = self.hidden_size * (4 * self.num_notes)
        output_memory = output_params * 4
        
        total_params = emb_params + lstm_params + output_params
        total_memory = emb_memory + lstm_memory + output_memory
        
        return {
            'total_parameters': total_params,
            'total_memory_mb': total_memory / (1024 * 1024)  # Convert to MB
        }