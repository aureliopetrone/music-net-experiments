"""Model package initialization.

This package contains the neural network models for music generation.
"""

import torch
import torch.nn as nn

from .music_net import EfficientHarmonicMusicNet
from .tokenizer import MusicTokenizer

__all__ = ['EfficientHarmonicMusicNet', 'MusicTokenizer']
