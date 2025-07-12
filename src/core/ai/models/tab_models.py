"""
Guitar tablature generation models.

This module contains neural network models for generating guitar tablature
from audio input using machine learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class TabTranscriptionModel(nn.Module):
    """
    Neural network model for guitar tab transcription.
    
    This model takes mel-spectrograms from hexaphonic audio input (one channel per string)
    and predicts string/fret positions for tablature.
    
    Architecture:
    - CNN layers to process spectrograms
    - Bidirectional LSTM for sequence modeling
    - Two output heads:
      1. String activation (which strings are played)
      2. Fret positions (what fret is played on each string)
    """
    
    def __init__(
        self,
        input_channels: int = 6,
        mel_bins: int = 128,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        max_fret: int = 24
    ):
        """
        Initialize the Tab Transcription Model.
        
        Args:
            input_channels: Number of input channels (6 for hexaphonic guitar)
            mel_bins: Number of mel bins in the input spectrograms
            hidden_size: Size of hidden layers
            num_layers: Number of recurrent layers
            dropout: Dropout rate
            max_fret: Maximum fret number to predict
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.mel_bins = mel_bins
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_fret = max_fret
        
        # CNN Feature Extractor
        self.conv_layers = nn.Sequential(
            # First convolutional block
            nn.Conv2d(input_channels, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            
            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            
            # Fourth convolutional block
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        
        # Calculate the size of the flattened CNN output
        # Mel bins dimension is reduced by 2^4 = 16 due to max pooling
        cnn_output_height = mel_bins // 16
        self.cnn_output_size = 256 * cnn_output_height
        
        # Recurrent layers for sequence modeling
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        # Fully connected layers for classification
        lstm_output_size = hidden_size * 2  # bidirectional
        
        # String activation head (which strings are played)
        self.string_classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, input_channels),  # 6 outputs (one per string)
        )
        
        # Fret position head (what fret is played on each string)
        self.fret_classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, input_channels * (max_fret + 1))  # 6 strings × (max_fret + 1) outputs
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape [batch_size, channels, mel_bins, time_steps]
               where channels = 6 for hexaphonic guitar
               
        Returns:
            tuple: (string_preds, fret_preds) where:
                   string_preds: String activation predictions [batch_size, time_steps, 6]
                   fret_preds: Fret position predictions [batch_size, time_steps, 6, max_fret+1]
        """
        batch_size, channels, mel_bins, time_steps = x.size()
        
        # CNN feature extraction (process all time steps at once)
        x = self.conv_layers(x)  # [batch_size, 256, mel_bins/16, time_steps/16]
        
        # Reshape for sequence modeling
        # Swap time and feature dimensions to get [batch_size, time_steps/16, features]
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size, time_steps // 16, self.cnn_output_size)
        
        # LSTM sequence modeling
        x, _ = self.lstm(x)  # [batch_size, time_steps/16, hidden_size*2]
        
        # Classification heads
        string_preds = self.string_classifier(x)  # [batch_size, time_steps/16, 6]
        
        # Fret classification
        fret_logits = self.fret_classifier(x)  # [batch_size, time_steps/16, 6*(max_fret+1)]
        
        # Reshape fret predictions to [batch_size, time_steps/16, 6, max_fret+1]
        fret_preds = fret_logits.reshape(batch_size, time_steps // 16, 6, self.max_fret + 1)
        
        return string_preds, fret_preds
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make a prediction and convert logits to probabilities.
        
        Args:
            x: Input tensor of shape [batch_size, channels, mel_bins, time_steps]
            
        Returns:
            Dictionary with predictions:
                - string_probs: String activation probabilities [batch_size, time_steps, 6]
                - fret_probs: Fret position probabilities [batch_size, time_steps, 6, max_fret+1]
        """
        string_logits, fret_logits = self.forward(x)
        
        # Convert to probabilities
        string_probs = torch.sigmoid(string_logits)  # Binary classification per string
        fret_probs = F.softmax(fret_logits, dim=3)  # Multi-class classification per string
        
        return {
            'string_probs': string_probs,
            'fret_probs': fret_probs
        }
    
    def calculate_loss(
        self, 
        x: torch.Tensor, 
        string_targets: torch.Tensor, 
        fret_targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate loss for model training.
        
        Args:
            x: Input tensor of shape [batch_size, channels, mel_bins, time_steps]
            string_targets: String activation targets [batch_size, 6, time_steps]
            fret_targets: Fret position targets [batch_size, 6, time_steps]
            
        Returns:
            Dictionary with loss values:
                - string_loss: Binary cross-entropy loss for string activation
                - fret_loss: Cross-entropy loss for fret positions
                - total_loss: Weighted sum of string_loss and fret_loss
        """
        # Forward pass
        string_logits, fret_logits = self.forward(x)
        
        # Get actual dimensions
        batch_size, channels, time_steps = string_targets.size()
        _, pred_time_steps, _ = string_logits.size()
        
        # Downsample targets to match prediction time steps exactly
        string_targets_ds = F.interpolate(
            string_targets.float().unsqueeze(1),
            size=(channels, pred_time_steps),
            mode='nearest'
        ).squeeze(1)
        
        fret_targets_ds = F.interpolate(
            fret_targets.float().unsqueeze(1),
            size=(channels, pred_time_steps),
            mode='nearest'
        ).squeeze(1)
        
        # Rearrange dimensions to match prediction shape
        string_targets_ds = string_targets_ds.permute(0, 2, 1)  # [batch_size, pred_time_steps, 6]
        fret_targets_ds = fret_targets_ds.permute(0, 2, 1)      # [batch_size, pred_time_steps, 6]
        
        # Calculate string activation loss (binary cross-entropy)
        string_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        string_loss = string_loss_fn(string_logits, string_targets_ds)
        
        # Calculate fret position loss (cross-entropy)
        # First reshape fret_logits to [batch_size * pred_time_steps * 6, max_fret+1]
        fret_logits_flat = fret_logits.reshape(-1, self.max_fret + 1)
        
        # Convert targets to class indices and reshape to [batch_size * pred_time_steps * 6]
        fret_targets_flat = fret_targets_ds.long().reshape(-1)
        
        # Only calculate loss for frames where a string is played (fret >= 0)
        mask = fret_targets_flat >= 0
        if mask.sum() > 0:
            fret_loss_fn = nn.CrossEntropyLoss(reduction='mean')
            fret_loss = fret_loss_fn(fret_logits_flat[mask], fret_targets_flat[mask])
        else:
            # No strings played in this batch
            fret_loss = torch.tensor(0.0, device=x.device)
        
        # Total loss (weighted sum)
        total_loss = string_loss + fret_loss
        
        return {
            'string_loss': string_loss,
            'fret_loss': fret_loss,
            'total_loss': total_loss
        }


class TabCRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network for guitar tab transcription.
    
    This model takes audio features (mel spectrograms, chromagrams, etc.)
    and predicts frame-level tablature (string and fret positions).
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        num_strings: int = 6,
        max_fret: int = 24,
        hidden_size: int = 256,
        num_layers: int = 2
    ):
        """
        Initialize the TabCRNN model.
        
        Args:
            input_dim: Input feature dimension (e.g., number of mel bins)
            num_strings: Number of strings on the guitar (default: 6)
            max_fret: Maximum fret number to predict (default: 24)
            hidden_size: Size of recurrent hidden layer
            num_layers: Number of recurrent layers
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_strings = num_strings
        self.max_fret = max_fret
        
        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # RNN for sequence modeling
        self.rnn = nn.GRU(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        # Output layers
        rnn_output_size = hidden_size * 2  # bidirectional
        
        # For each string, predict:
        # 1. Whether it's played (binary)
        # 2. Which fret is played (multi-class, 0 to max_fret)
        self.string_classifiers = nn.ModuleList([
            nn.Linear(rnn_output_size, 1) for _ in range(num_strings)
        ])
        
        self.fret_classifiers = nn.ModuleList([
            nn.Linear(rnn_output_size, max_fret + 1) for _ in range(num_strings)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, time_steps, input_dim]
            
        Returns:
            Tuple of (string_logits, fret_logits):
                - string_logits: Shape [batch_size, time_steps, num_strings]
                - fret_logits: Shape [batch_size, time_steps, num_strings, max_fret+1]
        """
        batch_size, time_steps, _ = x.size()
        
        # Transpose for CNN: [batch_size, input_dim, time_steps]
        x = x.transpose(1, 2)
        
        # Apply CNN
        x = self.cnn(x)  # [batch_size, 64, time_steps/2]
        
        # Transpose back for RNN: [batch_size, time_steps/2, 64]
        x = x.transpose(1, 2)
        
        # Apply RNN
        x, _ = self.rnn(x)  # [batch_size, time_steps/2, hidden_size*2]
        
        # Apply output layers
        string_logits_list = []
        fret_logits_list = []
        
        for i in range(self.num_strings):
            # String classification (played or not)
            string_logits = self.string_classifiers[i](x)  # [batch_size, time_steps/2, 1]
            string_logits_list.append(string_logits)
            
            # Fret classification
            fret_logits = self.fret_classifiers[i](x)  # [batch_size, time_steps/2, max_fret+1]
            fret_logits_list.append(fret_logits)
        
        # Concatenate results
        string_logits = torch.cat(string_logits_list, dim=2)  # [batch_size, time_steps/2, num_strings]
        
        # Stack fret logits for each string
        fret_logits = torch.stack(fret_logits_list, dim=2)  # [batch_size, time_steps/2, num_strings, max_fret+1]
        
        return string_logits, fret_logits