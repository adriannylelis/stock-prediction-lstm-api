"""
Modelo LSTM para previsão de preços de ações.

Implementação simplificada para carregar os pesos do modelo treinado.
"""

import torch
import torch.nn as nn


class StockLSTM(nn.Module):
    """
    LSTM model for stock price prediction.
    
    Architecture:
        - LSTM layer(s) with configurable hidden_size and num_layers
        - Dropout for regularization (optional)
        - Fully connected layer for final prediction
    
    Args:
        input_size (int): Number of input features (1 for univariate).
        hidden_size (int): Number of LSTM hidden units.
        num_layers (int): Number of LSTM layers.
        dropout (float): Dropout probability (0.0-1.0).
    
    Example:
        >>> model = StockLSTM(input_size=1, hidden_size=16, num_layers=1)
        >>> x = torch.randn(32, 60, 1)  # batch_size=32, seq_len=60, features=1
        >>> output = model(x)  # shape: (32, 1)
    """
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 16,
        num_layers: int = 1,
        dropout: float = 0.0
    ):
        super(StockLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_prob = dropout
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True  # Input shape: (batch, seq, features)
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_size).
        
        Returns:
            Tensor: Output predictions of shape (batch_size, 1).
        """
        # LSTM forward pass
        # output shape: (batch_size, seq_len, hidden_size)
        # h_n shape: (num_layers, batch_size, hidden_size)
        output, (h_n, c_n) = self.lstm(x)
        
        # Use last timestep output
        # Shape: (batch_size, hidden_size)
        last_output = output[:, -1, :]
        
        # Fully connected layer
        # Shape: (batch_size, 1)
        prediction = self.fc(last_output)
        
        return prediction
