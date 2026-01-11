"""
LSTM Model para previsão de preços de ações.
Arquitetura: 3 camadas LSTM com Dropout e suporte a embeddings de tickers.
"""

import torch
import torch.nn as nn
from typing import Optional


class StockLSTM(nn.Module):
    """
    LSTM para previsão de séries temporais de ações com suporte a múltiplos tickers.
    
    Arquitetura:
    - Embedding layer para tickers (opcional)
    - 3 camadas LSTM bidirecionais
    - Dropout entre camadas
    - Camada fully connected de saída
    
    Args:
        num_tickers: Número de tickers diferentes (para embedding)
        num_features: Número de features por timestep
        embedding_dim: Dimensão do embedding de ticker
        hidden_size: Unidades na camada oculta (default: 100)
        num_layers: Número de camadas LSTM (default: 3)
        dropout: Taxa de dropout (default: 0.3)
        bidirectional: Se True, usa LSTM bidirecional
        
    Legacy support (backward compatibility):
        input_size: Alias para num_features (mantido para compatibilidade)
        output_size: Sempre 1 (preço)
    """
    
    def __init__(
        self,
        num_tickers: int = 1,
        num_features: Optional[int] = None,
        embedding_dim: int = 8,
        hidden_size: int = 100,
        num_layers: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = False,
        # Legacy parameters
        input_size: Optional[int] = None,
        output_size: int = 1
    ):
        super(StockLSTM, self).__init__()
        
        # Handle legacy input_size parameter
        if num_features is None and input_size is not None:
            num_features = input_size
        elif num_features is None:
            raise ValueError("Either num_features or input_size must be provided")
        
        self.num_tickers = num_tickers
        self.num_features = num_features
        self.input_size = num_features  # Legacy alias
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.output_size = output_size
        self.bidirectional = bidirectional
        
        # Multiplicador para bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Ticker embedding (if num_tickers > 1)
        if num_tickers > 1 and embedding_dim > 0:
            self.ticker_embedding = nn.Embedding(num_tickers, embedding_dim)
            lstm_input_size = num_features + embedding_dim
        else:
            self.ticker_embedding = None
            lstm_input_size = num_features
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Dropout adicional
        self.dropout = nn.Dropout(dropout)
        
        # Camada de saída
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)
    
    def forward(
        self, 
        x: torch.Tensor,
        ticker_ids: Optional[torch.Tensor] = None,
        hidden: Optional[tuple] = None
    ) -> tuple:
        """
        Forward pass do modelo.
        
        Args:
            x: Tensor de entrada (batch_size, seq_len, num_features)
            ticker_ids: IDs dos tickers (batch_size,) - opcional
            hidden: Estado oculto inicial (opcional)
        
        Returns:
            output: Predição (batch_size, output_size)
            hidden: Novo estado oculto
        """
        # Add ticker embeddings if available
        if self.ticker_embedding is not None and ticker_ids is not None:
            # ticker_emb: (batch_size, embedding_dim)
            ticker_emb = self.ticker_embedding(ticker_ids)
            # Expand to sequence length: (batch_size, seq_len, embedding_dim)
            ticker_emb = ticker_emb.unsqueeze(1).expand(-1, x.size(1), -1)
            # Concatenate with features
            x = torch.cat([x, ticker_emb], dim=-1)
        
        # LSTM forward
        # lstm_out: (batch_size, seq_len, hidden_size * num_directions)
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Pegar apenas o último timestep
        # last_output: (batch_size, hidden_size * num_directions)
        last_output = lstm_out[:, -1, :]
        
        # Aplicar dropout
        last_output = self.dropout(last_output)
        
        # Camada fully connected
        # output: (batch_size, output_size)
        output = self.fc(last_output)
        
        return output, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device = None) -> tuple:
        """
        Inicializa o estado oculto do LSTM.
        
        Args:
            batch_size: Tamanho do batch
            device: Dispositivo (CPU/CUDA) - opcional
        
        Returns:
            Tupla (h0, c0) com estados ocultos inicializados
        """
        if device is None:
            device = next(self.parameters()).device
            
        h0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ).to(device)
        
        c0 = torch.zeros(
            self.num_layers * self.num_directions,
            batch_size,
            self.hidden_size
        ).to(device)
        
        return (h0, c0)
    
    def get_config(self) -> dict:
        """Retorna configuração do modelo."""
        return {
            'num_tickers': self.num_tickers,
            'num_features': self.num_features,
            'embedding_dim': self.embedding_dim,
            'input_size': self.input_size,  # Legacy
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout_rate,
            'output_size': self.output_size,
            'bidirectional': self.bidirectional
        }
    
    def count_parameters(self) -> int:
        """Conta o número total de parâmetros treináveis."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(
    num_tickers: int,
    num_features: int,
    embedding_dim: int = 8,
    hidden_size: int = 100,
    num_layers: int = 3,
    dropout: float = 0.3,
    bidirectional: bool = False,
    device: str = "cpu"
) -> StockLSTM:
    """Factory function to create and initialize a StockLSTM model.
    
    Args:
        num_tickers: Number of different stock tickers (for embedding layer)
        num_features: Number of input features per timestep
        embedding_dim: Dimensionality of ticker embeddings
        hidden_size: Number of hidden units in LSTM layers
        num_layers: Number of stacked LSTM layers
        dropout: Dropout probability for regularization
        bidirectional: Whether to use bidirectional LSTM
        device: Device to place the model on ('cpu' or 'cuda')
    
    Returns:
        Initialized StockLSTM model on specified device
    
    Example:
        >>> model = create_model(num_tickers=1, num_features=18, device='cpu')
        >>> print(model)
    """
    model = StockLSTM(
        num_tickers=num_tickers,
        num_features=num_features,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional
    )
    model = model.to(device)
    return model

