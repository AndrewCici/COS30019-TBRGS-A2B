import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. LSTM Model
# ---------------------------------------------------------------------------
class TrafficLSTM(nn.Module):
    """
    Two-layer stacked LSTM with dropout regularisation and a fully-connected
    output head.

    Input  : (batch, seq_len, input_size)   [batch_first=True]
    Output : (batch, 1)
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)          # (batch, seq_len, hidden_size)
        out = self.dropout(out[:, -1, :])  # take last time-step
        return self.fc(out)            # (batch, output_size)


# ---------------------------------------------------------------------------
# 2. GRU Model
# ---------------------------------------------------------------------------
class TrafficGRU(nn.Module):
    """
    Two-layer stacked GRU with dropout regularisation and a fully-connected
    output head.

    Input  : (batch, seq_len, input_size)   [batch_first=True]
    Output : (batch, 1)
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)               # (batch, seq_len, hidden_size)
        out = self.dropout(out[:, -1, :])  # last time-step
        return self.fc(out)                # (batch, output_size)


# ---------------------------------------------------------------------------
# 3. Transformer Model
# ---------------------------------------------------------------------------
class _PositionalEncoding(nn.Module):
    """
    Classic sinusoidal positional encoding (Vaswani et al., 2017).
    Adds position information to token embeddings before the encoder stack.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)                  # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                                 # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TrafficTransformer(nn.Module):
    """
    Time-series Transformer for univariate traffic volume forecasting.

    Architecture
    ------------
    Input projection  → Positional Encoding → TransformerEncoder (N layers)
    → mean pooling over sequence → FC output head

    Input  : (batch, seq_len, input_size)
    Output : (batch, 1)
    """

    def __init__(
        self,
        input_size: int = 1,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        output_size: int = 1,
    ):
        super().__init__()

        # Project raw feature dimension to d_model
        self.input_projection = nn.Linear(input_size, d_model)

        self.pos_encoding = _PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,          # (batch, seq, d_model)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        x = self.input_projection(x)           # (batch, seq_len, d_model)
        x = self.pos_encoding(x)               # (batch, seq_len, d_model)
        x = self.transformer_encoder(x)        # (batch, seq_len, d_model)
        x = x.mean(dim=1)                      # global average pooling over seq
        x = self.dropout(x)
        return self.fc(x)                      # (batch, output_size)
