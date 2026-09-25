import torch
import torch.nn as nn

from checkpoint_meta import SUPPORTED_ARCHITECTURES


class TemporalCNN(nn.Module):
    """
    Small temporal classifier — no LSTM, no learned attention pooling.
    Global average pooling gives translation-invariant features and has
    zero learnable parameters, which matters a lot when you only have a
    handful of samples per class: it structurally can't memorize *where*
    in the clip something happened, only *what pattern* occurred.

    Input:  (B, T, D)
    Output: (B, C)
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden: int = 192,          # slightly smaller than 128; drop to 64 if still overfitting
        conv_dropout: float = 0.25,  # NEW: dropout between conv blocks, not just before FC
        head_dropout: float = 0.3,   # NEW: a bit stronger than before, since FC->classes is the
                                      # single largest, most overfit-prone layer here
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_dim, hidden, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(conv_dropout),

            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(conv_dropout),

            nn.Conv1d(hidden, hidden * 2, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(conv_dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(head_dropout)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(self.dropout(x))


class CNNBiLSTM(nn.Module):
    """
    OPTIONAL experiment (train.py --architecture cnn_bilstm). NOT the
    default -- see project brief section 9/51: BiLSTM is only worth using
    in production if it demonstrates a meaningful Top-1/macro-F1
    improvement over TemporalCNN without an unacceptable latency or
    parameter-count cost, measured via the comparison table in
    README.md's experiment results (RULE 2: don't replace TemporalCNN
    without evidence).

    Architecture: the SAME conv-based temporal feature extractor as
    TemporalCNN (identical block-for-block, so any accuracy difference
    between the two architectures is attributable to the BiLSTM addition,
    not to also changing the conv front-end), followed by a bidirectional
    LSTM over the conv features, then average pooling over time and a
    linear classifier.

    Input:  (B, T, D)
    Output: (B, C)
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden: int = 192,
        conv_dropout: float = 0.25,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        head_dropout: float = 0.3,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_dim, hidden, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(conv_dropout),

            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(conv_dropout),

            nn.Conv1d(hidden, hidden * 2, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(conv_dropout),
        )
        self.lstm = nn.LSTM(
            input_size=hidden * 2,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(head_dropout)
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)          # (B, D, T)
        x = self.features(x)           # (B, hidden*2, T)
        x = x.transpose(1, 2)          # (B, T, hidden*2) -- LSTM wants time-major-ish (batch_first)
        x, _ = self.lstm(x)            # (B, T, lstm_hidden*2)
        x = x.transpose(1, 2)          # (B, lstm_hidden*2, T) -- back for pooling
        x = self.pool(x).squeeze(-1)   # (B, lstm_hidden*2)
        return self.fc(self.dropout(x))


def build_model(architecture: str, input_dim: int, num_classes: int, **kwargs) -> nn.Module:
    """Factory used by train.py/infer.py/api.py so architecture selection
    lives in exactly one place. `architecture` defaults are NOT applied
    here -- callers pass an explicit value (train.py's --architecture flag
    defaults to DEFAULT_ARCHITECTURE, imported from checkpoint_meta so
    train.py, this module, and api.py all agree on what "default" means)."""
    if architecture == "temporal_cnn":
        return TemporalCNN(input_dim, num_classes, **kwargs)
    if architecture == "cnn_bilstm":
        return CNNBiLSTM(input_dim, num_classes, **kwargs)
    raise ValueError(f"Unknown architecture {architecture!r}; expected one of {SUPPORTED_ARCHITECTURES}")