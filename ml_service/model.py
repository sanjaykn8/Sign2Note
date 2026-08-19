import torch
import torch.nn as nn


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
        hidden: int = 96,          # slightly smaller than 128; drop to 64 if still overfitting
        conv_dropout: float = 0.15,  # NEW: dropout between conv blocks, not just before FC
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
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(head_dropout)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(self.dropout(x))