import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super().__init__()
        # Bottleneck design: 1x1 (reduce) -> 3x3 (process) -> 1x1 (expand)
        mid_channels = out_channels // 2
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, mid_channels, kernel_size=3, 
                      padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
            
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))

class TemporalCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        # Initial Feature Extraction (The Stem)
        self.stem = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )

        # Deep Residual Backbone
        self.layers = nn.Sequential(
            # Stage 1: Local patterns
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            
            # Stage 2: Mid-range patterns (increasing width)
            ResidualBlock(128, 256),
            ResidualBlock(256, 256, dilation=2), # Dilated to see longer sequences
            
            # Stage 3: High-level features
            ResidualBlock(256, 512),
            ResidualBlock(512, 512, dilation=4), # Wide temporal context
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # x shape: (Batch, Time, Dim)
        x = x.transpose(1, 2) # (Batch, Dim, Time)
        
        x = self.stem(x)
        x = self.layers(x)
        
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)
