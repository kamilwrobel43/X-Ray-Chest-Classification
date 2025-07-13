import torch
from torch import nn

class ConvBaseline(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBaseline, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(6, 12, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(12, 24, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(24 * 26 * 26, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, out_channels)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
