"""Attention-BiLSTM classifier for six UCI HAR activities."""

import torch
from torch import nn
from layers import BiLSTM, Attention


class HARNet(nn.Module):
    """Encode sensor windows, attend over time, and predict activity logits."""

    def __init__(self, input_size=9, hidden_size=64, num_classes=6):
        super().__init__()
        # Bidirectional states have twice the configured hidden width.
        self.bilstm = BiLSTM(input_size, hidden_size)
        self.attention = Attention(2 * hidden_size, 64)
        self.classifier = nn.Linear(2 * hidden_size, num_classes)

    def forward(self, x):
        """Return class logits and the temporal attention distribution."""
        states = self.bilstm(x)
        context, alpha = self.attention(states)
        logits = self.classifier(context)
        return logits, alpha
