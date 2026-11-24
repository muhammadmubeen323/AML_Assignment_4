"""Focused shape and invariant checks for the manually built HAR model."""

import sys
import torch

sys.path.insert(0, "src")

from layers import RNNCell, LSTMCell, LSTM, BiLSTM, Attention
from model import HARNet


def run_checks():
    """Check layer outputs, attention normalization, and model size."""
    torch.manual_seed(0)
    x = torch.randn(3, 128, 9)

    # A recurrent cell consumes one time slice and returns one hidden vector.
    rnn = RNNCell(9, 64)
    h = torch.zeros(3, 64)
    h = rnn(x[:, 0, :], h)
    assert h.shape == (3, 64)

    # Both LSTM states must keep the requested batch and hidden dimensions.
    lstm_cell = LSTMCell(9, 64)
    h = torch.zeros(3, 64)
    c = torch.zeros(3, 64)
    h, c = lstm_cell(x[:, 0, :], h, c)
    assert h.shape == (3, 64) and c.shape == (3, 64)

    # Sequence and bidirectional layers preserve time and concatenate directions.
    lstm = LSTM(9, 64)
    assert lstm(x).shape == (3, 128, 64)

    bilstm = BiLSTM(9, 64)
    assert bilstm(x).shape == (3, 128, 128)

    # Attention weights form a probability distribution over the 128 steps.
    attention = Attention(128, 64)
    context, weights = attention(bilstm(x))
    assert context.shape == (3, 128)
    assert weights.shape == (3, 128)
    assert torch.all(weights >= 0)
    assert torch.allclose(weights.sum(dim=1), torch.ones(3), atol=1e-6)

    # The classifier must emit six logits and retain the expected parameter count.
    model = HARNet()
    logits, weights = model(x)
    assert logits.shape == (3, 6)
    assert weights.shape == (3, 128)
    assert sum(p.numel() for p in model.parameters()) == 46982

    # Fresh-model cross entropy should be near the uniform six-class baseline.
    fresh_model = HARNet()
    fresh_logits, _ = fresh_model(torch.randn(64, 128, 9))
    fresh_loss = torch.nn.functional.cross_entropy(
        fresh_logits,
        torch.zeros(64, dtype=torch.long),
    )
    assert abs(fresh_loss.item() - 1.791759) < 0.12

    print("PASS: shapes, parameter count, positive normalized attention, initial loss")


if __name__ == "__main__":
    # Run the checks when invoked with `python tests.py`.
    run_checks()
