"""Manually implemented recurrent, bidirectional, and attention layers."""

import math
import torch
from torch import nn


class RNNCell(nn.Module):
    """One tanh recurrent update for a batch of time-step inputs."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        # Scale the initial weights to keep activations in a useful range.
        bound = 1.0 / math.sqrt(hidden_size)
        self.hidden_size = hidden_size
        self.W = nn.Parameter(torch.empty(hidden_size, input_size))
        self.U = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.b = nn.Parameter(torch.empty(hidden_size))
        nn.init.uniform_(self.W, -bound, bound)
        nn.init.uniform_(self.U, -bound, bound)
        nn.init.uniform_(self.b, -bound, bound)

    def forward(self, x, h_prev):
        """Return the next hidden state from input and previous state."""
        return torch.tanh(x @ self.W.t() + h_prev @ self.U.t() + self.b)


class LSTMCell(nn.Module):
    """One LSTM update with input, forget, candidate, and output gates."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        bound = 1.0 / math.sqrt(hidden_size)
        self.hidden_size = hidden_size
        self.W = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.U = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.b = nn.Parameter(torch.empty(4 * hidden_size))
        nn.init.uniform_(self.W, -bound, bound)
        nn.init.uniform_(self.U, -bound, bound)
        nn.init.uniform_(self.b, -bound, bound)

    def forward(self, x, h_prev, c_prev):
        """Compute the new hidden and cell states for one time step."""
        # A single matrix multiplication produces all four gate preactivations.
        gates = x @ self.W.t() + h_prev @ self.U.t() + self.b
        i, f, g, o = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        # The additive cell update helps information persist over long sequences.
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class LSTM(nn.Module):
    """Apply the manually implemented LSTM cell across a complete sequence."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = LSTMCell(input_size, hidden_size)

    def forward(self, x):
        """Return a hidden state for every input time step."""
        batch_size = x.size(0)
        # Start each independent sequence with zero hidden and memory states.
        h = x.new_zeros(batch_size, self.hidden_size)
        c = x.new_zeros(batch_size, self.hidden_size)
        outputs = []
        for t in range(x.size(1)):
            h, c = self.cell(x[:, t, :], h, c)
            outputs.append(h)
        return torch.stack(outputs, dim=1)


class BiLSTM(nn.Module):
    """Concatenate aligned forward and backward sequence representations."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.forward_lstm = LSTM(input_size, hidden_size)
        self.backward_lstm = LSTM(input_size, hidden_size)

    def forward(self, x):
        """Encode each time step using both past and future context."""
        forward_states = self.forward_lstm(x)
        # Reverse the sequence for the backward pass, then restore time order.
        backward_states = self.backward_lstm(torch.flip(x, dims=[1]))
        backward_states = torch.flip(backward_states, dims=[1])
        return torch.cat((forward_states, backward_states), dim=-1)


class Attention(nn.Module):
    """Learn a normalized importance weight for each sequence time step."""

    def __init__(self, input_size, attention_size=64):
        super().__init__()
        bound = 1.0 / math.sqrt(attention_size)
        self.Wa = nn.Parameter(torch.empty(attention_size, input_size))
        self.ba = nn.Parameter(torch.empty(attention_size))
        self.v = nn.Parameter(torch.empty(attention_size))
        nn.init.uniform_(self.Wa, -bound, bound)
        nn.init.uniform_(self.ba, -bound, bound)
        nn.init.uniform_(self.v, -bound, bound)

    def forward(self, h):
        """Return a weighted sequence summary and its per-step attention."""
        e = torch.tanh(h @ self.Wa.t() + self.ba) @ self.v
        # Softmax makes weights non-negative and sum to one per example.
        alpha = torch.softmax(e, dim=1)
        context = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)
        return context, alpha
