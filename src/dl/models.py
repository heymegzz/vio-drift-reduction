"""Small causal regressors. No normalization pools over future timesteps."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, inputs, hidden=32, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(hidden, max(8, hidden // 2)), nn.ReLU())
        self.head = nn.Linear(max(8, hidden // 2), 3)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        return self.head(self.encoder(x))


class CausalConv(nn.Module):
    def __init__(self, inputs, outputs, dilation):
        super().__init__()
        self.left = 2 * dilation
        self.conv = nn.Conv1d(inputs, outputs, 3, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.left, 0)))


class ResidualBlock(nn.Module):
    def __init__(self, inputs, hidden, dilation, dropout):
        super().__init__()
        self.layers = nn.Sequential(CausalConv(inputs, hidden, dilation), nn.ReLU(), nn.Dropout(dropout),
                                    CausalConv(hidden, hidden, dilation), nn.ReLU(), nn.Dropout(dropout))
        self.skip = nn.Identity() if inputs == hidden else nn.Conv1d(inputs, hidden, 1)

    def forward(self, x):
        return F.relu(self.layers(x) + self.skip(x))


class TCN(nn.Module):
    def __init__(self, inputs, hidden=16, dropout=0.1, window=100):
        super().__init__()
        n_blocks = max(1, math.ceil(math.log2((window - 1) / 4 + 1)))
        self.receptive_field = 1 + 4 * (2 ** n_blocks - 1)
        self.encoder = nn.Sequential(*[
            ResidualBlock(inputs if i == 0 else hidden, hidden, 2 ** i, dropout)
            for i in range(n_blocks)
        ])
        self.head = nn.Linear(hidden, 3)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        return self.head(self.encoder(x.transpose(1, 2))[:, :, -1])


class GRU(nn.Module):
    def __init__(self, inputs, hidden=32, dropout=0.1):
        super().__init__()
        # Last channel is validity, consumed here rather than learned as a sensor.
        self.encoder = nn.GRU(inputs - 1, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 3)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        lengths = x[:, :, -1].sum(dim=1).long()
        if (lengths < 1).any():
            raise ValueError("A sequence must include its current observation")
        # Move left-padded valid samples to the start before packing.
        t = x.shape[1]
        index = (torch.arange(t, device=x.device)[None, :] + t - lengths[:, None]).clamp(max=t - 1)
        values = x[:, :, :-1].gather(1, index[:, :, None].expand(-1, -1, x.shape[2] - 1))
        packed = nn.utils.rnn.pack_padded_sequence(values, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.encoder(packed)
        return self.head(self.dropout(h[-1]))


def build_model(config, inputs):
    name = config["model"]
    common = {"inputs": inputs, "hidden": config.get("hidden", 32), "dropout": config.get("dropout", .1)}
    if name in {"mlp", "raw_mlp"}:
        return MLP(**common)
    if name == "tcn":
        return TCN(**common, window=config["window"])
    if name == "gru":
        return GRU(**common)
    raise ValueError(f"Unknown model: {name}")
