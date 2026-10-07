"""Required models: a CNN and an RNN (BiLSTM)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """Multi-width 1D CNN (Kim, 2014) with a second conv stage.

    Parallel convolutions of widths `kernel_sizes` detect local n-gram patterns
    (for char input: things like "**", "\\n- ", "—", "Excellent"). A second conv
    widens the receptive field so patterns spanning ~20+ tokens (e.g. a bullet
    line followed by a bolded header) can be detected. Global max-pooling asks
    "did this pattern occur anywhere?", global mean-pooling asks "how often?".
    """

    def __init__(self, vocab_size, n_classes, emb_dim=64, n_filters=128,
                 kernel_sizes=(3, 5, 7), dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(emb_dim, n_filters, k, padding=k // 2) for k in kernel_sizes)
        c = n_filters * len(kernel_sizes)
        self.conv2 = nn.Conv1d(c, c, 5, padding=2)
        self.norm1, self.norm2 = nn.BatchNorm1d(c), nn.BatchNorm1d(c)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * c, n_classes)

    def forward(self, x, lengths):
        mask = (x != 0).unsqueeze(1)                       # B,1,T
        h = self.emb(x).transpose(1, 2)                    # B,E,T
        h = F.relu(self.norm1(torch.cat([conv(h) for conv in self.convs], 1)))
        h = F.max_pool1d(h, 2, ceil_mode=True)
        m = F.max_pool1d(mask.float(), 2, ceil_mode=True).bool()
        h = F.relu(self.norm2(self.conv2(h))) * m
        h_max = h.masked_fill(~m, -1e4).max(2).values
        h_mean = h.sum(2) / m.sum(2).clamp(min=1)
        return self.fc(self.drop(torch.cat([h_max, h_mean], 1)))


class BiLSTMAttn(nn.Module):
    """Bidirectional LSTM with additive attention pooling.

    The LSTM reads the sequence in both directions, so each position's state
    carries context from the whole response (useful for document-level habits
    like "opens with praise, ends with an offer to help"). Attention pooling
    learns which positions matter; its weights are inspected in RQ4.
    """

    def __init__(self, vocab_size, n_classes, emb_dim=128, hidden=128, layers=1, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, layers, batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.attn = nn.Sequential(nn.Linear(2 * hidden, 64), nn.Tanh(), nn.Linear(64, 1))
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden, n_classes)
        self.last_attention = None

    def forward(self, x, lengths):
        e = self.drop(self.emb(x))
        packed = nn.utils.rnn.pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        scores = self.attn(out).squeeze(-1).masked_fill(x == 0, -1e4)
        w = torch.softmax(scores, 1)
        self.last_attention = w.detach()
        pooled = (w.unsqueeze(-1) * out).sum(1)
        return self.fc(self.drop(pooled))


def build_model(name, vocab_size, n_classes, **kw):
    return {"cnn": TextCNN, "lstm": BiLSTMAttn}[name](vocab_size, n_classes, **kw)
