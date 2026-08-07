import torch
import torch.nn as nn


class LSTM_Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, 1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out, _ = torch.max(out, dim=1)
        return self.fc(out)


def get_model(model_name, vocab_size=None):
    if model_name == "LSTM":
        return LSTM_Model(vocab_size=vocab_size)
