import torch
import torch.nn as nn


class GRU_Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.gru(x)
        out, _ = torch.max(out, dim=1)
        return self.fc(out)


class LSTM_Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out, _ = torch.max(out, dim=1)
        return self.fc(out)


class RNN_Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out, _ = torch.max(out, dim=1)
        return self.fc(out)


def get_model(model_name, vocab_size, embedding_dim, hidden_dim):
    if model_name == "GRU":
        return GRU_Model(vocab_size, embedding_dim, hidden_dim)
    elif model_name == "LSTM":
        return LSTM_Model(vocab_size, embedding_dim, hidden_dim)
    elif model_name == "RNN":
        return RNN_Model(vocab_size, embedding_dim, hidden_dim)
