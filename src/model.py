import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, model_name, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        if model_name == "GRU":
            self.recurrent_layer = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        elif model_name == "LSTM":
            self.recurrent_layer = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        elif model_name == "RNN":
            self.recurrent_layer = nn.RNN(embedding_dim, hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.recurrent_layer(x)
        out, _ = torch.max(out, dim=1)
        return self.fc(out)
