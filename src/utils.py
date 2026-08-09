import torch
from torch.nn.utils.rnn import pad_sequence


class EarlyStopper:
    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss, model, model_name, results):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.results = results
            save_checkpoint(model, f"{model_name}.pth.tar")
        elif validation_loss > (self.min_validation_loss):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def collate_fn(batch):
    sequences, labels = zip(*batch)
    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded_seqs, torch.tensor(labels, dtype=torch.float32)


def save_checkpoint(model, filename):
    checkpoint = {"state_dict": model.state_dict()}
    torch.save(checkpoint, filename)


def load_checkpoint(filename, model, device):
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
